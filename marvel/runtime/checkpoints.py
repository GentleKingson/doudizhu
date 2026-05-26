from __future__ import annotations

import os
import re
import shutil
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from marvel.model.checkpointable import load_checkpoint_payload_file, save_checkpoint_payload_file
from marvel.utils import unwrap_distributed_module

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WANDB_CACHE_DIR: Path = _REPO_ROOT / ".cache" / "wandb"


@dataclass(frozen = True)
class WandbCheckpointRef:
    entity: str
    project: str
    run_id: str
    iteration: int | None


@dataclass
class CheckpointState:
    """Binds payload keys directly to models, optimizers, and scalar values.

    One flat dict. Duck-typed save/load:
    - nn.Module with ``build_checkpoint_payload`` → model save/load
    - nn.Module without → ``state_dict`` / ``load_state_dict``
    - Object with ``state_dict`` / ``load_state_dict`` → optimizer
    - Otherwise → scalar value (int, float, dict, etc.)
    """

    entries: dict[str, Any] = field(default_factory = dict)
    iteration_key: str = "iteration"

    def __getitem__(self, key: str) -> Any:
        return self.entries[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.entries[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str, default: Any = None) -> Any:
        return self.entries.get(key, default)

    @property
    def iteration(self) -> int:
        value = self.entries.get(self.iteration_key)
        if value is None:
            return 0
        return int(value)


class CheckpointStore:
    """Path/reference and payload-file I/O for checkpoints."""

    _CKPT_FILE_RE = re.compile(r"^(?:checkpoints/)?ckpt_(?P<iteration>\d+)\.pt$")

    @staticmethod
    def parse_checkpoint_locator(locator: str | os.PathLike[str]) -> Path | WandbCheckpointRef:
        text = os.fspath(locator)
        if text.startswith("wandb:"):
            spec = text[len("wandb:") :].strip()
            run_spec = spec
            iteration: int | None = None
            if ":" in spec:
                run_spec, iteration_raw = spec.rsplit(":", 1)
                try:
                    iteration = int(iteration_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid iteration '{iteration_raw}' in W&B checkpoint reference."
                    ) from exc
                if iteration <= 0:
                    raise ValueError("iteration must be a positive integer for W&B checkpoint references.")

            parts = [part for part in run_spec.split("/") if part]
            if len(parts) != 3:
                raise ValueError(
                    "W&B checkpoint reference must be in the form "
                    "'wandb:<entity>/<project>/<run_id>[:<iteration>]'."
                )
            entity, project, run_id = parts
            return WandbCheckpointRef(
                entity = entity,
                project = project,
                run_id = run_id,
                iteration = iteration,
            )
        return Path(text).expanduser()

    @classmethod
    def _parse_checkpoint_iteration_from_name(cls, name: str) -> int | None:
        match = cls._CKPT_FILE_RE.search(str(name).strip())
        if match is None:
            return None
        return int(match.group("iteration"))

    @classmethod
    def _fetch_latest_wandb_iteration(cls, ref: WandbCheckpointRef) -> int:
        try:
            import wandb  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "wandb is required to resolve the latest checkpoint from a W&B run reference. "
                "Install wandb or provide an explicit ':<iteration>' in the checkpoint reference."
            ) from exc

        api = wandb.Api()
        run = api.run(f"{ref.entity}/{ref.project}/{ref.run_id}")

        latest_iteration = -1
        found_non_positive = False
        for run_file in run.files():
            step = cls._parse_checkpoint_iteration_from_name(str(getattr(run_file, "name", "")))
            if step is None:
                continue
            if step <= 0:
                found_non_positive = True
                continue
            latest_iteration = max(latest_iteration, int(step))

        if latest_iteration < 0:
            if found_non_positive:
                raise FileNotFoundError(
                    f"W&B run {ref.entity}/{ref.project}/{ref.run_id} only contains non-positive ckpt_*.pt files. "
                    "Checkpoint iterations must be positive integers."
                )
            raise FileNotFoundError(
                f"W&B run {ref.entity}/{ref.project}/{ref.run_id} does not contain any ckpt_*.pt files."
            )
        return latest_iteration

    @staticmethod
    def _download_wandb_checkpoint(
        ref: WandbCheckpointRef,
        *,
        checkpoint_relpath: str,
        download_root: Path,
    ) -> Path:
        if ref.iteration is None:
            raise ValueError("_download_wandb_checkpoint requires an explicit checkpoint iteration.")

        try:
            import wandb  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "wandb is required to download checkpoints from W&B. Install wandb or pass a local path."
            ) from exc

        api = wandb.Api()
        run = api.run(f"{ref.entity}/{ref.project}/{ref.run_id}")

        download_root.mkdir(parents = True, exist_ok = True)
        run_files = list(run.files())
        file_by_name = {str(getattr(file, "name", "")): file for file in run_files}
        for remote_name in (checkpoint_relpath, Path(checkpoint_relpath).name):
            wandb_file = file_by_name.get(str(remote_name))
            if wandb_file is None:
                continue
            wandb_file.download(root = str(download_root), replace = True)
            downloaded = download_root / wandb_file.name
            if downloaded.exists():
                return downloaded

        available = [getattr(file, "name", "<unknown>") for file in run_files]
        raise FileNotFoundError(
            f"W&B run {ref.entity}/{ref.project}/{ref.run_id} does not contain '{checkpoint_relpath}'. "
            f"Available files: {sorted(available)}."
        )

    @staticmethod
    def upload_checkpoint_to_wandb(checkpoint_path: Path) -> None:
        try:
            import wandb  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "wandb is required to upload checkpoints to W&B. Install wandb or disable W&B."
            ) from exc

        run_dir = getattr(getattr(wandb, "run", None), "dir", None)
        if run_dir:
            dst_dir = Path(run_dir) / "checkpoints"
            dst_dir.mkdir(parents = True, exist_ok = True)
            dst_path = dst_dir / checkpoint_path.name
            shutil.copy2(checkpoint_path, dst_path)
            wandb.save(str(dst_path), base_path = str(Path(run_dir)))
            return
        wandb.save(str(checkpoint_path))

    @classmethod
    def resolve_checkpoint_file_path(
        cls,
        locator: str | os.PathLike[str],
        *,
        cache_dir: str | os.PathLike[str] = DEFAULT_WANDB_CACHE_DIR,
    ) -> Path:
        parsed = cls.parse_checkpoint_locator(locator)
        if isinstance(parsed, Path):
            return parsed

        if parsed.iteration is None:
            parsed = WandbCheckpointRef(
                entity = parsed.entity,
                project = parsed.project,
                run_id = parsed.run_id,
                iteration = cls._fetch_latest_wandb_iteration(parsed),
            )

        cache_root_dir = Path(os.fspath(cache_dir)).expanduser()
        filename = f"ckpt_{parsed.iteration}.pt"
        checkpoint_relpath = str(Path("checkpoints") / filename)
        cache_root = cache_root_dir / parsed.entity / parsed.project / parsed.run_id
        dst_path = cache_root / "checkpoints" / filename
        if dst_path.exists():
            return dst_path

        downloaded = cls._download_wandb_checkpoint(
            parsed,
            checkpoint_relpath = checkpoint_relpath,
            download_root = cache_root,
        )
        if downloaded != dst_path:
            dst_path.parent.mkdir(parents = True, exist_ok = True)
            shutil.copy2(downloaded, dst_path)
        return dst_path

    @staticmethod
    def load_checkpoint_payload_file(
        checkpoint_path: str | Path,
        *,
        device: torch.device,
    ) -> Mapping[str, Any]:
        return load_checkpoint_payload_file(checkpoint_path, device = device)

    @staticmethod
    def save_checkpoint_payload_file(
        payload: Mapping[str, Any],
        checkpoint_path: str | Path,
    ) -> Path:
        return save_checkpoint_payload_file(payload, checkpoint_path)


def _is_stateful_object(obj: Any) -> bool:
    """Return True if obj has state_dict/load_state_dict (nn.Module or Optimizer)."""
    return hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict"))


def load_checkpoint_into_state(
    checkpoint: Mapping[str, Any],
    state: CheckpointState,
) -> None:
    """Populate *state* from a loaded checkpoint payload using duck-typed dispatch."""
    for key, obj in list(state.entries.items()):
        if obj is None:
            continue
        if key not in checkpoint:
            continue

        base = unwrap_distributed_module(obj) if isinstance(obj, torch.nn.Module) else obj

        # nn.Module with load_checkpoint_payload (Agent, CentralCritic, etc.)
        load_payload = getattr(base, "load_checkpoint_payload", None)
        if isinstance(obj, torch.nn.Module) and callable(load_payload):
            load_payload(checkpoint, checkpoint_state_key = key)
            continue

        # nn.Module without load_checkpoint_payload — use load_state_dict
        if isinstance(obj, torch.nn.Module):
            load_state_dict_fn = getattr(base, "load_state_dict", None)
            if callable(load_state_dict_fn):
                incompatible_keys = load_state_dict_fn(checkpoint[key], strict = False)
                missing_keys = list(getattr(incompatible_keys, "missing_keys", ()))
                if missing_keys:
                    raise RuntimeError(
                        f"Error(s) in loading state_dict for '{key}': "
                        f"Missing key(s) in state_dict: {missing_keys}."
                    )
            continue

        # Optimizer / MultiOptimizer — has load_state_dict
        load_sd = getattr(obj, "load_state_dict", None)
        if callable(load_sd) and _is_stateful_object(obj):
            load_sd(checkpoint[key])
            continue

        # Scalar value
        state.entries[key] = checkpoint[key]


def build_checkpoint_payload_from_state(state: CheckpointState) -> dict[str, Any]:
    """Build a flat checkpoint payload dict from *state* using duck-typed dispatch."""
    payload: dict[str, Any] = {}

    for key, obj in state.entries.items():
        if obj is None:
            continue
        base = unwrap_distributed_module(obj) if isinstance(obj, torch.nn.Module) else obj

        # nn.Module with build_checkpoint_payload (Agent, CentralCritic, etc.)
        build_fn = getattr(base, "build_checkpoint_payload", None)
        if isinstance(obj, torch.nn.Module) and callable(build_fn):
            field_payload = build_fn(checkpoint_state_key = key)
            if not isinstance(field_payload, Mapping):
                raise TypeError(
                    f"build_checkpoint_payload() for '{key}' must return a mapping."
                )
            state_value = field_payload.get(key)
            if state_value is None:
                raise KeyError(
                    f"build_checkpoint_payload() for '{key}' must include '{key}' in the returned mapping."
                )
            payload[key] = state_value
            continue

        # nn.Module without build_checkpoint_payload — use state_dict
        if isinstance(obj, torch.nn.Module):
            payload[key] = base.state_dict()
            continue

        # Optimizer / MultiOptimizer — has state_dict
        if _is_stateful_object(obj):
            payload[key] = obj.state_dict()
            continue

        # Scalar value
        payload[key] = obj

    return payload


class CheckpointManager:
    """Checkpoint payload save/restore orchestration."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        config_payload: Any = None,
        config_payload_key: str = "train_config",
        wandb_disabled: bool = True,
        store: CheckpointStore | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_payload = config_payload
        self.config_payload_key = str(config_payload_key)
        self.wandb_disabled = bool(wandb_disabled)
        self.store = CheckpointStore() if store is None else store

    @staticmethod
    def _default_checkpoint_state_key(
        model_cls: type[torch.nn.Module],
        checkpoint_state_key: str | None,
    ) -> str:
        if checkpoint_state_key is not None and str(checkpoint_state_key).strip():
            return str(checkpoint_state_key)
        key = getattr(model_cls, "checkpoint_state_key", None)
        if key is None or not str(key).strip():
            raise ValueError(
                f"{model_cls.__name__} must define a non-empty checkpoint_state_key "
                "or checkpoint_state_key must be provided explicitly."
            )
        return str(key)

    @classmethod
    def load_model_from_checkpoint(
        cls,
        checkpoint_path: str | os.PathLike[str],
        *,
        model_cls: type[torch.nn.Module],
        device: torch.device,
        cfg: Any | None = None,
        checkpoint_state_key: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        eval_mode: bool = True,
        freeze_parameters: bool = True,
    ) -> torch.nn.Module:
        checkpoint = CheckpointStore.load_checkpoint_payload_file(checkpoint_path, device = device)

        resolved_cfg = cfg
        resolve_cfg = getattr(model_cls, "resolve_checkpoint_config", None)
        if callable(resolve_cfg):
            resolved_cfg = resolve_cfg(checkpoint, cfg = cfg)

        init_kwargs = dict(model_kwargs or {})
        if resolved_cfg is None:
            model = model_cls(**init_kwargs)
        else:
            model = model_cls(resolved_cfg, **init_kwargs)
        model = model.to(device)

        load_payload = getattr(model, "load_checkpoint_payload", None)
        if not callable(load_payload):
            raise TypeError(
                f"{model_cls.__name__} must provide load_checkpoint_payload() for checkpoint restore."
            )
        resolved_key = cls._default_checkpoint_state_key(model_cls, checkpoint_state_key)
        load_payload(checkpoint, checkpoint_state_key = resolved_key)

        model = model.to(device)
        if eval_mode:
            model.eval()
        if freeze_parameters:
            for param in model.parameters():
                param.requires_grad_(False)
        if resolved_cfg is not None:
            setattr(model, "train_config", resolved_cfg)
        return model

    def load_into_state(
        self,
        checkpoint_path: Path,
        state: CheckpointState,
        *,
        device: torch.device,
        restore_optimizers: bool = True,
    ) -> None:
        """Load a checkpoint file and populate *state* entries."""
        checkpoint = self.store.load_checkpoint_payload_file(checkpoint_path, device = device)
        if not restore_optimizers:
            # Skip optimizer entries by temporarily removing them
            saved_optimizers: dict[str, Any] = {}
            for key, obj in list(state.entries.items()):
                if not isinstance(obj, torch.nn.Module) and _is_stateful_object(obj):
                    saved_optimizers[key] = state.entries.pop(key)
            load_checkpoint_into_state(checkpoint, state)
            state.entries.update(saved_optimizers)
        else:
            load_checkpoint_into_state(checkpoint, state)

    def save_from_state(
        self,
        state: CheckpointState,
        *,
        checkpoint_path: Path | None = None,
        upload_to_wandb: bool | None = None,
    ) -> Path:
        """Build a checkpoint payload from *state* and write to disk."""
        payload = build_checkpoint_payload_from_state(state)

        serialized_config = self._serialized_config_payload()
        if serialized_config is not None:
            payload[self.config_payload_key] = serialized_config

        iteration = state.iteration
        resolved_path = checkpoint_path
        if resolved_path is None:
            if iteration <= 0:
                raise ValueError(
                    "checkpoint_path must be provided when checkpoint iteration is absent or non-positive."
                )
            resolved_path = self.checkpoint_dir / f"ckpt_{iteration}.pt"
        resolved_path = Path(resolved_path)
        resolved_path.parent.mkdir(parents = True, exist_ok = True)
        self.store.save_checkpoint_payload_file(payload, resolved_path)
        print(f"[Checkpoint] saved to {resolved_path}")

        should_upload = (not self.wandb_disabled) if upload_to_wandb is None else bool(upload_to_wandb)
        if should_upload:
            self.store.upload_checkpoint_to_wandb(resolved_path)
        return resolved_path

    def _serialized_config_payload(self) -> Any:
        if is_dataclass(self.config_payload):
            return asdict(self.config_payload)
        return self.config_payload


__all__ = [
    "CheckpointStore",
    "CheckpointManager",
    "CheckpointState",
    "DEFAULT_WANDB_CACHE_DIR",
    "WandbCheckpointRef",
    "build_checkpoint_payload_from_state",
    "load_checkpoint_into_state",
]
