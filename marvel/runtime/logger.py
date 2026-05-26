"""Training logging and progress helpers."""

from __future__ import annotations

import csv
import json
import os

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
from tqdm import tqdm

from marvel.config import TrainConfig
from marvel.game_identity import extract_game_identifier
from marvel.runtime.intervals import IntervalTimer
from marvel.utils import unwrap_distributed_module

from .distributed import NullProgressBar


class MetricSerializer:
    """Serialize metric payload values into JSON/CSV-safe Python values."""

    @staticmethod
    def serialize(value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                return tensor.item()
            return tensor.tolist()
        if isinstance(value, dict):
            return {str(k): MetricSerializer.serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [MetricSerializer.serialize(v) for v in value]
        return str(value)


class MetricSinks:
    """Write metrics to optional JSONL/CSV sinks configured via environment variables."""

    def __init__(
        self,
        *,
        jsonl_path: Path | None,
        csv_path: Path | None,
    ) -> None:
        self._jsonl_path = jsonl_path
        self._csv_path = csv_path
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents = True, exist_ok = True)
        if self._csv_path is not None:
            self._csv_path.parent.mkdir(parents = True, exist_ok = True)

    @classmethod
    def from_env(cls, *, default_csv_path: Path | None = None) -> MetricSinks:
        jsonl_raw = os.environ.get("MARVEL_METRICS_JSONL", "").strip()
        jsonl_path = Path(jsonl_raw) if jsonl_raw else None

        csv_raw = os.environ.get("MARVEL_METRICS_CSV", "").strip()
        if csv_raw:
            csv_path = Path(csv_raw)
        else:
            csv_path = default_csv_path

        return cls(jsonl_path = jsonl_path, csv_path = csv_path)

    def log(self, payload: Mapping[str, object], *, step: int) -> None:
        if self._jsonl_path is not None:
            record: dict[str, object] = {"step": int(step)}
            record.update({k: MetricSerializer.serialize(v) for k, v in payload.items()})
            with self._jsonl_path.open("a", encoding = "utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

        if self._csv_path is not None:
            should_write_header = True
            try:
                should_write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
            except OSError:
                should_write_header = True

            with self._csv_path.open("a", newline = "", encoding = "utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames = ["step", "key", "value"])
                if should_write_header:
                    writer.writeheader()
                for key, value in payload.items():
                    serialized = MetricSerializer.serialize(value)
                    if serialized is None or isinstance(serialized, (str, int, float, bool)):
                        csv_value = "" if serialized is None else str(serialized)
                    else:
                        csv_value = json.dumps(serialized)
                    writer.writerow({"step": int(step), "key": str(key), "value": csv_value})


class WandbSession:
    """Optional W&B lifecycle and logging wrapper."""

    def __init__(self, module: Any | None) -> None:
        self._module = module

    @property
    def disabled(self) -> bool:
        return self._module is None

    @staticmethod
    def _import_optional() -> Any | None:
        try:
            import wandb  # type: ignore[import-not-found]
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            return None
        return wandb

    @classmethod
    def require_module(cls) -> Any:
        module = cls._import_optional()
        if module is None:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "wandb is required for this operation. Install wandb or set WANDB_DISABLED=true."
            )
        return module

    @staticmethod
    def _build_run_name(cfg: TrainConfig, game: object) -> str:
        return str(cfg.run_name or WandbSession._extract_game_identifier(game))

    @staticmethod
    def _extract_game_identifier(game: object) -> str:
        try:
            return extract_game_identifier(game)
        except ValueError as exc:
            raise RuntimeError(
                "Game module is missing GAME_IDENTIFIER; cannot tag W&B run. "
                "Ensure the game module/adapter exposes a canonical GAME_IDENTIFIER."
            ) from exc

    @classmethod
    def from_cfg(cls, cfg: TrainConfig, game: object) -> WandbSession:
        wandb_disabled_env = os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
        if wandb_disabled_env:
            return cls(None)

        module = cls._import_optional()
        if module is None:
            print("[Logging] wandb not installed; disabling W&B logging.")
            return cls(None)

        game_identifier = cls._extract_game_identifier(game)
        config_payload = asdict(cfg)
        config_payload["game_identifier"] = game_identifier

        try:
            module.init(
                project = cfg.project_name,
                name = cls._build_run_name(cfg, game),
                config = config_payload,
                tags = [game_identifier],
            )
        except Exception as exc:
            raise RuntimeError(
                "W&B initialization failed. Set WANDB_DISABLED=true to disable W&B explicitly."
            ) from exc

        return cls(module)

    def log(self, payload: Mapping[str, object], *, step: int) -> None:
        if self._module is None:
            return
        try:
            self._module.log(dict(payload), step = int(step))
        except Exception as exc:
            raise RuntimeError(f"W&B metric logging failed at step={step}.") from exc

    def close(self) -> None:
        if self._module is not None:
            self._module.finish()


class RuntimeLoggingSession:
    """Primary runtime logging object used by training and tools."""

    def __init__(self, cfg: TrainConfig, game: object) -> None:
        os.makedirs(cfg.logging_dir, exist_ok = True)

        self._wandb = WandbSession.from_cfg(cfg, game)

        default_csv_path = None
        if self._wandb.disabled and not os.environ.get("MARVEL_METRICS_CSV", "").strip():
            default_csv_path = Path(cfg.logging_dir) / "metrics.csv"
        self._sinks = MetricSinks.from_env(default_csv_path = default_csv_path)

    @property
    def wandb_disabled(self) -> bool:
        return self._wandb.disabled

    def log_metrics(self, payload: dict[str, object], *, step: int) -> None:
        self._wandb.log(payload, step = step)
        self._sinks.log(payload, step = step)

    def close(self) -> None:
        self._wandb.close()


class ParamStatsLogger:
    """Compute and emit parameter/gradient norm diagnostics."""

    @staticmethod
    def log(
        model: torch.nn.Module,
        step: int,
        scope: str,
        log_metrics: Callable[..., None],
    ) -> None:
        base_model = unwrap_distributed_module(model)
        wd_param_names = set(base_model.weight_decay_param_names)
        metric_keys: list[str] = []
        metric_values: list[torch.Tensor] = []
        for name, param in base_model.named_parameters():
            if name not in wd_param_names:
                continue

            metric_keys.append(f"{scope}/{name}_weight_norm")
            metric_values.append(param.detach().norm().float())
            if param.grad is not None:
                metric_keys.append(f"{scope}/{name}_grad_norm")
                metric_values.append(param.grad.detach().norm().float())

        if not metric_keys:
            return

        metric_values_cpu = torch.stack(metric_values).detach().cpu().tolist()
        log_dict = dict(zip(metric_keys, metric_values_cpu))
        log_metrics(log_dict, step = int(step))


class TrainingLogger:
    """Lightweight helper for metrics and progress."""

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        game: object,
        enabled: bool = True,
        total_steps: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.is_primary = bool(enabled)
        self.param_stats_timer = IntervalTimer(
            getattr(cfg, "param_stats_interval", 0) or 0,
            total_steps = getattr(cfg, "total_iterations", None),
        )
        self._session: RuntimeLoggingSession | None

        if self.is_primary:
            self._session = RuntimeLoggingSession(cfg, game)
            progress_bar: tqdm | NullProgressBar = tqdm(
                total = total_steps if total_steps is not None else cfg.total_iterations,
                desc = "Training",
                unit = "step",
            )
            self.log_metrics = self._session.log_metrics
            self.wandb_disabled = self._session.wandb_disabled
        else:
            self._session = None
            self.log_metrics = lambda *_args, **_kwargs: None
            self.wandb_disabled = True
            progress_bar = NullProgressBar()

        self.progress_bar = progress_bar

    def param_stats_due(self, step: int) -> bool:
        """Check whether parameter stats should be logged at this step."""
        return self.is_primary and self.param_stats_timer.triggered(step)

    def log_param_stats(self, model: torch.nn.Module, step: int, scope: str) -> None:
        ParamStatsLogger.log(model, step, scope, self.log_metrics)

    def progress_bar_step(self) -> None:
        self.progress_bar.update(1)

    def close(self) -> None:
        self.progress_bar.close()
        if self._session is not None:
            self._session.close()

__all__ = [
    "MetricSerializer",
    "MetricSinks",
    "WandbSession",
    "RuntimeLoggingSession",
    "ParamStatsLogger",
    "TrainingLogger",
]
