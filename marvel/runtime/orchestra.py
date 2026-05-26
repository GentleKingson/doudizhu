from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, Mapping

import torch
import torch.distributed as dist

from marvel.runtime.checkpoints import CheckpointManager
from marvel.runtime.compilation import ModuleCompiler
from marvel.runtime.distributed import DistributedContext, DistributedRuntime
from marvel.runtime.logger import TrainingLogger


class TrainingRuntime:
    """Runtime-agnostic training orchestration for distributed/logging/checkpoint plumbing."""

    def __init__(
        self,
        cfg: Any,
        *,
        game: object,
        distributed: DistributedContext | None = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        config_payload: Mapping[str, Any] | Any | None = None,
        total_steps: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.game = game
        self.distributed = (
            distributed if distributed is not None else DistributedRuntime.init_from_env(str(cfg.device))
        )
        self.device = self.distributed.device
        self.rank = int(self.distributed.rank)
        self.world_size = int(self.distributed.world_size)
        self.is_primary = bool(self.distributed.is_primary)

        self.start_iteration = 1
        self.current_iteration: int | None = None
        self._env_steps_total: float | int | None = None
        self._game_steps_total: float | int | None = None
        self.sampler: Any | None = None
        self._metric_buffer: dict[str, object] = {}

        resolved_total_steps = total_steps
        if resolved_total_steps is None:
            cfg_total_steps = getattr(cfg, "total_iterations", None)
            resolved_total_steps = int(cfg_total_steps) if cfg_total_steps is not None else None

        self.logger = TrainingLogger(
            cfg,
            game = game,
            enabled = self.is_primary,
            total_steps = resolved_total_steps,
        )
        self.compiler = ModuleCompiler.from_cfg(cfg, log = print if self.is_primary else None)

        resolved_checkpoint_dir = checkpoint_dir
        if resolved_checkpoint_dir is None:
            resolved_checkpoint_dir = Path(str(cfg.logging_dir)) / "checkpoints"
        self.checkpoint_dir = Path(os.fspath(resolved_checkpoint_dir))
        self.checkpoint_dir.mkdir(parents = True, exist_ok = True)

        resolved_config_payload = config_payload if config_payload is not None else cfg
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir = self.checkpoint_dir,
            config_payload = resolved_config_payload,
            wandb_disabled = self.logger.wandb_disabled,
        )

    def compile(self, module: torch.nn.Module) -> torch.nn.Module:
        return self.compiler.compile(module)

    def distribute(
        self,
        module: torch.nn.Module,
        *,
        find_unused_parameters: bool = False,
    ) -> torch.nn.Module:
        """Wrap a module in DistributedDataParallel when multi-GPU training is active."""
        return self.distributed.wrap_model(
            module,
            find_unused_parameters = find_unused_parameters,
        )

    def prepare(
        self,
        module: torch.nn.Module,
        *,
        find_unused_parameters: bool = False,
    ) -> torch.nn.Module:
        """Compile and distribute a module for training in one step."""
        return self.distribute(
            self.compile(module),
            find_unused_parameters = find_unused_parameters,
        )

    def reduce_mean(self, value: float | int | torch.Tensor | None) -> float | None:
        return self.distributed.reduce_mean(value)

    def reduce_sum(self, value: float | int | torch.Tensor | None) -> float | int | None:
        return self.distributed.reduce_sum(value)

    def _require_iteration(self) -> int:
        if self.current_iteration is None:
            raise RuntimeError("Operation requires runtime iteration context.")
        return int(self.current_iteration)

    def capture_metrics(self, metrics: Mapping[str, object]) -> None:
        """Buffer metrics for end-of-iteration emission."""
        self._metric_buffer.update(metrics)

    def log_metrics(self) -> None:
        """Emit all buffered metrics for the current iteration, then clear the buffer."""
        step = self._require_iteration()
        self.logger.log_metrics(dict(self._metric_buffer), step = step)
        self._metric_buffer.clear()

    def param_stats_due(self) -> bool:
        return self.logger.param_stats_due(self._require_iteration())

    def capture_param_stats(self, model: torch.nn.Module, scope: str) -> None:
        step = self._require_iteration()
        self.logger.log_param_stats(model, step, scope)

    def progress_bar_step(self) -> None:
        self.logger.progress_bar_step()

    def iterations(self) -> Iterator[int]:
        total_iterations = int(getattr(self.cfg, "total_iterations"))
        for iteration in range(self.start_iteration, total_iterations + 1):
            self.current_iteration = int(iteration)
            yield int(iteration)

    def __iter__(self) -> Iterator[int]:
        return self.iterations()

    def attach_sampler(self, sampler: Any) -> None:
        self.sampler = sampler

    def _require_sampler(self) -> Any:
        if self.sampler is None:
            raise RuntimeError("Sampler is not attached to this runtime.")
        return self.sampler

    @property
    def micro_batch_size(self) -> int:
        sampler = self._require_sampler()
        return int(getattr(sampler, "rollout_batch_size"))

    @property
    def collect_batch_size(self) -> int:
        sampler = self._require_sampler()
        return int(getattr(sampler, "collect_batch_size"))

    @property
    def total_env_steps(self) -> int:
        sampler = self._require_sampler()
        return int(getattr(sampler, "total_env_steps"))

    @property
    def total_game_steps(self) -> int:
        sampler = self._require_sampler()
        return int(getattr(sampler, "total_game_steps"))

    @property
    def env_steps_total(self) -> float | int:
        if self._env_steps_total is None:
            raise RuntimeError("env_steps_total is unavailable before runtime step totals are synchronized.")
        return self._env_steps_total

    @property
    def game_steps_total(self) -> float | int:
        if self._game_steps_total is None:
            raise RuntimeError("game_steps_total is unavailable before runtime step totals are synchronized.")
        return self._game_steps_total

    def sync_step_totals_from_sampler(self) -> None:
        _ = self._require_sampler()
        self._env_steps_total = self.reduce_sum(self.total_env_steps)
        self._game_steps_total = self.reduce_sum(self.total_game_steps)

    def set_sampler_counters(
        self,
        *,
        total_env_steps: int,
        total_game_steps: int,
    ) -> None:
        sampler = self._require_sampler()
        setattr(sampler, "total_env_steps", int(total_env_steps))
        setattr(sampler, "total_game_steps", int(total_game_steps))

    def _distributed_enabled(self) -> bool:
        return bool(getattr(self.distributed, "enabled", False)) and dist.is_initialized()

    def resolve_checkpoint_path(
        self,
        locator: str | os.PathLike[str],
        *,
        broadcast: bool = True,
    ) -> Path:
        checkpoint_path = Path("")
        if self.is_primary or not broadcast:
            checkpoint_path = self.checkpoint_manager.store.resolve_checkpoint_file_path(locator)

        if broadcast and self._distributed_enabled():
            payload = [str(checkpoint_path) if self.is_primary else ""]
            broadcast_kwargs: dict[str, Any] = {}
            if dist.get_backend() == "nccl":
                device = getattr(self.distributed, "device", None)
                if isinstance(device, torch.device):
                    broadcast_kwargs["device"] = device
            dist.broadcast_object_list(payload, src = 0, **broadcast_kwargs)
            checkpoint_path = Path(str(payload[0]))

        if not self.is_primary and self._distributed_enabled() and not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Expected checkpoint to be present at {checkpoint_path} after path broadcast. "
                "Ensure cfg.logging_dir is on a shared filesystem."
            )

        return checkpoint_path

    def load_checkpoint_payload(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        device: torch.device | None = None,
    ) -> Mapping[str, Any]:
        target_device = self.device if device is None else device
        return self.checkpoint_manager.store.load_checkpoint_payload_file(
            checkpoint_path,
            device = target_device,
        )

    def load_model_from_checkpoint(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        model_cls: type[torch.nn.Module],
        cfg: Any | None = None,
        checkpoint_state_key: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        eval_mode: bool = True,
        freeze_parameters: bool = True,
        device: torch.device | None = None,
    ) -> torch.nn.Module:
        target_device = self.device if device is None else device
        return self.checkpoint_manager.load_model_from_checkpoint(
            checkpoint_path,
            model_cls = model_cls,
            device = target_device,
            cfg = cfg,
            checkpoint_state_key = checkpoint_state_key,
            model_kwargs = model_kwargs,
            eval_mode = eval_mode,
            freeze_parameters = freeze_parameters,
        )

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> TrainingRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.logger.close()
        if dist.is_initialized():
            dist.destroy_process_group()



__all__ = ["TrainingRuntime"]
