from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
import torch.distributed as dist

from marvel.config import TrainConfig, deserialize_train_config
from marvel.runtime.distributed import DistributedContext
from marvel.runtime.intervals import IntervalTimer
from marvel.utils import unwrap_distributed_module


@dataclass(frozen = True)
class EvaluationResult:
    metrics: dict[str, object]
    duration_sec: float | None = None


@dataclass
class TrainingEvaluationHook:
    name: str
    timer: IntervalTimer
    requires_checkpoint: bool
    _run: Callable[[str, int, int | None, Path | None, torch.nn.Module | None], None]

    def should_run(self, step: int) -> bool:
        return self.timer.triggered(step)

    def __call__(
        self,
        step_label: str,
        step_index: int,
        *,
        env_steps: int | None,
        checkpoint_path: Path | None,
        agent_policy: torch.nn.Module | None,
    ) -> None:
        self._run(step_label, step_index, env_steps, checkpoint_path, agent_policy)


TrainingHookFactory = Callable[
    [
        TrainConfig,
        object,
        Callable[..., None],
        Optional[DistributedContext],
        IntervalTimer,
        Path,
    ],
    TrainingEvaluationHook,
]

CheckpointEvalFn = Callable[[Path, TrainConfig, torch.device], EvaluationResult]


@dataclass(frozen = True)
class EvaluationRoutine:
    name: str
    make_training_hook: TrainingHookFactory
    evaluate_checkpoint: CheckpointEvalFn


_EVALUATION_ROUTINES: dict[str, EvaluationRoutine] = {}


def register_routine(routine: EvaluationRoutine) -> None:
    if routine.name in _EVALUATION_ROUTINES:
        raise KeyError(f"Duplicate evaluation routine name: {routine.name}")
    _EVALUATION_ROUTINES[routine.name] = routine


def available() -> tuple[str, ...]:
    return tuple(sorted(_EVALUATION_ROUTINES.keys()))


def resolve(name: str) -> EvaluationRoutine:
    return _EVALUATION_ROUTINES[name]


def _lookup_routines(names: Sequence[str]) -> list[EvaluationRoutine]:
    routines: list[EvaluationRoutine] = []
    for name in names:
        try:
            routines.append(resolve(name))
        except KeyError as exc:
            choices = ", ".join(available())
            raise ValueError(f"Unknown evaluation routine '{name}'. Available options: {choices or 'none'}.") from exc
    return routines


def _save_eval_checkpoint(
    checkpoint_dir: Path,
    *,
    iteration: int,
    total_env_steps: int | None,
    total_game_steps: int | None,
    agent_model: torch.nn.Module,
    cfg: TrainConfig,
) -> Path:
    checkpoint_dir.mkdir(parents = True, exist_ok = True)
    agent_model = unwrap_distributed_module(agent_model)
    ckpt = {
        "iteration": iteration,
        "total_env_steps": int(total_env_steps) if total_env_steps is not None else None,
        "total_game_steps": int(total_game_steps) if total_game_steps is not None else None,
        "agent_state_dict": agent_model.state_dict(),
        "train_config": asdict(cfg),
    }
    ckpt_path = checkpoint_dir / f"eval_ckpt_{iteration}.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


class TrainingEvaluator:
    """Run configured evaluation routines during training (separate from TrainingLogger)."""

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        game: object,
        log_metrics: Callable[..., None],
        distributed: Optional[DistributedContext],
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.game = game
        self.log_metrics = log_metrics
        self.distributed = distributed
        self._dist_enabled = distributed is not None and distributed.enabled
        self._is_primary = distributed is None or distributed.is_primary

        self.timer = IntervalTimer(cfg.evaluation_interval, total_steps = cfg.total_iterations)
        self.checkpoint_dir = checkpoint_dir or Path(cfg.logging_dir) / "checkpoints"

        routine_names = cfg.evaluation_routines
        if routine_names is None:
            self.hooks: list[TrainingEvaluationHook] = []
            return

        routines = _lookup_routines(routine_names)
        hooks: list[TrainingEvaluationHook] = []
        for routine in routines:
            hooks.append(
                routine.make_training_hook(
                    cfg,
                    game,
                    log_metrics,
                    distributed,
                    self.timer,
                    self.checkpoint_dir,
                )
            )
        self.hooks = hooks

    def should_run(self, iteration: int) -> bool:
        """Whether any hook is scheduled at this iteration.

        This must return the same value on every rank so that all ranks
        agree on entering handle_step() (which contains dist.barrier calls).
        Rank-specific gating lives inside handle_step(), not here.
        """
        if not self.hooks:
            return False
        return any(hook.should_run(iteration) for hook in self.hooks)

    def handle_step(
        self,
        iteration: int,
        *,
        env_steps: int | None = None,
        game_steps: int | None = None,
        agent_policy: torch.nn.Module | None = None,
    ) -> None:
        scheduled = [hook for hook in self.hooks if hook.should_run(iteration)]

        if self._dist_enabled:
            dist.barrier()
        if not self._is_primary:
            if self._dist_enabled:
                dist.barrier()
            return

        try:
            step_label = f"iteration {iteration}"
            in_memory_hooks = [hook for hook in scheduled if not hook.requires_checkpoint]
            checkpoint_hooks = [hook for hook in scheduled if hook.requires_checkpoint]

            for hook in in_memory_hooks:
                hook(
                    step_label,
                    iteration,
                    env_steps = env_steps,
                    checkpoint_path = None,
                    agent_policy = agent_policy,
                )

            checkpoint_path: Path | None = None
            if checkpoint_hooks:
                candidate = self.checkpoint_dir / f"ckpt_{iteration}.pt"
                if candidate.exists():
                    checkpoint_path = candidate
                else:
                    if agent_policy is None:
                        raise ValueError(
                            "TrainingEvaluator.handle_step requires agent_policy when no checkpoint file exists."
                        )
                    checkpoint_path = _save_eval_checkpoint(
                        self.checkpoint_dir,
                        iteration = iteration,
                        total_env_steps = env_steps,
                        total_game_steps = game_steps,
                        agent_model = agent_policy,
                        cfg = self.cfg,
                    )

                for hook in checkpoint_hooks:
                    hook(
                        step_label,
                        iteration,
                        env_steps = env_steps,
                        checkpoint_path = checkpoint_path,
                        agent_policy = agent_policy,
                    )
        finally:
            if self._dist_enabled:
                # Non-primary ranks enter the exit barrier immediately after the entry barrier.
                # Always release them even if primary-only evaluation fails.
                dist.barrier()


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    cfg: TrainConfig | None = None,
    device: torch.device,
    routines: Sequence[str] | None = None,
) -> EvaluationResult:
    checkpoint = torch.load(str(checkpoint_path), map_location = device)
    cfg_payload = checkpoint.get("train_config")
    checkpoint_cfg = deserialize_train_config(cfg_payload) if cfg_payload is not None else None
    resolved_cfg = cfg or checkpoint_cfg
    if resolved_cfg is None:
        raise KeyError(
            "Checkpoint does not include 'train_config'. Pass cfg=TrainConfig(...) explicitly."
        )

    routine_names = tuple(resolved_cfg.evaluation_routines or ()) if routines is None else tuple(routines)
    if not routine_names:
        return EvaluationResult(metrics = {})

    resolved_routines = _lookup_routines(routine_names)
    metrics: dict[str, object] = {}
    total_duration = 0.0
    for routine in resolved_routines:
        start_time = time.perf_counter()
        result = routine.evaluate_checkpoint(checkpoint_path, resolved_cfg, device)
        duration = time.perf_counter() - start_time
        metrics.update(result.metrics)
        total_duration += result.duration_sec if result.duration_sec is not None else duration

    return EvaluationResult(metrics = metrics, duration_sec = total_duration)


__all__ = [
    "EvaluationResult",
    "EvaluationRoutine",
    "TrainingEvaluationHook",
    "TrainingEvaluator",
    "available",
    "evaluate_checkpoint",
    "register_routine",
    "resolve",
]
