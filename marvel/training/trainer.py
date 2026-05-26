from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from marvel.config import TrainConfig
from marvel.training.workflow import TrainingWorkflow
from marvel.utils import set_global_seed

RuntimeFactory = Callable[[TrainConfig], Any]


def compute_stable_decay_scale(
    *,
    stable_iterations: int,
    iteration: int,
    decay_power: float,
) -> float:
    if iteration <= 0:
        raise ValueError(f"iteration must be positive; got {iteration}.")
    if stable_iterations < 0:
        raise ValueError(f"stable_iterations must be non-negative; got {stable_iterations}.")
    if decay_power < 0.0:
        raise ValueError(f"decay_power must be non-negative; got {decay_power}.")
    return min(stable_iterations / iteration, 1.0) ** decay_power


@dataclass(frozen = True)
class ScheduleState:
    """Per-iteration schedule values shared across all trainers."""

    lr_scale: float
    critic_lr_scale: float
    clip_coef_scale: float
    agent_lr: float
    clip_coef: float
    reg_coef: float
    reg_scale: float


class AlgorithmTrainer(ABC):
    """Minimal base class for MARVEL algorithm trainers."""

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.cfg = cfg
        self._runtime_factory = runtime_factory

    def _default_runtime_factory(self, cfg: TrainConfig) -> TrainingWorkflow:
        return TrainingWorkflow(cfg, game = cfg.game)

    def _make_runtime(self) -> Any:
        factory = self._runtime_factory or self._default_runtime_factory
        return factory(self.cfg)

    def _set_seed(self, runtime: Any) -> None:
        set_global_seed(self.cfg.seed, offset = runtime.rank)

    def _compute_stable_decay_scale(
        self,
        *,
        stable_iterations: int,
        iteration: int,
        decay_power: float,
    ) -> float:
        return compute_stable_decay_scale(
            stable_iterations = stable_iterations,
            iteration = iteration,
            decay_power = decay_power,
        )

    def _compute_agent_schedule(self, iteration: int) -> ScheduleState:
        """Compute common per-iteration schedule values from config."""
        cfg = self.cfg
        lr_scale = self._compute_stable_decay_scale(
            stable_iterations = cfg.lr_stable_iterations,
            iteration = iteration,
            decay_power = cfg.schedule.agent_lr_decay_power,
        )
        critic_lr_scale = self._compute_stable_decay_scale(
            stable_iterations = cfg.lr_stable_iterations,
            iteration = iteration,
            decay_power = cfg.schedule.critic_lr_decay_power,
        )
        clip_coef_scale = self._compute_stable_decay_scale(
            stable_iterations = cfg.lr_stable_iterations,
            iteration = iteration,
            decay_power = cfg.schedule.clip_coef_decay_power,
        )
        reg_scale = self._compute_stable_decay_scale(
            stable_iterations = cfg.reg_stable_iterations,
            iteration = iteration,
            decay_power = cfg.schedule.reg_coef_decay_power,
        )
        return ScheduleState(
            lr_scale = lr_scale,
            critic_lr_scale = critic_lr_scale,
            clip_coef_scale = clip_coef_scale,
            agent_lr = cfg.ppo.base_lr * lr_scale,
            clip_coef = cfg.ppo.base_clip_coef * clip_coef_scale,
            reg_coef = cfg.ppo.base_reg_coef * reg_scale,
            reg_scale = reg_scale,
        )

    @abstractmethod
    def run(self) -> None:
        ...


__all__ = ["AlgorithmTrainer", "ScheduleState", "compute_stable_decay_scale"]
