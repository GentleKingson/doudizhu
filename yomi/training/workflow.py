from __future__ import annotations

import os
from typing import Any, Mapping

import torch

from marvel.model import Agent
from marvel.runtime.checkpoints import CheckpointState, load_checkpoint_into_state
from marvel.runtime.distributed import DistributedContext, DistributedRuntime
from marvel.runtime.intervals import IntervalTimer
from marvel.runtime.orchestra import TrainingRuntime as BaseTrainingRuntime
from marvel.runtime.sampler import Sampler, SamplerConfig
from marvel.utils import unwrap_distributed_module

from yomi.config import TrainConfig
from yomi.runtime.sampler import CompletedEpisodeRootCollector


class YomiResumePolicy:
    """Resume-time checkpoint policy for yomi workflow restore."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def resume(self, state: CheckpointState) -> None:
        runtime = self.runtime
        resume_locator = str(getattr(runtime.cfg, "resume_from", "") or "").strip()
        if not resume_locator:
            runtime.start_iteration = 1
            return

        checkpoint_path = runtime.resolve_checkpoint_path(resume_locator, broadcast = True)

        base_model: torch.nn.Module | None = None
        for obj in state.entries.values():
            if isinstance(obj, torch.nn.Module):
                base_model = unwrap_distributed_module(obj)
                break
        try:
            target_device = next(base_model.parameters()).device if base_model is not None else torch.device("cpu")
        except StopIteration:
            target_device = torch.device("cpu")

        restore_optimizer = bool(getattr(runtime.cfg, "resume_optimizer", True))
        runtime.checkpoint_manager.load_into_state(
            checkpoint_path, state, device = target_device,
            restore_optimizers = restore_optimizer,
        )

        iteration = state.iteration
        if iteration <= 0:
            raise RuntimeError("Yomi resume requires checkpoint iteration metadata.")

        progress_bar = getattr(runtime.logger, "progress_bar", None)
        update = getattr(progress_bar, "update", None)
        if callable(update):
            advance = min(max(iteration, 0), int(runtime.cfg.total_iterations))
            update(advance)

        runtime.start_iteration = iteration + 1


class YomiCheckpointPolicy:
    """Save-time checkpoint policy for yomi workflow payload assembly and cadence."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.timer = IntervalTimer(runtime.cfg.checkpoint_interval, total_steps = runtime.cfg.total_iterations)

    def due(self, iteration: int) -> bool:
        """Check whether a checkpoint should be saved at this iteration."""
        return self.runtime.is_primary and self.timer.triggered(iteration)

    def save(self, state: CheckpointState) -> None:
        self.runtime.checkpoint_manager.save_from_state(state)


class YomiTrainingWorkflow(BaseTrainingRuntime):
    """Yomi-specific training workflow built on top of runtime orchestration."""

    def __init__(
        self,
        cfg: TrainConfig,
        game: object,
        distributed: DistributedContext | None = None,
        config_payload: TrainConfig | Mapping[str, Any] | None = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        total_steps: int | None = None,
    ) -> None:
        resolved_distributed = (
            distributed
            if distributed is not None
            else DistributedRuntime.init_from_env(str(cfg.device))
        )
        resolved_total_steps = int(cfg.total_iterations) if total_steps is None else int(total_steps)
        resolved_config_payload = cfg if config_payload is None else config_payload
        super().__init__(
            cfg,
            game = game,
            distributed = resolved_distributed,
            checkpoint_dir = checkpoint_dir,
            config_payload = resolved_config_payload,
            total_steps = resolved_total_steps,
        )
        self.resume_policy = YomiResumePolicy(self)
        self.checkpoint_policy = YomiCheckpointPolicy(self)
        self.attach_sampler(self._make_sampler())
        self._state: CheckpointState | None = None

    def _make_sampler(self) -> Sampler:
        global_rollout_batch_size = int(self.cfg.rollout_batch_size)
        if global_rollout_batch_size <= 0:
            raise ValueError(f"rollout_batch_size must be positive, got {global_rollout_batch_size}.")
        if global_rollout_batch_size % self.world_size != 0:
            raise ValueError(
                f"rollout_batch_size ({global_rollout_batch_size}) must be divisible by world_size ({self.world_size})."
            )
        rollout_batch_size = global_rollout_batch_size // self.world_size
        env_seed = int(self.cfg.seed) + int(self.rank)
        return Sampler(
            config = SamplerConfig(
                game = self.game,
                device = self.device,
                rollout_batch_size = rollout_batch_size,
                collect_batch_size = 0,
                sampling_temperature = float(self.cfg.rollout_temperature),
                env_seed = env_seed,
            )
        )

    @property
    def local_rollout_batch_size(self) -> int:
        sampler = self._require_sampler()
        return int(getattr(sampler, "rollout_batch_size"))

    @property
    def collect_interval(self) -> int:
        raw_interval = getattr(self.cfg, "collect_interval", None)
        interval = int(self.game.MAX_GAME_HORIZON) if raw_interval is None else int(raw_interval)
        if interval <= 0:
            raise ValueError(f"collect_interval must be positive, got {interval}.")
        return interval

    @property
    def collect_warmup_steps(self) -> int:
        raw_steps = getattr(self.cfg, "collect_warmup_steps", None)
        if raw_steps is None:
            has_reconstruct = callable(getattr(self.game, "reconstruct_envs", None))
            has_history_prefill = callable(getattr(self.game, "parse_history_strs", None))
            steps = int(self.cfg.rollout_batch_size) if has_reconstruct and has_history_prefill else 0
        else:
            steps = int(raw_steps)
        if steps < 0:
            raise ValueError(f"collect_warmup_steps must be non-negative, got {steps}.")
        return steps

    def iterations(self):
        total_iterations = int(getattr(self.cfg, "total_iterations"))
        for iteration in range(self.start_iteration, total_iterations + 1):
            self.current_iteration = int(iteration)
            if self._state is not None:
                self._state[self._state.iteration_key] = int(iteration)
            yield int(iteration)

    def resolve_policy_checkpoint_path(self, locator: str | os.PathLike[str]) -> os.PathLike[str]:
        checkpoint_ref = str(locator).strip()
        if not checkpoint_ref:
            raise ValueError("policy_checkpoint_path must be a non-empty path or wandb reference.")
        return self.resolve_checkpoint_path(checkpoint_ref, broadcast = True)

    def load_frozen_policy(self, *, checkpoint_path: str | os.PathLike[str]) -> torch.nn.Module:
        return self.load_model_from_checkpoint(
            checkpoint_path,
            model_cls = Agent,
            eval_mode = True,
            freeze_parameters = True,
        )

    def make_action_root_collector(
        self,
        *,
        policy: torch.nn.Module,
        pad_token: int,
    ) -> CompletedEpisodeRootCollector:
        sampler = self._require_sampler()
        return CompletedEpisodeRootCollector(
            sampler = sampler,
            policy = policy,
            game = self.game,
            pad_token = int(pad_token),
        )

    def resume(self, state: CheckpointState) -> None:
        self._state = state
        self.resume_policy.resume(state)

    def checkpoint_due(self) -> bool:
        return self.checkpoint_policy.due(self._require_iteration())

    def save_checkpoint(self, state: CheckpointState) -> None:
        self.checkpoint_policy.save(state)


__all__ = [
    "YomiResumePolicy",
    "YomiCheckpointPolicy",
    "YomiTrainingWorkflow",
]
