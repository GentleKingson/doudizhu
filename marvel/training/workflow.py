from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from marvel.config import TrainConfig
from marvel.eval import TrainingEvaluator
from marvel.model import BaseMarvelModule
from marvel.runtime.checkpoints import CheckpointManager, CheckpointState, load_checkpoint_into_state
from marvel.runtime.distributed import DistributedContext
from marvel.runtime.intervals import IntervalTimer
from marvel.runtime.orchestra import TrainingRuntime as RuntimeOrchestrator
from marvel.runtime.sampler import Sampler, SamplerConfig
from marvel.utils import compute_commit_step_counts, unwrap_distributed_module


class ResumePolicy:
    """Resume-time policy for training workflow checkpoint restore."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def resume(self, state: CheckpointState) -> None:
        runtime = self.runtime
        if not runtime.cfg.resume_from:
            runtime.start_iteration = 1
            return

        checkpoint_path = runtime.resolve_checkpoint_path(runtime.cfg.resume_from, broadcast = True)

        # Determine device from first model in state.
        target_device = torch.device("cpu")
        for obj in state.entries.values():
            if isinstance(obj, torch.nn.Module):
                base = unwrap_distributed_module(obj)
                try:
                    target_device = next(base.parameters()).device
                except StopIteration:
                    pass
                break

        # Load checkpoint payload and populate the state.
        checkpoint = runtime.load_checkpoint_payload(checkpoint_path, device = target_device)
        BaseMarvelModule.resolve_checkpoint_config(checkpoint, cfg = runtime.cfg)

        restore_optimizer = bool(getattr(runtime.cfg, "resume_optimizer", True))
        if not restore_optimizer:
            # Temporarily remove optimizer entries so they are not loaded.
            from marvel.runtime.checkpoints import _is_stateful_object
            saved_optimizers: dict[str, Any] = {}
            for key, obj in list(state.entries.items()):
                if not isinstance(obj, torch.nn.Module) and _is_stateful_object(obj):
                    saved_optimizers[key] = state.entries.pop(key)
            load_checkpoint_into_state(checkpoint, state)
            state.entries.update(saved_optimizers)
        else:
            load_checkpoint_into_state(checkpoint, state)

        iteration = state.iteration
        if iteration <= 0:
            raise RuntimeError("Training resume requires checkpoint iteration metadata.")

        # Restore sampler counters from state.
        env_steps = int(state.get("total_env_steps", 0) or 0)
        game_steps = int(state.get("total_game_steps", 0) or 0)
        if runtime._distributed_enabled() and not runtime.is_primary:
            env_steps = 0
            game_steps = 0
        runtime.set_sampler_counters(
            total_env_steps = env_steps,
            total_game_steps = game_steps,
        )

        progress_bar = getattr(runtime.logger, "progress_bar", None)
        update = getattr(progress_bar, "update", None)
        if callable(update):
            advance = min(max(iteration, 0), int(runtime.cfg.total_iterations))
            update(advance)

        runtime.start_iteration = iteration + 1


class CheckpointPolicy:
    """Save-time policy for workflow checkpoint payload assembly and cadence."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.timer = IntervalTimer(runtime.cfg.checkpoint_interval, total_steps = runtime.cfg.total_iterations)
        self.last_capture = time.monotonic()
        self.last_iteration = 0

    def due(self, iteration: int) -> bool:
        """Check whether a checkpoint should be saved at this iteration."""
        cfg = self.runtime.cfg
        wallclock_due = (cfg.full_state_checkpoints and cfg.checkpoint_wallclock_seconds > 0
                         and time.monotonic() - self.last_capture >= cfg.checkpoint_wallclock_seconds)
        full_boundary_due = cfg.full_state_checkpoints and iteration in (cfg.replay_buffer_scale, cfg.total_iterations)
        return (self.runtime.is_primary and iteration > self.last_iteration
                and (self.timer.triggered(iteration) or wallclock_due or full_boundary_due))

    def save(self, state: CheckpointState) -> None:
        if self.runtime.cfg.full_state_checkpoints:
            from marvel.runtime.full_state import capture
            capture(self.runtime, state, self.runtime.full_replay, self.runtime.full_ema)
        self.runtime.checkpoint_manager.save_from_state(state)
        self.last_capture = time.monotonic()
        self.last_iteration = state.iteration


class TrainingWorkflow(RuntimeOrchestrator):
    """MARVEL-specific training workflow built on top of runtime orchestration."""

    def __init__(
        self,
        cfg: TrainConfig,
        game: object,
        distributed: DistributedContext | None = None,
        config_payload: TrainConfig | Mapping[str, Any] | None = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        total_steps: int | None = None,
    ) -> None:
        super().__init__(
            cfg,
            game = game,
            distributed = distributed,
            checkpoint_dir = checkpoint_dir,
            config_payload = config_payload,
            total_steps = total_steps,
        )

        self.evaluator = TrainingEvaluator(
            cfg,
            game = game,
            log_metrics = self.logger.log_metrics,
            distributed = self.distributed,
            checkpoint_dir = self.checkpoint_dir,
        )
        self.resume_policy = ResumePolicy(self)
        self.checkpoint_policy = CheckpointPolicy(self)
        self.attach_sampler(self._make_sampler())
        if cfg.full_state_checkpoints:
            if self.world_size != 1 or cfg.algo_name != "vrpo" or cfg.use_compile:
                raise ValueError("Full-state v1 requires single-process eager VRPO.")
            from marvel.runtime.full_state import ActionJournal
            self.sampler.action_journal = ActionJournal(
                Path(cfg.logging_dir) / "journal", self.sampler._env_seed, self.sampler.rollout_batch_size)
        self._state: CheckpointState | None = None

    def _make_sampler(self) -> Sampler:
        if int(self.cfg.rollout_batch_size) % self.world_size != 0:
            raise ValueError(
                f"rollout_batch_size ({self.cfg.rollout_batch_size}) must be divisible by world_size ({self.world_size})."
            )
        if int(self.cfg.collect_batch_size) % self.world_size != 0:
            raise ValueError(
                f"collect_batch_size ({self.cfg.collect_batch_size}) must be divisible by world_size ({self.world_size})."
            )
        rollout_batch_size = int(self.cfg.rollout_batch_size) // self.world_size
        collect_batch_size = int(self.cfg.collect_batch_size) // self.world_size
        env_seed = self.cfg.seed + self.rank

        return Sampler(
            config = SamplerConfig(
                game = self.game,
                device = self.device,
                rollout_batch_size = rollout_batch_size,
                collect_batch_size = collect_batch_size,
                env_seed = env_seed,
            )
        )

    # ------------------------------------------------------------------
    # Iteration lifecycle
    # ------------------------------------------------------------------

    def iterations(self):
        total_iterations = int(getattr(self.cfg, "total_iterations"))
        for iteration in range(self.start_iteration, total_iterations + 1):
            self.current_iteration = int(iteration)
            if self._state is not None:
                self._state[self._state.iteration_key] = int(iteration)
            yield int(iteration)

    def collect_trajectories(self, agent_model: torch.nn.Module) -> Any:
        sampler = self._require_sampler()
        rollout = sampler.collect(agent_model = agent_model)
        self.sync_step_totals_from_sampler()
        # Auto-sync step totals into checkpoint state.
        if self._state is not None:
            if "total_env_steps" in self._state:
                self._state["total_env_steps"] = self.env_steps_total
            if "total_game_steps" in self._state:
                self._state["total_game_steps"] = self.game_steps_total
        return rollout

    def log_rollouts(
        self,
        rollout: Any,
    ) -> None:
        collect_batch_size = int(rollout["rewards"].shape[0])
        if int(collect_batch_size) <= 0:
            raise ValueError("rollout batch dimension must be positive when logging statistics.")

        env_steps_total = self.env_steps_total
        game_steps_total = self.game_steps_total
        game_steps_info = compute_commit_step_counts(
            rollout["avail_is_commit"],
            rollout["action_taken"],
            rollout["num_steps"],
        )
        hand_return0_mean = rollout["rewards"][:, :, 0].sum() / int(collect_batch_size)
        metrics: dict[str, float | int | None] = {
            "statistics/env_steps_total": env_steps_total,
            "statistics/game_steps_total": game_steps_total,
            "statistics/return0_mean": self.reduce_mean(hand_return0_mean),
            "statistics/env_steps_mean": self.reduce_mean(rollout["num_steps"].float().mean()),
            "statistics/game_steps_mean": self.reduce_mean(game_steps_info.float().mean()),
        }
        self.capture_metrics(metrics)

    def resume(self, state: CheckpointState) -> None:
        self._state = state
        self.resume_policy.resume(state)

    def checkpoint_due(self) -> bool:
        return self.checkpoint_policy.due(self._require_iteration())

    def save_checkpoint(self, state: CheckpointState) -> None:
        self.checkpoint_policy.save(state)

    def evaluation_due(self) -> bool:
        return self.evaluator.should_run(self._require_iteration())

    def evaluate(self, *, agent_policy: torch.nn.Module) -> None:
        self.evaluator.handle_step(
            self._require_iteration(),
            env_steps = int(self.env_steps_total),
            game_steps = int(self.game_steps_total),
            agent_policy = agent_policy,
        )


__all__ = [
    "CheckpointManager",
    "TrainingWorkflow",
    "ResumePolicy",
    "CheckpointPolicy",
]
