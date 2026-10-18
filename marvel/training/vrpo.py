from __future__ import annotations

import math
import json
import os
import time
from pathlib import Path
from contextlib import ExitStack

import torch
import torch.nn as nn

from marvel.buffer import BufferConfig, TrajectoryBuffer, trajectory_field_specs
from marvel.config import TrainConfig
from marvel.model import Agent, CentralCritic
from marvel.optim.common import ExponentialMovingAverageUpdater, apply_learning_rate
from marvel.runtime.checkpoints import CheckpointState
from marvel.runtime.profiling import VRPODiagnostics
from marvel.training.gae import compute_gae_from_qvalue
from marvel.training.losses import compute_clipped_pg_loss, compute_qvalue_loss
from marvel.training.trainer import AlgorithmTrainer
from marvel.utils import get_game_attr


class VRPOTrainer(AlgorithmTrainer):
    def run(self) -> None:
        cfg = self.cfg
        with ExitStack() as stack:
            diagnostics = stack.enter_context(VRPODiagnostics(cfg))
            runtime = self._make_runtime()
            self._set_seed(runtime)
            game = cfg.game
            rollout_buffer_device = torch.device(cfg.vrpo.replay_buffer_device)
            stack.enter_context(runtime)
            agent_model_base = Agent(cfg, enable_value_head = False).to(runtime.device)
            critic_model_base = CentralCritic(cfg, output_mode = "qvalue").to(runtime.device)
            agent_model_merged = Agent(cfg, enable_value_head = False).to(runtime.device)
            agent_model_merge_updater = ExponentialMovingAverageUpdater(
                model = agent_model_base,
                merged_model = agent_model_merged,
                beta = cfg.agent_merging_beta,
            )

            agent_optimizer = agent_model_base.configure_optimizer(lr = cfg.ppo.base_lr, optimizer = cfg.optimizer)
            critic_optimizer = critic_model_base.configure_optimizer(lr = cfg.vrpo.base_lr, optimizer = cfg.optimizer)

            state = CheckpointState({
                "agent_resume_state_dict": agent_model_base,
                "critic_state_dict": critic_model_base,
                "agent_state_dict": agent_model_merged,
                "agent_optimizer_state_dict": agent_optimizer,
                "critic_optimizer_state_dict": critic_optimizer,
                "iteration": 0,
                "total_env_steps": 0,
                "total_game_steps": 0,
            })
            if not cfg.full_state_checkpoints:
                runtime.resume(state)
                agent_model_merge_updater.set_reference_iteration(state.iteration)

            agent_model = runtime.prepare(agent_model_base)
            critic_model = runtime.prepare(critic_model_base)

            P = int(get_game_attr(game, "num_players"))

            policy_epochs = cfg.ppo.agent_epoch
            aux_phase_epochs = cfg.vrpo.aux_epoch

            per_rank_capacity = max(cfg.replay_buffer_scale * cfg.collect_batch_size // runtime.world_size, 1)
            specs = trajectory_field_specs(game)
            buffer_config = BufferConfig(
                game = cfg.game,
                device = rollout_buffer_device,
                field_specs = specs,
            )
            replay_buffer = TrajectoryBuffer(per_rank_capacity, buffer_config)
            if cfg.full_state_checkpoints:
                from marvel.runtime.full_state import restore, model_hash, write_json
                runtime.full_replay = replay_buffer
                runtime.full_ema = agent_model_merge_updater
                state["actor_optimizer_steps"] = 0
                state["critic_optimizer_steps"] = 0
                if cfg.resume_from:
                    restore(runtime, state, replay_buffer, agent_model_merge_updater, cfg.resume_from)
                else:
                    runtime._state = state
                    initial_models = {
                        "seed": cfg.seed, "actor_sha256": model_hash(agent_model_base),
                        "critic_sha256": model_hash(critic_model_base)}
                    expected = getattr(runtime, "expected_initial_models", initial_models)
                    if initial_models != expected:
                        raise ValueError("Initial Actor/Critic hashes do not match this seed's frozen pairing")
                    write_json(Path(cfg.logging_dir)/"initial_models.json", initial_models)

            for iteration in runtime:
                iteration_started = time.monotonic()
                with diagnostics.iteration(iteration, runtime.sampler if diagnostics.enabled else None):
                    with diagnostics.scope("vrpo.schedule"):
                        sched = self._compute_agent_schedule(iteration)
                        critic_lr = cfg.vrpo.base_lr * sched.critic_lr_scale
                        apply_learning_rate(agent_optimizer, sched.agent_lr)
                        apply_learning_rate(critic_optimizer, critic_lr)

                    with diagnostics.scope("vrpo.rollout.collect"):
                        rollout = runtime.collect_trajectories(agent_model = agent_model)

                    with diagnostics.scope("vrpo.rollout.metrics"):
                        runtime.log_rollouts(rollout = rollout)

                    collect_batch_size = int(rollout["num_steps"].shape[0])
                    mini_batch_size = max(1, math.ceil(collect_batch_size / cfg.mini_batches))

                    with diagnostics.scope("vrpo.bootstrap.critic_forward"):
                        with torch.no_grad():
                            roll_critic_output = critic_model(rollout = rollout)
                            roll_qvalues = roll_critic_output.qvalues.detach().float()

                    with diagnostics.scope("vrpo.replay.add"):
                        replay_buffer.add(rollout)

                    log_param_stats = runtime.param_stats_due()

                    # Policy updates
                    for epoch in range(policy_epochs):
                        with diagnostics.scope("vrpo.policy.batch_prepare"):
                            indices = torch.randperm(collect_batch_size, device = runtime.device)

                        for start in range(0, collect_batch_size, mini_batch_size):
                            with diagnostics.scope("vrpo.policy.batch_prepare"):
                                end = min(start + mini_batch_size, collect_batch_size)
                                mb_indices = indices[start:end]
                                if mb_indices.shape[0] == 0:
                                    continue

                                is_first = (epoch == 0 and start == 0)
                                is_last = (epoch == policy_epochs - 1 and end >= collect_batch_size)

                                mb_rollout = rollout[mb_indices]

                            with diagnostics.scope("vrpo.policy.forward"):
                                mb_agent_output = agent_model(rollout = mb_rollout)

                            with diagnostics.scope("vrpo.policy.gae_loss"):
                                mb_avail_probs_new = mb_agent_output.avail_probs.float()
                                mb_qvalues = roll_qvalues[mb_indices]

                                with torch.no_grad():
                                    mb_gae = compute_gae_from_qvalue(
                                        mb_rollout,
                                        mb_qvalues,
                                        avail_probs_new = mb_avail_probs_new,
                                        discount_gamma = cfg.discount_gamma,
                                        gae_lambda = cfg.ppo.gae_lambda,
                                    )

                                pg_loss = compute_clipped_pg_loss(
                                    mb_rollout,
                                    clip_coef = sched.clip_coef,
                                    avail_probs_new = mb_avail_probs_new,
                                    player_advantage = mb_gae.player_advantage,
                                )

                                agent_loss_mean = pg_loss.pg_loss_mean + sched.reg_coef * pg_loss.kl_unif_mean
                                if cfg.full_state_checkpoints and not torch.isfinite(agent_loss_mean).item():
                                    raise FloatingPointError("Non-finite Actor loss")

                            with diagnostics.scope("vrpo.policy.zero_grad"):
                                agent_optimizer.zero_grad()

                            with diagnostics.scope("vrpo.policy.backward"):
                                agent_loss_mean.backward()

                            with diagnostics.scope("vrpo.policy.grad_clip"):
                                nn.utils.clip_grad_norm_(agent_model_base.parameters(), cfg.optimizer.grad_max_norm,
                                                         error_if_nonfinite=cfg.full_state_checkpoints)

                            if is_first and log_param_stats:
                                with diagnostics.scope("vrpo.policy.param_stats"):
                                    runtime.capture_param_stats(agent_model_base, "agent_stats")

                            with diagnostics.scope("vrpo.policy.optimizer"):
                                agent_optimizer.step()
                                if cfg.full_state_checkpoints:
                                    state["actor_optimizer_steps"] += 1

                            if is_first or is_last:
                                with diagnostics.scope("vrpo.policy.metrics"):
                                    tag = "agent_initial" if is_first else "agent"
                                    runtime.capture_metrics(
                                        {
                                            f"{tag}/pg_loss": runtime.reduce_mean(pg_loss.pg_loss_mean),
                                            f"{tag}/kl_unif": runtime.reduce_mean(pg_loss.kl_unif_mean),
                                            f"{tag}/kl_ref": runtime.reduce_mean(pg_loss.kl_ref_mean),
                                            f"{tag}/clip_frac": runtime.reduce_mean(pg_loss.clip_frac),
                                            f"{tag}/adv_mean": runtime.reduce_mean(pg_loss.adv_mean),
                                            f"{tag}/adv_std": runtime.reduce_mean(pg_loss.adv_std),
                                        }
                                    )

                    # Critic updates — first epoch uses fresh rollout (on-policy),
                    # remaining epochs sample uniformly from replay.
                    replay_keys = (
                        "num_steps",
                        "player_ids",
                        "obs_tokens",
                        "obs_channels",
                        "avail_tokens",
                        "avail_weights",
                        "action_taken",
                        "rewards",
                    )
                    with diagnostics.scope("vrpo.critic.batch_prepare"):
                        collect_batch = rollout.select(*replay_keys).to(runtime.device)

                    for aux_ep in range(aux_phase_epochs):
                        with diagnostics.scope("vrpo.critic.batch_prepare"):
                            total_trajectories = int(collect_batch.batch_size[0])
                            if total_trajectories == 0:
                                continue

                            perm = torch.randperm(total_trajectories, device = runtime.device)

                        for start in range(0, total_trajectories, mini_batch_size):
                            with diagnostics.scope("vrpo.critic.batch_prepare"):
                                end = min(start + mini_batch_size, total_trajectories)
                                batch_idx = perm[start:end]
                                if batch_idx.numel() == 0:
                                    continue

                                is_first_critic = (aux_ep == 0 and start == 0)
                                is_last_critic = (aux_ep == aux_phase_epochs - 1 and end >= collect_batch_size)

                                mb_rollout = collect_batch[batch_idx]

                            with diagnostics.scope("vrpo.critic.agent_forward"):
                                with torch.no_grad():
                                    mb_agent_output = agent_model(rollout = mb_rollout)
                                    mb_avail_probs_new = mb_agent_output.avail_probs.float()

                            with diagnostics.scope("vrpo.critic.forward"):
                                mb_critic_out = critic_model(rollout = mb_rollout)

                            with diagnostics.scope("vrpo.critic.gae_loss"):
                                mb_qvalues = mb_critic_out.qvalues.float()

                                with torch.no_grad():
                                    mb_gae = compute_gae_from_qvalue(
                                        mb_rollout,
                                        mb_qvalues,
                                        avail_probs_new = mb_avail_probs_new,
                                        discount_gamma = cfg.discount_gamma,
                                        gae_lambda = cfg.vrpo.gae_lambda,
                                    )

                                qvalue_loss = compute_qvalue_loss(
                                    mb_rollout,
                                    mb_qvalues,
                                    qvalue_lbd_true = mb_gae.qvalue_lbd_true,
                                    qvalue_td1_true = mb_gae.qvalue_td1_true,
                                )
                                qv_lbd_loss_mean = qvalue_loss.qv_lbd_loss_mean
                                qv_td1_loss_mean = qvalue_loss.qv_td1_loss_mean
                                if cfg.full_state_checkpoints and not torch.isfinite(qv_lbd_loss_mean).item():
                                    raise FloatingPointError("Non-finite Critic loss")

                            with diagnostics.scope("vrpo.critic.zero_grad"):
                                critic_optimizer.zero_grad()

                            with diagnostics.scope("vrpo.critic.backward"):
                                qv_lbd_loss_mean.backward()

                            with diagnostics.scope("vrpo.critic.grad_clip"):
                                nn.utils.clip_grad_norm_(critic_model_base.parameters(), cfg.optimizer.grad_max_norm,
                                                         error_if_nonfinite=cfg.full_state_checkpoints)

                            if is_first_critic and log_param_stats:
                                with diagnostics.scope("vrpo.critic.param_stats"):
                                    runtime.capture_param_stats(critic_model_base, "critic_stats")

                            with diagnostics.scope("vrpo.critic.optimizer"):
                                critic_optimizer.step()
                                if cfg.full_state_checkpoints:
                                    state["critic_optimizer_steps"] += 1

                            if is_first_critic or is_last_critic:
                                with diagnostics.scope("vrpo.critic.metrics"):
                                    tag = "critic_initial" if is_first_critic else "critic"
                                    runtime.capture_metrics(
                                        {
                                            f"{tag}/qv_lbd_loss": runtime.reduce_mean(qv_lbd_loss_mean),
                                            f"{tag}/qv_td1_loss": runtime.reduce_mean(qv_td1_loss_mean),
                                        }
                                    )

                        # Sample replay for next epoch.
                        with diagnostics.scope("vrpo.critic.replay_sample"):
                            collect_batch = replay_buffer.sample(
                                collect_batch_size,
                                keys = replay_keys,
                            )
                            collect_batch = collect_batch.to(runtime.device)

                    with diagnostics.scope("vrpo.ema"):
                        agent_model_merge_updater.step(iteration = iteration)

                    with diagnostics.scope("vrpo.log_and_save"):
                        runtime.capture_metrics(
                            {
                                "agent/learning_rate": sched.agent_lr,
                                "agent/clip_coef": sched.clip_coef,
                                "agent/reg_coef": sched.reg_coef,
                                "critic/learning_rate": critic_lr,
                                "buffer/replay_buffer_size": runtime.reduce_sum(len(replay_buffer)),
                            }
                        )

                        rollout = None
                        roll_qvalues = None
                        collect_batch = None
                        runtime.log_metrics()
                        runtime.progress_bar_step()
                        stop_path = os.environ.get("MARVEL_STOP_AFTER_BOUNDARY")
                        stop_requested = bool(cfg.full_state_checkpoints and stop_path and Path(stop_path).is_file())
                        if runtime.checkpoint_due() or stop_requested:
                            runtime.save_checkpoint(state)
                        if runtime.evaluation_due():
                            runtime.evaluate(agent_policy = agent_model_merged)
                        if cfg.full_state_checkpoints:
                            if runtime.device.type == "cuda":
                                torch.cuda.synchronize()
                            record = {
                                "iteration": iteration, "iteration_wall_seconds_including_checkpoint_wait": time.monotonic()-iteration_started,
                                "consumed_trajectories": cfg.collect_batch_size*iteration,
                                "native_env_steps": runtime.sampler.total_env_steps,
                                "valid_game_steps": runtime.sampler.total_game_steps,
                                "completed_games": runtime.sampler.total_completed_games,
                                "commit_leftovers": runtime.sampler.commit_buffer_size,
                                "actor_optimizer_steps": state["actor_optimizer_steps"],
                                "critic_optimizer_steps": state["critic_optimizer_steps"],
                                "ema_updates": agent_model_merge_updater.update_count,
                                "replay_size": len(replay_buffer), "replay_ptr": replay_buffer._ptr,
                            }
                            with (Path(cfg.logging_dir)/"iterations.jsonl").open("a") as stream:
                                stream.write(json.dumps(record, allow_nan=False)+"\n")
                        if stop_requested:
                            print(f"[FullState] stopped at complete iteration {iteration}", flush=True)
                            break


def run(cfg) -> None:
    VRPOTrainer(cfg).run()


__all__ = ["VRPOTrainer", "run"]
