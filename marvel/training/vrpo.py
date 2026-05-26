from __future__ import annotations

import math

import torch
import torch.nn as nn

from marvel.buffer import BufferConfig, TrajectoryBuffer, trajectory_field_specs
from marvel.config import TrainConfig
from marvel.model import Agent, CentralCritic
from marvel.optim.common import ExponentialMovingAverageUpdater, apply_learning_rate
from marvel.runtime.checkpoints import CheckpointState
from marvel.training.gae import compute_gae_from_qvalue
from marvel.training.losses import compute_clipped_pg_loss, compute_qvalue_loss
from marvel.training.trainer import AlgorithmTrainer
from marvel.utils import get_game_attr


class VRPOTrainer(AlgorithmTrainer):
    def run(self) -> None:
        cfg = self.cfg
        runtime = self._make_runtime()
        self._set_seed(runtime)
        game = cfg.game
        rollout_buffer_device = torch.device(cfg.vrpo.replay_buffer_device)

        with runtime:
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

            for iteration in runtime:
                sched = self._compute_agent_schedule(iteration)
                critic_lr = cfg.vrpo.base_lr * sched.critic_lr_scale
                apply_learning_rate(agent_optimizer, sched.agent_lr)
                apply_learning_rate(critic_optimizer, critic_lr)

                rollout = runtime.collect_trajectories(agent_model = agent_model)
                runtime.log_rollouts(rollout = rollout)
                collect_batch_size = int(rollout["num_steps"].shape[0])
                mini_batch_size = max(1, math.ceil(collect_batch_size / cfg.mini_batches))

                with torch.no_grad():
                    roll_critic_output = critic_model(rollout = rollout)
                    roll_qvalues = roll_critic_output.qvalues.detach().float()

                replay_buffer.add(rollout)

                log_param_stats = runtime.param_stats_due()

                # Policy updates
                for epoch in range(policy_epochs):
                    indices = torch.randperm(collect_batch_size, device = runtime.device)

                    for start in range(0, collect_batch_size, mini_batch_size):
                        end = min(start + mini_batch_size, collect_batch_size)
                        mb_indices = indices[start:end]
                        if mb_indices.shape[0] == 0:
                            continue

                        is_first = (epoch == 0 and start == 0)
                        is_last = (epoch == policy_epochs - 1 and end >= collect_batch_size)

                        mb_rollout = rollout[mb_indices]

                        mb_agent_output = agent_model(rollout = mb_rollout)

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

                        agent_optimizer.zero_grad()
                        agent_loss_mean.backward()
                        nn.utils.clip_grad_norm_(agent_model_base.parameters(), cfg.optimizer.grad_max_norm)
                        if is_first and log_param_stats:
                            runtime.capture_param_stats(agent_model_base, "agent_stats")
                        agent_optimizer.step()

                        if is_first or is_last:
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
                collect_batch = rollout.select(*replay_keys).to(runtime.device)
                for aux_ep in range(aux_phase_epochs):
                    total_trajectories = int(collect_batch.batch_size[0])
                    if total_trajectories == 0:
                        continue

                    perm = torch.randperm(total_trajectories, device = runtime.device)

                    for start in range(0, total_trajectories, mini_batch_size):
                        end = min(start + mini_batch_size, total_trajectories)
                        batch_idx = perm[start:end]
                        if batch_idx.numel() == 0:
                            continue

                        is_first_critic = (aux_ep == 0 and start == 0)
                        is_last_critic = (aux_ep == aux_phase_epochs - 1 and end >= collect_batch_size)

                        mb_rollout = collect_batch[batch_idx]

                        with torch.no_grad():
                            mb_agent_output = agent_model(rollout = mb_rollout)
                            mb_avail_probs_new = mb_agent_output.avail_probs.float()

                        mb_critic_out = critic_model(rollout = mb_rollout)
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

                        critic_optimizer.zero_grad()
                        qv_lbd_loss_mean.backward()
                        nn.utils.clip_grad_norm_(critic_model_base.parameters(), cfg.optimizer.grad_max_norm)
                        if is_first_critic and log_param_stats:
                            runtime.capture_param_stats(critic_model_base, "critic_stats")
                        critic_optimizer.step()

                        if is_first_critic or is_last_critic:
                            tag = "critic_initial" if is_first_critic else "critic"
                            runtime.capture_metrics(
                                {
                                    f"{tag}/qv_lbd_loss": runtime.reduce_mean(qv_lbd_loss_mean),
                                    f"{tag}/qv_td1_loss": runtime.reduce_mean(qv_td1_loss_mean),
                                }
                            )

                    # Sample replay for next epoch.
                    collect_batch = replay_buffer.sample(
                        collect_batch_size,
                        keys = replay_keys,
                    )
                    collect_batch = collect_batch.to(runtime.device)

                agent_model_merge_updater.step(iteration = iteration)

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
                if runtime.checkpoint_due():
                    runtime.save_checkpoint(state)
                if runtime.evaluation_due():
                    runtime.evaluate(agent_policy = agent_model_merged)


def run(cfg) -> None:
    VRPOTrainer(cfg).run()


__all__ = ["VRPOTrainer", "run"]
