from __future__ import annotations

import math
import torch
import torch.nn as nn

from marvel.model import Agent
from marvel.optim.common import ExponentialMovingAverageUpdater, apply_learning_rate
from marvel.runtime.checkpoints import CheckpointState
from marvel.training.gae import compute_gae_from_value
from marvel.training.losses import compute_clipped_pg_loss, compute_value_loss
from marvel.training.trainer import AlgorithmTrainer


class IPPOTrainer(AlgorithmTrainer):
    def run(self) -> None:
        cfg = self.cfg
        runtime = self._make_runtime()
        self._set_seed(runtime)

        with runtime:
            agent_model_base = Agent(cfg, enable_value_head = True).to(runtime.device)
            agent_model_merged = Agent(cfg, enable_value_head = True).to(runtime.device)
            agent_model_merge_updater = ExponentialMovingAverageUpdater(
                model = agent_model_base,
                merged_model = agent_model_merged,
                beta = cfg.agent_merging_beta,
            )

            agent_optimizer = agent_model_base.configure_optimizer(lr = cfg.ppo.base_lr, optimizer = cfg.optimizer)

            state = CheckpointState({
                "agent_resume_state_dict": agent_model_base,
                "agent_state_dict": agent_model_merged,
                "agent_optimizer_state_dict": agent_optimizer,
                "iteration": 0,
                "total_env_steps": 0,
                "total_game_steps": 0,
            })
            runtime.resume(state)
            agent_model_merge_updater.set_reference_iteration(state.iteration)

            agent_model = runtime.prepare(agent_model_base)

            policy_epochs = cfg.ppo.agent_epoch

            for iteration in runtime:
                sched = self._compute_agent_schedule(iteration)
                apply_learning_rate(agent_optimizer, sched.agent_lr)

                rollout = runtime.collect_trajectories(agent_model = agent_model)
                runtime.log_rollouts(rollout = rollout)
                collect_batch_size = int(rollout["num_steps"].shape[0])
                mini_batch_size = max(1, math.ceil(collect_batch_size / cfg.mini_batches))

                with torch.no_grad():
                    roll_agent_output = agent_model(rollout = rollout)

                    roll_state_values = roll_agent_output.state_values.detach().float()
                    roll_gae = compute_gae_from_value(
                        rollout,
                        roll_state_values,
                        discount_gamma = cfg.discount_gamma,
                        gae_lambda = cfg.ppo.gae_lambda,
                    )

                log_param_stats = runtime.param_stats_due()

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
                        pg_loss = compute_clipped_pg_loss(
                            mb_rollout,
                            clip_coef = sched.clip_coef,
                            avail_probs_new = mb_avail_probs_new,
                            player_advantage = roll_gae.player_advantage[mb_indices],
                        )
                        value_loss = compute_value_loss(
                            mb_rollout,
                            player_state_value_pred = mb_agent_output.player_state_value,
                            player_state_value_target = roll_gae.player_state_value[mb_indices],
                        )

                        agent_loss_mean = (
                            pg_loss.pg_loss_mean
                            + sched.reg_coef * pg_loss.kl_unif_mean
                            + cfg.value_loss_coef * value_loss.value_loss_mean
                        )

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
                                    f"{tag}/value_loss": runtime.reduce_mean(value_loss.value_loss_mean),
                                    f"{tag}/adv_mean": runtime.reduce_mean(pg_loss.adv_mean),
                                    f"{tag}/adv_std": runtime.reduce_mean(pg_loss.adv_std),
                                }
                            )

                agent_model_merge_updater.step(iteration = iteration)

                runtime.capture_metrics(
                    {
                        "agent/learning_rate": sched.agent_lr,
                        "agent/clip_coef": sched.clip_coef,
                        "agent/reg_coef": sched.reg_coef,
                    }
                )

                rollout = None
                roll_agent_output = None
                roll_state_values = None
                roll_gae = None
                runtime.log_metrics()
                runtime.progress_bar_step()
                if runtime.checkpoint_due():
                    runtime.save_checkpoint(state)
                if runtime.evaluation_due():
                    runtime.evaluate(agent_policy = agent_model_merged)


def run(cfg) -> None:
    IPPOTrainer(cfg).run()


__all__ = ["IPPOTrainer", "run"]
