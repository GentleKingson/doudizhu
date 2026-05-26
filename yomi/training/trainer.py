from __future__ import annotations

import torch

from marvel.optim.common import apply_learning_rate
from marvel.runtime.checkpoints import CheckpointState
from marvel.training.trainer import AlgorithmTrainer

from yomi.config import TrainConfig
from yomi.model import BeliefDecoderConcat
from yomi.training.step import compute_batch_metrics, run_training_step
from yomi.training.workflow import YomiTrainingWorkflow


torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


class BeliefTrainer(AlgorithmTrainer):
    """Yomi supervised-belief trainer built on MARVEL runtime/trainer contracts."""

    cfg: TrainConfig

    def _default_runtime_factory(self, cfg: TrainConfig) -> YomiTrainingWorkflow:
        return YomiTrainingWorkflow(cfg, game = cfg.game)

    @staticmethod
    def _batch_scalar(batch: object, key: str) -> int:
        tensor = batch[key]  # type: ignore[index]
        if int(tensor.numel()) <= 0:
            raise RuntimeError(f"Collection metadata field {key!r} is empty.")
        return int(tensor.reshape(-1)[0].detach().cpu().item())

    def run(self) -> None:
        runtime = self._make_runtime()
        self._set_seed(runtime)

        with runtime:
            device = runtime.device
            pad_token = int(runtime.game.PAD_TOKEN)

            policy_checkpoint_path = runtime.resolve_policy_checkpoint_path(self.cfg.policy_checkpoint_path)
            policy = runtime.load_frozen_policy(checkpoint_path = policy_checkpoint_path)

            model_base = BeliefDecoderConcat(game = runtime.game, model_cfg = self.cfg.model).to(device)
            optimizer = model_base.configure_optimizer(
                lr = float(self.cfg.base_lr),
                optimizer = self.cfg.optimizer,
            )
            model = runtime.prepare(model_base)

            state = CheckpointState({
                "model_state_dict": model,
                "optimizer_state_dict": optimizer,
                "iteration": 0,
            })
            runtime.resume(state)

            action_root_collector = runtime.make_action_root_collector(
                policy = policy,
                pad_token = pad_token,
            )
            collect_warmup_steps = int(runtime.collect_warmup_steps)
            if collect_warmup_steps > 0:
                local_rollout_batch_size = int(runtime.local_rollout_batch_size)
                action_root_collector.warm_start_staggered(
                    warmup_steps = collect_warmup_steps,
                    global_env_start = int(runtime.rank) * local_rollout_batch_size,
                    global_env_count = int(self.cfg.rollout_batch_size),
                    seed = int(self.cfg.seed) + 104729 * (int(runtime.rank) + 1),
                )

            total_iterations = int(self.cfg.total_iterations)
            stable_iters = int(total_iterations * float(self.cfg.lr_stable_fraction))
            decay_denominator = max(total_iterations - stable_iters, 1)
            validation_interval = int(self.cfg.validation_interval)
            if validation_interval <= 0:
                raise ValueError(f"validation_interval must be positive, got {validation_interval}.")
            first_validation_iteration = min(validation_interval, total_iterations)
            collect_interval = int(runtime.collect_interval)
            validation_batch = None

            for iteration in runtime:
                lr_scale = (
                    1.0
                    if iteration <= stable_iters
                    else max(0.0, float(total_iterations - iteration) / float(decay_denominator))
                )
                current_lr = float(self.cfg.base_lr) * float(lr_scale)
                apply_learning_rate(optimizer, current_lr)

                batch = action_root_collector.collect(collect_interval = collect_interval)
                step_metrics = run_training_step(
                    model = model,
                    optimizer = optimizer,
                    batch = batch,
                    pad_token = pad_token,
                    grad_max_norm = float(self.cfg.optimizer.grad_max_norm),
                    reward_to_go_loss_weight = float(self.cfg.reward_to_go_loss_weight),
                )

                batch_n = int(batch.batch_size[0])
                local_loss_weighted = float(step_metrics["loss"]) * float(batch_n)
                local_belief_loss_weighted = float(step_metrics.get("belief_loss", step_metrics["loss"])) * float(batch_n)
                local_accuracy_weighted = float(step_metrics["token_accuracy"]) * float(batch_n)
                local_reward_loss_weighted = (
                    float(step_metrics["reward_to_go_loss"]) * float(batch_n)
                    if "reward_to_go_loss" in step_metrics
                    else None
                )
                local_reward_mae_weighted = (
                    float(step_metrics["reward_to_go_mae"]) * float(batch_n)
                    if "reward_to_go_mae" in step_metrics
                    else None
                )
                local_completed_episodes = self._batch_scalar(batch, "collection_completed_episodes")
                local_env_steps = self._batch_scalar(batch, "collection_env_steps")
                local_interval_chunks = self._batch_scalar(batch, "collection_interval_chunks")

                collected_rows = int(runtime.reduce_sum(batch_n) or 0)
                if collected_rows <= 0:
                    raise RuntimeError("No training sequences after cross-rank reduction.")

                loss_weighted = float(runtime.reduce_sum(float(local_loss_weighted)) or 0.0)
                belief_loss_weighted = float(runtime.reduce_sum(float(local_belief_loss_weighted)) or 0.0)
                accuracy_weighted = float(runtime.reduce_sum(float(local_accuracy_weighted)) or 0.0)
                loss_value = float(loss_weighted / float(collected_rows))
                belief_loss_value = float(belief_loss_weighted / float(collected_rows))
                token_accuracy = float(accuracy_weighted / float(collected_rows))
                completed_episodes = int(runtime.reduce_sum(local_completed_episodes) or 0)
                env_steps = int(runtime.reduce_sum(local_env_steps) or 0)
                interval_chunks = int(runtime.reduce_sum(local_interval_chunks) or 0)

                metrics = {
                    "belief_train/loss": float(loss_value),
                    "belief_train/nll_loss": float(belief_loss_value),
                    "belief_train/token_acc": float(token_accuracy),
                    "belief_train/learning_rate": float(current_lr),
                    "belief_train/root_count": float(collected_rows),
                    "belief_train/completed_episodes": float(completed_episodes),
                    "belief_train/env_steps": float(env_steps),
                    "belief_train/interval_chunks": float(interval_chunks),
                }
                if local_reward_loss_weighted is not None:
                    reward_loss_weighted = float(runtime.reduce_sum(float(local_reward_loss_weighted)) or 0.0)
                    metrics["belief_train/rtg_loss"] = float(reward_loss_weighted / float(collected_rows))
                if local_reward_mae_weighted is not None:
                    reward_mae_weighted = float(runtime.reduce_sum(float(local_reward_mae_weighted)) or 0.0)
                    metrics["belief_train/rtg_mae"] = float(reward_mae_weighted / float(collected_rows))
                validation_due = iteration >= first_validation_iteration and (
                    iteration == first_validation_iteration or iteration % validation_interval == 0
                )
                if validation_due:
                    if validation_batch is None:
                        validation_batch = action_root_collector.collect(
                            collect_interval = collect_interval,
                        )
                    validation_metrics = compute_batch_metrics(
                        model = model,
                        batch = validation_batch,
                        pad_token = pad_token,
                        reward_to_go_loss_weight = float(self.cfg.reward_to_go_loss_weight),
                    )
                    validation_batch_n = int(validation_batch.batch_size[0])
                    total_validation_rows = int(runtime.reduce_sum(validation_batch_n) or 0)
                    if total_validation_rows <= 0:
                        raise RuntimeError("No validation sequences after cross-rank reduction.")
                    validation_loss_weighted = float(validation_metrics["loss"]) * float(validation_batch_n)
                    validation_belief_loss_weighted = (
                        float(validation_metrics.get("belief_loss", validation_metrics["loss"]))
                        * float(validation_batch_n)
                    )
                    validation_accuracy_weighted = float(validation_metrics["token_accuracy"]) * float(validation_batch_n)
                    validation_loss_sum = float(runtime.reduce_sum(float(validation_loss_weighted)) or 0.0)
                    validation_belief_loss_sum = float(
                        runtime.reduce_sum(float(validation_belief_loss_weighted)) or 0.0
                    )
                    validation_accuracy_sum = float(runtime.reduce_sum(float(validation_accuracy_weighted)) or 0.0)
                    metrics["belief_val/loss"] = float(validation_loss_sum / float(total_validation_rows))
                    metrics["belief_val/nll_loss"] = float(validation_belief_loss_sum / float(total_validation_rows))
                    metrics["belief_val/token_acc"] = float(validation_accuracy_sum / float(total_validation_rows))
                    if "reward_to_go_loss" in validation_metrics:
                        validation_reward_loss_weighted = (
                            float(validation_metrics["reward_to_go_loss"]) * float(validation_batch_n)
                        )
                        validation_reward_loss_sum = float(
                            runtime.reduce_sum(float(validation_reward_loss_weighted)) or 0.0
                        )
                        metrics["belief_val/rtg_loss"] = float(
                            validation_reward_loss_sum / float(total_validation_rows)
                        )
                    if "reward_to_go_mae" in validation_metrics:
                        validation_reward_mae_weighted = (
                            float(validation_metrics["reward_to_go_mae"]) * float(validation_batch_n)
                        )
                        validation_reward_mae_sum = float(
                            runtime.reduce_sum(float(validation_reward_mae_weighted)) or 0.0
                        )
                        metrics["belief_val/rtg_mae"] = float(
                            validation_reward_mae_sum / float(total_validation_rows)
                        )

                runtime.capture_metrics(metrics)
                if runtime.param_stats_due():
                    runtime.capture_param_stats(model_base, "belief_stats")
                runtime.log_metrics()
                runtime.progress_bar_step()

                if runtime.checkpoint_due():
                    runtime.save_checkpoint(state)


__all__ = ["BeliefTrainer"]
