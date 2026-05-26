from __future__ import annotations

import torch

from marvel.model.tensordict import TensorDict
from marvel.optim.common import MultiOptimizer

from yomi.training.loss import masked_belief_candidate_nll, masked_reward_to_go_huber


_MODEL_BATCH_FIELDS = (
    "obs_prefix_tokens",
    "obs_prefix_mask",
    "state_channels",
    "belief_input_tokens",
    "belief_target_tokens",
    "belief_length",
    "candidate_tokens",
    "candidate_valid",
)


def _mark_batch_axis_dynamic(tensor: torch.Tensor) -> None:
    dynamo = getattr(torch, "_dynamo", None)
    mark_dynamic = getattr(dynamo, "maybe_mark_dynamic", None)
    if mark_dynamic is None:
        mark_dynamic = getattr(dynamo, "mark_dynamic", None)
    if mark_dynamic is None or tensor.ndim <= 0:
        return
    mark_dynamic(tensor, 0)


def _mark_model_batch_axis_dynamic(batch: TensorDict) -> None:
    for key in _MODEL_BATCH_FIELDS:
        _mark_batch_axis_dynamic(batch[key])


def _build_reward_to_go_input_tokens(*, batch: TensorDict, pad_token: int) -> torch.Tensor:
    target_tokens = batch["belief_target_tokens"]
    if target_tokens.ndim != 3:
        raise RuntimeError(f"belief_target_tokens must be [B, S, E], got {tuple(target_tokens.shape)}")

    batch_size, max_rows, row_width = target_tokens.shape
    belief_length = batch["belief_length"].to(dtype = torch.long)
    if belief_length.shape != (batch_size,):
        raise RuntimeError(f"belief_length must be [B], got {tuple(belief_length.shape)} for B={batch_size}.")
    if bool(((belief_length <= 0) | (belief_length > int(max_rows))).any().item()):
        raise RuntimeError(f"belief_length must be in [1, {int(max_rows)}] for reward-to-go prediction.")

    reward_input = torch.full(
        (batch_size, max_rows + 1, row_width),
        fill_value = int(pad_token),
        dtype = target_tokens.dtype,
        device = target_tokens.device,
    )
    reward_input[:, 1:, :] = target_tokens
    return reward_input


def _compute_loss_output_from_batch(
    *,
    model: torch.nn.Module,
    batch: TensorDict,
    pad_token: int,
    reward_to_go_loss_weight: float,
) -> dict[str, torch.Tensor | int | None]:
    reward_weight = float(reward_to_go_loss_weight)
    if reward_weight < 0.0:
        raise RuntimeError(f"reward_to_go_loss_weight must be non-negative, got {reward_weight}.")

    _mark_model_batch_axis_dynamic(batch)
    reward_to_go_input_tokens = _build_reward_to_go_input_tokens(batch = batch, pad_token = int(pad_token))
    _mark_batch_axis_dynamic(reward_to_go_input_tokens)
    out = model(
        obs_prefix_tokens = batch["obs_prefix_tokens"],
        obs_prefix_mask = batch["obs_prefix_mask"],
        state_channels = batch["state_channels"],
        belief_input_tokens = batch["belief_input_tokens"],
        candidate_tokens = batch["candidate_tokens"],
        candidate_valid = batch["candidate_valid"],
        reward_to_go_input_tokens = reward_to_go_input_tokens,
    )

    row_has_tokens = batch["belief_target_tokens"][:, :, 0] != int(pad_token)
    target_mask_bool = batch["belief_target_mask"].to(dtype = torch.bool)
    if bool((row_has_tokens & ~target_mask_bool).any().item()):
        raise RuntimeError("belief_target_mask excludes non-PAD target rows.")
    if bool((target_mask_bool & ~row_has_tokens).any().item()):
        raise RuntimeError("belief_target_mask includes PAD target rows.")

    target_index = batch["belief_target_index"].to(dtype = torch.long)
    if bool((target_mask_bool & (target_index < 0)).any().item()):
        raise RuntimeError("belief_target_index must be non-negative for active belief rows.")
    if bool((~target_mask_bool & (target_index != -1)).any().item()):
        raise RuntimeError("belief_target_index must be -1 for inactive belief rows.")

    belief_loss = masked_belief_candidate_nll(
        logits = out.logits,
        target_index = target_index,
        target_mask = batch["belief_target_mask"],
        candidate_valid = batch["candidate_valid"],
    )
    total_loss = belief_loss.loss

    reward_loss = None
    reward_mae = None
    reward_targets = batch.get("reward_to_go")
    reward_mask = batch.get("reward_to_go_mask")
    reward_pred = getattr(out, "reward_to_go", None)
    if reward_pred is not None and reward_targets is not None and reward_mask is not None:
        reward_loss_out = masked_reward_to_go_huber(
            pred = reward_pred,
            target = reward_targets.to(dtype = torch.float32),
            mask = reward_mask,
        )
        if reward_weight > 0.0:
            sequence_dtype = belief_loss.sequence_nll_sum.dtype
            reward_row_weight = reward_loss_out.active_mask.to(dtype = sequence_dtype) * reward_weight
            sequence_loss_sum = (
                belief_loss.sequence_nll_sum
                + reward_loss_out.per_sequence_loss.to(dtype = sequence_dtype) * reward_weight
            )
            sequence_rows = belief_loss.active_rows_per_sequence.to(dtype = sequence_dtype) + reward_row_weight
            if bool((sequence_rows <= 0).any().item()):
                raise RuntimeError("Combined loss normalization requires at least one active target per sequence.")
            total_loss = (sequence_loss_sum / sequence_rows).mean()
        reward_loss = reward_loss_out.loss
        reward_mae = reward_loss_out.mae

    return {
        "loss": total_loss,
        "belief_loss": belief_loss.loss,
        "token_accuracy": belief_loss.token_accuracy,
        "num_active_rows": belief_loss.num_active_rows,
        "reward_to_go_loss": reward_loss,
        "reward_to_go_mae": reward_mae,
    }


def _metrics_from_loss_output(loss_out: dict[str, torch.Tensor | int | None]) -> dict[str, float]:
    metrics = {
        "loss": float(loss_out["loss"].detach().cpu().item()),  # type: ignore[union-attr]
        "belief_loss": float(loss_out["belief_loss"].detach().cpu().item()),  # type: ignore[union-attr]
        "token_accuracy": float(loss_out["token_accuracy"].detach().cpu().item()),  # type: ignore[union-attr]
    }
    reward_loss = loss_out.get("reward_to_go_loss")
    reward_mae = loss_out.get("reward_to_go_mae")
    if reward_loss is not None:
        metrics["reward_to_go_loss"] = float(reward_loss.detach().cpu().item())
    if reward_mae is not None:
        metrics["reward_to_go_mae"] = float(reward_mae.detach().cpu().item())
    return metrics


def compute_batch_metrics(
    *,
    model: torch.nn.Module,
    batch: TensorDict,
    pad_token: int,
    reward_to_go_loss_weight: float = 1.0,
) -> dict[str, float]:
    batch = batch.to(next(model.parameters()).device)
    if int(batch.batch_size[0]) <= 0:
        raise RuntimeError("Training batch is empty.")

    was_training = bool(model.training)
    model.eval()
    with torch.inference_mode():
        loss_out = _compute_loss_output_from_batch(
            model = model,
            batch = batch,
            pad_token = pad_token,
            reward_to_go_loss_weight = float(reward_to_go_loss_weight),
        )
    if was_training:
        model.train()
    return _metrics_from_loss_output(loss_out)


def run_training_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | MultiOptimizer,
    batch: TensorDict,
    pad_token: int,
    grad_max_norm: float,
    reward_to_go_loss_weight: float = 1.0,
) -> dict[str, float]:
    batch = batch.to(next(model.parameters()).device)
    model.train()
    optimizer.zero_grad(set_to_none = True)

    if int(batch.batch_size[0]) <= 0:
        raise RuntimeError("Training batch is empty.")

    loss_out = _compute_loss_output_from_batch(
        model = model,
        batch = batch,
        pad_token = pad_token,
        reward_to_go_loss_weight = float(reward_to_go_loss_weight),
    )
    loss_tensor = loss_out["loss"]
    if not isinstance(loss_tensor, torch.Tensor):
        raise RuntimeError("Training loss must be a tensor.")
    loss_tensor.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_max_norm))
    optimizer.step()
    return _metrics_from_loss_output(loss_out)


__all__ = ["compute_batch_metrics", "run_training_step"]
