from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen = True)
class BeliefCandidateLoss:
    loss: torch.Tensor
    sequence_nll_sum: torch.Tensor
    active_rows_per_sequence: torch.Tensor
    num_active_rows: int
    token_accuracy: torch.Tensor


@dataclass(frozen = True)
class RewardToGoLoss:
    loss: torch.Tensor
    per_sequence_loss: torch.Tensor
    active_mask: torch.Tensor
    mae: torch.Tensor
    num_active: int


def masked_belief_candidate_nll(
    *,
    logits: torch.Tensor,
    target_index: torch.Tensor,
    target_mask: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> BeliefCandidateLoss:
    if logits.ndim != 3:
        raise RuntimeError(f"logits must be [B, S, A], got {tuple(logits.shape)}")
    if target_index.shape != logits.shape[:2]:
        raise RuntimeError(
            "target_index shape mismatch: "
            f"{tuple(target_index.shape)} vs {tuple(logits.shape[:2])}"
        )
    if target_mask.shape != logits.shape[:2]:
        raise RuntimeError(
            "target_mask shape mismatch: "
            f"{tuple(target_mask.shape)} vs {tuple(logits.shape[:2])}"
        )
    if candidate_valid.shape != logits.shape:
        raise RuntimeError(
            "candidate_valid shape mismatch: "
            f"{tuple(candidate_valid.shape)} vs {tuple(logits.shape)}"
        )

    active_rows = target_mask.to(dtype = torch.bool)
    num_active_rows = int(active_rows.sum().item())
    active_rows_per_sequence = active_rows.to(dtype = torch.int64).sum(dim = 1)
    if bool((active_rows_per_sequence <= 0).any().item()):
        raise RuntimeError(
            "Per-sequence normalization requires at least one active belief row per sequence."
        )

    active_has_candidates = candidate_valid.to(dtype = torch.bool).any(dim = -1)
    if bool((active_rows & ~active_has_candidates).any().item()):
        raise RuntimeError("Active belief rows must have at least one valid candidate.")

    target_index_long = target_index.to(dtype = torch.long)
    num_candidates = int(logits.shape[-1])
    if bool((active_rows & (target_index_long < 0)).any().item()):
        raise RuntimeError("Active belief rows must have non-negative target_index.")
    if bool((active_rows & (target_index_long >= num_candidates)).any().item()):
        raise RuntimeError(
            f"target_index out of bounds for candidate axis: num_candidates={num_candidates}."
        )

    safe_target_index = target_index_long.clamp(min = 0)
    active_target_valid = candidate_valid.to(dtype = torch.bool).gather(
        dim = -1,
        index = safe_target_index.unsqueeze(-1),
    ).squeeze(-1)
    if bool((active_rows & ~active_target_valid).any().item()):
        raise RuntimeError("target_index must point to a valid candidate for every active belief row.")

    flat_logits = logits.reshape(-1, num_candidates)
    flat_targets = safe_target_index.reshape(-1)
    row_nll = F.cross_entropy(flat_logits, flat_targets, reduction = "none").reshape_as(target_index_long)
    active_rows_float = active_rows.to(dtype = row_nll.dtype)
    sequence_nll_sum = (row_nll * active_rows_float).sum(dim = 1)
    sequence_nll = sequence_nll_sum / active_rows_float.sum(dim = 1)
    loss = sequence_nll.mean()

    with torch.no_grad():
        pred_rows = torch.argmax(logits, dim = -1)
        row_acc = (pred_rows == safe_target_index).to(dtype = logits.dtype)
        sequence_acc = (row_acc * active_rows_float).sum(dim = 1) / active_rows_float.sum(dim = 1)
        acc = sequence_acc.mean()

    return BeliefCandidateLoss(
        loss = loss,
        sequence_nll_sum = sequence_nll_sum,
        active_rows_per_sequence = active_rows_per_sequence,
        num_active_rows = num_active_rows,
        token_accuracy = acc,
    )


def masked_reward_to_go_huber(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 10.0,
) -> RewardToGoLoss:
    if pred.ndim != 1:
        raise RuntimeError(f"reward_to_go pred must be [B], got {tuple(pred.shape)}")
    if target.shape != pred.shape:
        raise RuntimeError(
            "reward_to_go target shape mismatch: "
            f"{tuple(target.shape)} vs {tuple(pred.shape)}"
        )
    if mask.shape != pred.shape:
        raise RuntimeError(
            "reward_to_go mask shape mismatch: "
            f"{tuple(mask.shape)} vs {tuple(pred.shape)}"
        )

    active = mask.to(dtype = torch.bool)
    num_active = int(active.sum().item())
    if num_active <= 0:
        raise RuntimeError("reward_to_go loss requires at least one active target.")
    if float(delta) <= 0.0:
        raise RuntimeError(f"reward_to_go Huber delta must be positive, got {float(delta)}.")

    diff = pred.to(dtype = torch.float32) - target.to(dtype = torch.float32)
    per_sequence_loss = F.huber_loss(
        pred.to(dtype = torch.float32),
        target.to(dtype = torch.float32),
        reduction = "none",
        delta = float(delta),
    )
    active_diff = diff[active]
    return RewardToGoLoss(
        loss = torch.mean(per_sequence_loss[active]),
        per_sequence_loss = per_sequence_loss * active.to(dtype = per_sequence_loss.dtype),
        active_mask = active,
        mae = torch.mean(active_diff.abs()),
        num_active = num_active,
    )


__all__ = [
    "BeliefCandidateLoss",
    "RewardToGoLoss",
    "masked_belief_candidate_nll",
    "masked_reward_to_go_huber",
]
