"""Shared helpers for Kagami inference validation and policy evaluation."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from kagami.core.types import PolicyEval


def validate_temperature(value: float, *, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


def validate_infoset_strs(infoset_strs: Sequence[str]) -> list[str]:
    state_strings = [str(s).strip() for s in infoset_strs]
    if any(not s for s in state_strings):
        raise ValueError("infoset_strs must not contain empty items.")
    return state_strings


def validate_policy_eval_row(eval_row: PolicyEval) -> tuple[list[str], list[bool], torch.Tensor]:
    action_strs = [str(action).strip() for action in eval_row.action_strs]
    if not action_strs:
        raise RuntimeError("Policy evaluator returned no actions.")
    if any(not action for action in action_strs):
        raise RuntimeError("Policy evaluator returned an empty semantic action string.")
    is_commit = [bool(flag) for flag in eval_row.is_commit]
    logits_raw = eval_row.logits
    if not isinstance(logits_raw, torch.Tensor):
        raise TypeError(f"policy eval logits must be torch.Tensor, got {type(logits_raw).__name__}.")
    logits = logits_raw.detach().to(dtype = torch.float64)
    if int(logits.ndim) != 1:
        raise RuntimeError(f"Expected 1D logits, got shape {tuple(logits.shape)}.")
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("Logits contain non-finite values.")
    if len(action_strs) != len(is_commit) or len(action_strs) != int(logits.shape[0]):
        raise RuntimeError(
            "Policy evaluator output alignment mismatch: "
            f"len(action_strs)={len(action_strs)}, len(is_commit)={len(is_commit)}, "
            f"len(logits)={int(logits.shape[0])}."
        )
    return action_strs, is_commit, logits


def softmax_policy_probs(logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    validate_temperature(float(temperature), name = "temperature")
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"softmax_policy_probs expects torch.Tensor logits, got {type(logits).__name__}.")
    logits_t = logits.detach().to(dtype = torch.float64)
    if int(logits_t.ndim) != 1:
        raise RuntimeError(f"softmax_policy_probs expects 1D logits, got shape {tuple(logits_t.shape)}.")
    if int(logits_t.numel()) == 0:
        raise RuntimeError("softmax_policy_probs received an empty logits vector.")
    if not bool(torch.isfinite(logits_t).all().item()):
        raise RuntimeError("softmax_policy_probs received non-finite logits.")

    if float(temperature) <= 0.0:
        out = torch.zeros_like(logits_t, dtype = torch.float64)
        out[int(torch.argmax(logits_t).item())] = 1.0
        return out

    scaled = logits_t / float(temperature)
    shifted = scaled - torch.max(scaled)
    exp = torch.exp(shifted)
    denom = float(torch.sum(exp).item())
    if not math.isfinite(denom) or denom <= 0.0:
        raise RuntimeError("Softmax normalization failed with non-positive denominator.")
    return exp / denom


def select_action_from_avail_probs(
    *,
    avail_probs: torch.Tensor,
    avail_weights: torch.Tensor,
    temperature: float,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Prior-preserving action selection over a batch of policy outputs.

    Given model-produced ``avail_probs = softmax(learned_logits + log(avail_weights))``
    and the raw per-action prior ``avail_weights`` (same shape), returns per-env
    action indices. This matches the convention already used by
    :class:`marvel.runtime.sampler.Sampler.rollout_step`:

      raw_logits = log(avail_probs) - log(avail_weights)   # recovers learned_logits
      # temperature <= 0.0: argmax(raw_logits), masking illegal (weight==0) -> -inf
      # temperature > 0.0:  softmax(raw_logits/T + log(avail_weights)) then sample

    Properties of this scheme:
      * At T=1 the output distribution equals the raw policy ``avail_probs``.
      * As T→∞ the sampling distribution approaches the prior (``avail_weights``
        normalized), not flat-uniform. This is usually the intent for chance-like
        priors.
      * At T=0 selection reduces to argmax over the learned component only; the
        prior no longer biases the selection. For uniform-prior player nodes this
        is equivalent to ``argmax(log_probs)`` used elsewhere.

    Args:
        avail_probs: tensor ``[..., A]`` with probability rows.
        avail_weights: tensor ``[..., A]`` of non-negative per-action weights; entries
            with weight ``0`` are treated as illegal and masked out.
        temperature: non-negative scalar. At 0 (or below), returns argmax; else
            samples via ``torch.multinomial`` on the scaled distribution.
        rng: optional ``torch.Generator`` for deterministic sampling.

    Returns:
        Long tensor of action indices with shape ``avail_probs.shape[:-1]``.
    """
    validate_temperature(float(temperature), name = "temperature")
    if not isinstance(avail_probs, torch.Tensor) or not isinstance(avail_weights, torch.Tensor):
        raise TypeError("select_action_from_avail_probs requires torch.Tensor inputs.")
    if avail_probs.shape != avail_weights.shape:
        raise RuntimeError(
            "avail_probs / avail_weights shape mismatch: "
            f"{tuple(avail_probs.shape)} vs {tuple(avail_weights.shape)}."
        )
    probs = avail_probs.float()
    weights = avail_weights.float()
    mask = weights > 0
    log_probs = probs.clamp(min = 1e-9).log()
    log_prior = weights.clamp(min = 1e-9).log()
    raw_logits = log_probs - log_prior
    neg_inf = torch.finfo(raw_logits.dtype).min
    if float(temperature) <= 0.0:
        return torch.argmax(raw_logits.masked_fill(~mask, neg_inf), dim = -1)
    scaled = (raw_logits / float(temperature) + log_prior).masked_fill(~mask, neg_inf)
    probs_for_sampling = torch.softmax(scaled, dim = -1)
    flat = probs_for_sampling.reshape(-1, probs_for_sampling.shape[-1])
    sampled = torch.multinomial(flat, 1, generator = rng).squeeze(-1)
    return sampled.reshape(probs_for_sampling.shape[:-1])


__all__ = [
    "select_action_from_avail_probs",
    "softmax_policy_probs",
    "validate_infoset_strs",
    "validate_policy_eval_row",
    "validate_temperature",
]
