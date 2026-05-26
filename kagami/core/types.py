"""Core shared typing contracts for Kagami inference engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from marvel.model.tensordict import TensorDict
import torch


@dataclass(frozen = True)
class PolicyWeight:
    action_str: str
    prob: float


@dataclass(frozen = True)
class PolicyActionProb:
    action_strs: list[str]
    tensors: TensorDict

    @property
    def probs(self) -> torch.Tensor:
        return self.tensors["probs"]

    @property
    def is_commit(self) -> torch.Tensor:
        return self.tensors["is_commit"]


@dataclass(frozen = True)
class PolicyActionProbBatch:
    action_strs_batch: list[list[str]]
    tensors: TensorDict

    @property
    def probs(self) -> torch.Tensor:
        return self.tensors["probs"]

    @property
    def is_commit(self) -> torch.Tensor:
        return self.tensors["is_commit"]

    @property
    def valid_mask(self) -> torch.Tensor:
        return self.tensors["valid_mask"]


class PolicyEval(Protocol):
    action_strs: Sequence[str]
    is_commit: Sequence[bool]
    logits: torch.Tensor


class PolicyModel(Protocol):
    def evaluate(self, *, infoset_str: str, pending_action: str) -> PolicyEval:
        ...

    def evaluate_batch(self, *, infoset_strs: Sequence[str], pending_actions: Sequence[str]) -> list[PolicyEval]:
        ...

    def policy(
        self,
        *,
        infoset_str: str,
        pending_action: str,
        action_temp: float,
    ) -> "PolicyActionProb":
        ...

    def policy_batch(
        self,
        *,
        infoset_strs: Sequence[str],
        pending_actions: Sequence[str],
        action_temps: Sequence[float],
    ) -> "PolicyActionProbBatch":
        ...


class ActionValueEstimator(Protocol):
    def estimate_action_qvalues(
        self,
        *,
        infoset_str: str,
        root_player_id: int,
        pending_action: str,
        pending_eval: "PendingPolicyEval",
        seed: int,
    ) -> dict[str, float]:
        ...


def policy_action_prob_batch_from_rows(rows: Sequence[PolicyActionProb]) -> PolicyActionProbBatch:
    row_count = len(rows)
    if row_count == 0:
        tensors = TensorDict(
            {
                "probs": torch.zeros((0, 0), dtype = torch.float64),
                "is_commit": torch.zeros((0, 0), dtype = torch.bool),
                "valid_mask": torch.zeros((0, 0), dtype = torch.bool),
            },
            batch_size = torch.Size([0]),
            device = torch.device("cpu"),
        )
        return PolicyActionProbBatch(action_strs_batch = [], tensors = tensors)

    action_strs_batch = [list(row.action_strs) for row in rows]
    max_actions = max(len(action_strs) for action_strs in action_strs_batch)
    if max_actions <= 0:
        raise RuntimeError("Cannot build PolicyActionProbBatch with empty action rows.")

    probs = torch.zeros((row_count, max_actions), dtype = torch.float64)
    is_commit = torch.zeros((row_count, max_actions), dtype = torch.bool)
    valid_mask = torch.zeros((row_count, max_actions), dtype = torch.bool)
    for row_idx, row in enumerate(rows):
        row_probs = row.probs.detach().to(dtype = torch.float64, device = probs.device)
        row_commit = row.is_commit.detach().to(dtype = torch.bool, device = probs.device)
        if int(row_probs.ndim) != 1:
            raise RuntimeError(f"PolicyActionProb row probs must be 1D, got shape {tuple(row_probs.shape)}.")
        if int(row_commit.ndim) != 1:
            raise RuntimeError(f"PolicyActionProb row is_commit must be 1D, got shape {tuple(row_commit.shape)}.")
        action_count = len(row.action_strs)
        if action_count <= 0:
            raise RuntimeError("PolicyActionProb row action_strs must be non-empty.")
        if int(row_probs.shape[0]) != action_count:
            raise RuntimeError(
                "PolicyActionProb row action_strs/probs mismatch: "
                f"len(action_strs)={action_count}, len(probs)={int(row_probs.shape[0])}."
            )
        if int(row_commit.shape[0]) != action_count:
            raise RuntimeError(
                "PolicyActionProb row action_strs/is_commit mismatch: "
                f"len(action_strs)={action_count}, len(is_commit)={int(row_commit.shape[0])}."
            )
        probs[row_idx, :action_count] = row_probs
        is_commit[row_idx, :action_count] = row_commit
        valid_mask[row_idx, :action_count] = True

    tensors = TensorDict(
        {
            "probs": probs,
            "is_commit": is_commit,
            "valid_mask": valid_mask,
        },
        batch_size = torch.Size([row_count]),
        device = probs.device,
    )
    return PolicyActionProbBatch(action_strs_batch = action_strs_batch, tensors = tensors)


def policy_action_prob_row_from_batch(batch: PolicyActionProbBatch, *, row_idx: int) -> PolicyActionProb:
    total_rows = len(batch.action_strs_batch)
    if row_idx < 0 or row_idx >= total_rows:
        raise IndexError(f"row_idx out of range: {row_idx} (rows={total_rows}).")
    action_strs = list(batch.action_strs_batch[int(row_idx)])
    action_count = len(action_strs)
    if action_count <= 0:
        raise RuntimeError(f"PolicyActionProbBatch row {row_idx} has no actions.")
    row_probs = batch.probs[int(row_idx), :action_count].detach().clone()
    row_commit = batch.is_commit[int(row_idx), :action_count].detach().clone()
    tensors = TensorDict(
        {
            "probs": row_probs,
            "is_commit": row_commit,
        },
        batch_size = torch.Size([action_count]),
        device = row_probs.device,
    )
    return PolicyActionProb(action_strs = action_strs, tensors = tensors)


__all__ = [
    "ActionValueEstimator",
    "PolicyActionProb",
    "PolicyActionProbBatch",
    "PolicyEval",
    "PolicyModel",
    "PolicyWeight",
    "policy_action_prob_batch_from_rows",
    "policy_action_prob_row_from_batch",
]
