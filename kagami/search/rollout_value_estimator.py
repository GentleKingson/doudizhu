"""Rollout-backed action-value estimation for update-equivalent search."""

from __future__ import annotations

from typing import Any
from zlib import crc32

import numpy as np

from kagami.search.pending_policy import PendingPolicyEval
from kagami.search.value_estimator import PolicyActionValueEstimator


class RolloutActionValueEstimator:
    """Yomi belief sampling plus rollout Q estimation for legal actions."""

    def __init__(
        self,
        *,
        game: Any,
        policy_model: Any,
        belief_model: Any,
        belief_temp: float,
        rollout_temp: float,
        rollout_budget: int,
    ) -> None:
        self._belief_model = belief_model
        self._belief_temp = float(belief_temp)
        self._rollout_budget = int(rollout_budget)
        if self._rollout_budget <= 0:
            raise ValueError("rollout_budget must be positive.")
        self._value_estimator = PolicyActionValueEstimator(
            game = game,
            policy_model = policy_model,
            rollout_temp = float(rollout_temp),
        )

    def estimate_action_qvalues(
        self,
        *,
        infoset_str: str,
        root_player_id: int,
        pending_action: str,
        pending_eval: PendingPolicyEval,
        seed: int,
    ) -> dict[str, float]:
        rollout_counts = self._value_estimator.allocate_rollout_counts(
            rollout_budget = int(self._rollout_budget),
            action_count = len(pending_eval.action_strs),
        )
        belief_samples = self._belief_model.sample_determinized_histories(
            infoset_strs = [str(infoset_str)],
            num_samples = int(max(rollout_counts)),
            temperature = float(self._belief_temp),
            seed = self._mix_seed(int(seed), "belief"),
        )[0]
        return self._value_estimator.estimate_policy_action_values_batch(
            root_player_id = int(root_player_id),
            pending_action = str(pending_action),
            action_strs = pending_eval.action_strs,
            action_token_rows = np.asarray(pending_eval.action_token_rows, dtype = np.int64),
            rollout_counts = rollout_counts,
            belief_samples = belief_samples,
            seed = self._mix_seed(int(seed), "rollout"),
        )

    @staticmethod
    def _mix_seed(seed: int, *parts: str) -> int:
        mixed = int(seed) & 0xFFFFFFFF
        for part in parts:
            mixed ^= int(crc32(str(part).encode("utf-8")) & 0xFFFFFFFF)
        return int(mixed & 0x7FFFFFFF)


__all__ = ["RolloutActionValueEstimator"]
