"""Game-agnostic policy-action value estimation for search rollouts."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from kagami.core.types import PolicyModel
from kagami.search.horizon import resolve_max_game_horizon
from kagami.search.rollout_session import SearchRolloutSession


class PolicyActionValueEstimator:
    """Batched committed-action value estimation via determinized rollouts."""

    def __init__(self, *, game: Any, policy_model: PolicyModel, rollout_temp: float) -> None:
        self._game = game
        self._policy_model = policy_model
        self._rollout_temp = float(rollout_temp)
        self._max_horizon = int(resolve_max_game_horizon(self._game))
        if self._max_horizon <= 0:
            raise RuntimeError(f"max_horizon must be positive, got {self._max_horizon}.")
        self._rollout_sessions: dict[int, SearchRolloutSession] = {}

    @staticmethod
    def allocate_rollout_counts(*, rollout_budget: int, action_count: int) -> list[int]:
        budget = int(rollout_budget)
        count = int(action_count)
        if budget <= 0:
            raise ValueError("rollout_budget must be positive.")
        if count <= 0:
            raise RuntimeError("Update-equivalent search encountered an empty action set.")
        per_action = int(math.ceil(float(budget) / float(count)))
        return [per_action] * count

    def estimate_policy_action_values_batch(
        self,
        *,
        root_player_id: int,
        pending_action: str,
        action_strs: Sequence[str],
        action_token_rows: np.ndarray,
        rollout_counts: Sequence[int],
        belief_samples: Sequence[Any],
        seed: int,
    ) -> dict[str, float]:
        actions = [str(action).strip() for action in action_strs]
        if any(not action for action in actions):
            raise RuntimeError("Action strings must be non-empty when estimating batched policy action values.")
        token_rows = np.asarray(action_token_rows, dtype = np.int64)
        if int(token_rows.ndim) != 2:
            raise RuntimeError(
                "action_token_rows must have shape [num_actions, max_action_tokens] for batched value estimation: "
                f"{tuple(token_rows.shape)}."
            )
        counts = [int(count) for count in rollout_counts]
        if len(actions) != int(token_rows.shape[0]) or len(actions) != len(counts):
            raise RuntimeError(
                "Batched policy action value inputs are misaligned: "
                f"len(actions)={len(actions)} token_rows={int(token_rows.shape[0])} len(rollout_counts)={len(counts)}."
            )

        history_pool = [str(item.history_str) for item in belief_samples]
        if not history_pool:
            raise RuntimeError("Update-equivalent value estimation requires at least one belief sample.")

        expanded_histories: list[str] = []
        expanded_tokens: list[np.ndarray] = []
        action_indices: list[int] = []
        for action_idx, rollout_count in enumerate(counts):
            if int(rollout_count) <= 0:
                raise RuntimeError(
                    "Batched policy action value estimation requires positive rollout counts per action: "
                    f"action_idx={action_idx} rollout_count={rollout_count}."
                )
            if int(rollout_count) > len(history_pool):
                raise RuntimeError(
                    "Batched policy action value estimation requested more rollout samples than available belief samples: "
                    f"action_idx={action_idx} rollout_count={rollout_count} available={len(history_pool)}."
                )
            selected_histories = history_pool[: int(rollout_count)]
            expanded_histories.extend(selected_histories)
            expanded_tokens.extend([token_rows[action_idx]] * int(rollout_count))
            action_indices.extend([int(action_idx)] * int(rollout_count))

        session = self._lookup_rollout_session(root_player_id = int(root_player_id))
        rewards = session.rollout_selected_action_rows(
            root_history_strs = expanded_histories,
            pending_action = str(pending_action),
            action_token_rows = np.asarray(expanded_tokens, dtype = np.int64),
            seed = int(seed),
        )
        if int(rewards.shape[0]) != len(action_indices):
            raise RuntimeError(
                "Batched rollout reward shape mismatch during policy action value estimation: "
                f"len(rewards)={int(rewards.shape[0])} expanded_rows={len(action_indices)}."
            )

        reward_sums = np.zeros(len(actions), dtype = np.float64)
        reward_counts = np.zeros(len(actions), dtype = np.int64)
        for action_idx, reward in zip(action_indices, rewards.tolist(), strict = False):
            reward_sums[int(action_idx)] += float(reward)
            reward_counts[int(action_idx)] += 1

        qvalues: dict[str, float] = {}
        for action_idx, action_str in enumerate(actions):
            count = int(reward_counts[action_idx])
            if count <= 0:
                raise RuntimeError(
                    "Batched rollout produced no rewards for action during policy action value estimation: "
                    f"action_idx={action_idx} action_str={action_str!r}."
                )
            qvalues[str(action_str)] = float(reward_sums[action_idx] / float(count))
        return qvalues

    def _lookup_rollout_session(self, *, root_player_id: int) -> SearchRolloutSession:
        resolved_root_player_id = int(root_player_id)
        session = self._rollout_sessions.get(resolved_root_player_id)
        if session is not None:
            return session
        session = SearchRolloutSession(
            game = self._game,
            policy_model = self._policy_model,
            root_player_id = resolved_root_player_id,
            rollout_temp = float(self._rollout_temp),
            max_horizon = int(self._max_horizon),
        )
        self._rollout_sessions[resolved_root_player_id] = session
        return session


__all__ = ["PolicyActionValueEstimator"]
