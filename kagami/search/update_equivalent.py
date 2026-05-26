"""Game-agnostic update-equivalent policy search orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from zlib import crc32

import numpy as np
from marvel.model.tensordict import TensorDict
import torch

from kagami.core.common import softmax_policy_probs
from kagami.core.types import ActionValueEstimator, PolicyActionProb, PolicyWeight
from kagami.search.pending_policy import PendingPolicyQueryService


@dataclass(frozen = True)
class PendingPolicyAction:
    action_str: str
    prob: float
    is_commit: bool


class UpdateEquivalentSearchEngine:
    """Object-owned update-equivalent policy search for a game/policy/value bundle."""

    def __init__(
        self,
        *,
        game: Any,
        policy_model: Any,
        action_value_estimator: ActionValueEstimator,
        policy_query_service: PendingPolicyQueryService | None = None,
        search_beta: float,
        action_temp: float,
        seed: int,
    ) -> None:
        self._game = game
        self._action_value_estimator = action_value_estimator
        self._search_beta = float(search_beta)
        self._action_temp = float(action_temp)
        self._seed = int(seed)

        if self._search_beta < 0.0:
            raise ValueError("search_beta must be non-negative.")

        self._policy_query_service = (
            PendingPolicyQueryService(game = game, policy_model = policy_model)
            if policy_query_service is None
            else policy_query_service
        )

    def compute_policy_action_prob(self, *, infoset_str: str, pending_action: str) -> PolicyActionProb:
        decoded = self._game.parse_infoset_str(str(infoset_str), "")
        root_player_id = int(np.asarray(decoded.player_ids, dtype = np.int64)[-1])
        if root_player_id < 0:
            raise RuntimeError("compute_policy_action_prob requires an active-player infoset_str.")

        normalized_pending = str(pending_action).strip()
        search_cache: dict[str, list[PendingPolicyAction]] = {}
        out = self._compute_pending_policy_action_prob(
            root_player_id = int(root_player_id),
            infoset_str = str(infoset_str),
            pending_action = normalized_pending,
            seed = self.mix_seed(self._seed, normalized_pending, "policy_action"),
            search_cache = search_cache,
        )

        tuples = [
            (str(item.action_str), bool(item.is_commit), float(item.prob))
            for item in out
            if float(item.prob) > 0.0
        ]
        tuples.sort(key = lambda item: (-float(item[2]), str(item[0]), int(bool(item[1]))))
        if not tuples:
            raise RuntimeError("Update-equivalent search produced an empty policy-action row.")

        probs = torch.as_tensor([float(item[2]) for item in tuples], dtype = torch.float64)
        is_commit = torch.as_tensor([bool(item[1]) for item in tuples], dtype = torch.bool)
        tensors = TensorDict(
            {
                "probs": probs,
                "is_commit": is_commit,
            },
            batch_size = torch.Size([len(tuples)]),
            device = probs.device,
        )
        return PolicyActionProb(
            action_strs = [str(item[0]) for item in tuples],
            tensors = tensors,
        )

    def _compute_pending_policy_action_prob(
        self,
        *,
        root_player_id: int,
        infoset_str: str,
        pending_action: str,
        seed: int,
        search_cache: dict[str, list[PendingPolicyAction]],
    ) -> list[PendingPolicyAction]:
        cache_key = str(pending_action)
        cached = search_cache.get(cache_key)
        if cached is not None:
            return cached

        pending_eval = self._policy_query_service.evaluate_pending_policy(
            infoset_str = infoset_str,
            pending_action = pending_action,
        )

        policy_probs = softmax_policy_probs(pending_eval.logits, temperature = 1.0)
        policy_prob_values = policy_probs.detach().cpu().tolist()
        pi_policy = [
            PolicyWeight(action_str = str(action_str), prob = float(prob))
            for action_str, prob in zip(pending_eval.action_strs, policy_prob_values, strict = False)
        ]

        qvalues = self._action_value_estimator.estimate_action_qvalues(
            infoset_str = infoset_str,
            root_player_id = int(root_player_id),
            pending_action = str(pending_action),
            pending_eval = pending_eval,
            seed = self.mix_seed(int(seed), cache_key, "qvalues"),
        )

        solved = self.solve_update_equivalent_policy(
            qvalues = qvalues,
            pi_policy = pi_policy,
            search_beta = float(self._search_beta),
            action_temp = float(self._action_temp),
        )
        solved_probs = {str(item.action_str): float(item.prob) for item in solved}
        out = [
            PendingPolicyAction(
                action_str = str(action_str),
                prob = float(solved_probs[str(action_str)]),
                is_commit = bool(action_is_commit),
            )
            for action_str, action_is_commit in zip(
                pending_eval.action_strs,
                pending_eval.is_commit,
                strict = False,
            )
        ]
        out.sort(key = lambda item: (-float(item.prob), str(item.action_str), int(bool(item.is_commit))))
        search_cache[cache_key] = out
        return out

    @staticmethod
    def solve_update_equivalent_policy(
        *,
        qvalues: Mapping[str, float],
        pi_policy: Sequence[PolicyWeight],
        search_beta: float,
        action_temp: float,
    ) -> list[PolicyWeight]:
        if float(search_beta) < 0.0:
            raise ValueError("search_beta must be non-negative.")

        policy_probs = {str(item.action_str): float(item.prob) for item in pi_policy}
        actions = [str(item.action_str) for item in pi_policy]
        q = torch.as_tensor([float(qvalues.get(action, 0.0)) for action in actions], dtype = torch.float64)
        policy = torch.as_tensor([policy_probs[action] for action in actions], dtype = torch.float64)
        policy = policy / torch.sum(policy)
        logits = float(search_beta) * q + torch.log(torch.clamp(policy, min = 1e-30, max = 1.0))
        probs = softmax_policy_probs(logits, temperature = float(action_temp))
        out = [
            PolicyWeight(action_str = action, prob = float(prob))
            for action, prob in zip(actions, probs.detach().cpu().tolist(), strict = False)
        ]
        out.sort(key = lambda item: (-float(item.prob), str(item.action_str)))
        return out

    @staticmethod
    def mix_seed(seed: int, *parts: str) -> int:
        mixed = int(seed) & 0xFFFFFFFF
        for part in parts:
            mixed ^= int(crc32(str(part).encode("utf-8")) & 0xFFFFFFFF)
        return int(mixed & 0x7FFFFFFF)


__all__ = [
    "PendingPolicyAction",
    "UpdateEquivalentSearchEngine",
]
