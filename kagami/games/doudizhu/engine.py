"""Dou Dizhu inference engine for Kagami one-step distributions."""

from __future__ import annotations

from dataclasses import dataclass, field
from zlib import crc32
from typing import Sequence

import nanopokerlib as npl
import torch
from kagami.core.common import validate_infoset_strs, validate_temperature
from kagami.core.interfaces import InferenceEngine
from kagami.core.types import (
    PolicyActionProb,
    PolicyActionProbBatch,
    PolicyModel,
    policy_action_prob_batch_from_rows,
)
from kagami.search.local_subgame import LocalSubgameSearchEngine
from kagami.search.pending_policy import PendingPolicyQueryService
from kagami.search.routines import (
    DEFAULT_LOCAL_SEARCH_STABILITY_MARGIN,
    SEARCH_ROUTINE_LOCAL_SUBGAME,
    SEARCH_ROUTINE_UPDATE_EQUIVALENT,
    normalize_search_routine,
    validate_search_floor,
)
from kagami.search.rollout_value_estimator import RolloutActionValueEstimator
from kagami.search.update_equivalent import UpdateEquivalentSearchEngine


@dataclass(frozen = True)
class DoudizhuInferenceEngineConfig:
    @dataclass(frozen = True)
    class UpdateEquivalent:
        enabled: bool = False
        routine: str = SEARCH_ROUTINE_LOCAL_SUBGAME
        rollout_budget: int = 512
        search_floor: int | None = None
        # Inverse KL scale against the base policy prior: update-equiv uses Q,
        # local-subgame uses PCFR counterfactual advantages.
        search_beta: float = 1.0
        seed: int = 0
        stability_margin: float = DEFAULT_LOCAL_SEARCH_STABILITY_MARGIN

    # Belief-sampling temperature for search refinement.
    belief_temp: float = 1.0
    # Search-side temperature: rollout sampling for update-equiv, solution/stability temperature for local-subgame.
    search_temp: float = 1.0
    action_temp: float = 0.0
    sample_seed: int = 0
    update_equivalent: "DoudizhuInferenceEngineConfig.UpdateEquivalent" = field(
        default_factory = lambda: DoudizhuInferenceEngineConfig.UpdateEquivalent()
    )


class DoudizhuInferenceEngine(InferenceEngine):
    """One-step action-distribution inference for Dou Dizhu infoset strings."""

    def __init__(
        self,
        *,
        policy_model: PolicyModel | None = None,
        cfg: DoudizhuInferenceEngineConfig | None = None,
        belief_model: object | None = None,
        game: object | None = None,
    ) -> None:
        resolved_game = game
        if resolved_game is None:
            resolved_game = getattr(npl, "doudizhu_tiny", None)
        if resolved_game is None:
            resolved_game = getattr(npl, "doudizhu", None)
        if resolved_game is None:
            raise RuntimeError(
                "nanopokerlib Dou Dizhu game module is not available. "
                "Build/install nanopokerlib with Dou Dizhu game modules."
            )
        self._policy_model = policy_model
        self._belief_model = belief_model
        self._critic_model: object | None = None
        self._critic_device: torch.device | None = None
        self._cached_search_policy_model: object | None = None
        self._cached_search_belief_model: object | None = None
        self._cached_search_critic_model: object | None = None
        self._cached_search_routine: str | None = None
        self._cached_rollout_value_estimator: RolloutActionValueEstimator | None = None
        self._cached_policy_query_service: PendingPolicyQueryService | None = None
        self.cfg = cfg or DoudizhuInferenceEngineConfig()
        self._validate_config(self.cfg)
        self._allow_implicit_game_rebind = game is None
        super().__init__(
            game = resolved_game,
            action_temp = float(self.cfg.action_temp),
            sample_seed = int(self.cfg.sample_seed),
        )

    def infer_batch(
        self,
        infoset_strs: Sequence[str],
        pending_actions: Sequence[str] | None = None,
    ) -> PolicyActionProbBatch:
        policy_model = self._policy_model
        if policy_model is None:
            raise RuntimeError(
                "Dou Dizhu MARVEL model is not configured. "
                "Use DoudizhuInferenceEngine.bind_marvel_model(...) or pass policy_model explicitly."
            )
        state_strings = validate_infoset_strs(infoset_strs)
        if pending_actions is None:
            pending_list = [""] * len(state_strings)
        else:
            pending_list = [str(pending_action).strip() for pending_action in pending_actions]
            if len(pending_list) != len(state_strings):
                raise ValueError("pending_actions must match the number of infoset_strs.")
        if not state_strings:
            return policy_action_prob_batch_from_rows([])

        if self.cfg.update_equivalent.enabled:
            routine = self._normalized_search_routine()
            if routine == SEARCH_ROUTINE_UPDATE_EQUIVALENT:
                return policy_action_prob_batch_from_rows(
                    [
                        self._policy_action_prob_update_equivalent(
                            infoset_str = infoset,
                            pending_action = pending,
                        )
                        for infoset, pending in zip(state_strings, pending_list, strict = False)
                    ]
                )
            if routine == SEARCH_ROUTINE_LOCAL_SUBGAME:
                return policy_action_prob_batch_from_rows(
                    [
                        self._policy_action_prob_local_subgame(
                            infoset_str = infoset,
                            pending_action = pending,
                        )
                        for infoset, pending in zip(state_strings, pending_list, strict = False)
                    ]
                )
            raise RuntimeError(f"Unsupported Kagami search routine: {routine!r}.")

        return policy_model.policy_batch(
            infoset_strs = state_strings,
            pending_actions = pending_list,
            action_temps = [float(self._action_temp)] * len(state_strings),
        )

    def _policy_action_prob_update_equivalent(
        self,
        *,
        infoset_str: str,
        pending_action: str,
    ) -> PolicyActionProb:
        policy_model = self._policy_model
        if policy_model is None:
            raise RuntimeError(
                "Dou Dizhu MARVEL model is not configured. "
                "Use DoudizhuInferenceEngine.bind_marvel_model(...) or pass policy_model explicitly."
            )
        cfg = self.cfg.update_equivalent
        if self._belief_model is None:
            raise ValueError("Dou Dizhu update-equivalent mode requires a Yomi belief connector.")
        if self._game is None:
            raise ValueError("Dou Dizhu update-equivalent mode requires a bound game module.")
        rollout_value_estimator, policy_query_service = self._lookup_update_equivalent_services()
        seed = int(cfg.seed) ^ int(crc32(str(infoset_str).encode("utf-8")) & 0xFFFFFFFF)
        search_engine = UpdateEquivalentSearchEngine(
            game = self._game,
            policy_model = policy_model,
            action_value_estimator = rollout_value_estimator,
            policy_query_service = policy_query_service,
            search_beta = float(cfg.search_beta),
            seed = seed,
            action_temp = float(self._action_temp),
        )
        return search_engine.compute_policy_action_prob(
            infoset_str = infoset_str,
            pending_action = str(pending_action).strip(),
        )

    def _policy_action_prob_local_subgame(
        self,
        *,
        infoset_str: str,
        pending_action: str,
    ) -> PolicyActionProb:
        policy_model = self._policy_model
        if policy_model is None:
            raise RuntimeError(
                "Dou Dizhu MARVEL model is not configured. "
                "Use DoudizhuInferenceEngine.bind_marvel_model(...) or pass policy_model explicitly."
            )
        cfg = self.cfg.update_equivalent
        if self._belief_model is None:
            raise ValueError("Dou Dizhu local-subgame routine requires a Yomi belief connector.")
        if self._critic_model is None or self._critic_device is None:
            raise ValueError("Dou Dizhu local-subgame routine requires a qvalue CentralCritic.")
        if self._game is None:
            raise ValueError("Dou Dizhu local-subgame routine requires a bound game module.")

        policy_query_service = self._cached_policy_query_service
        if (
            policy_query_service is None
            or self._cached_search_policy_model is not policy_model
            or self._cached_search_belief_model is not self._belief_model
            or self._cached_search_critic_model is not self._critic_model
            or self._cached_search_routine != SEARCH_ROUTINE_LOCAL_SUBGAME
        ):
            policy_query_service = PendingPolicyQueryService(game = self._game, policy_model = policy_model)
            self._cached_policy_query_service = policy_query_service
            self._cached_search_policy_model = policy_model
            self._cached_search_belief_model = self._belief_model
            self._cached_search_critic_model = self._critic_model
            self._cached_search_routine = SEARCH_ROUTINE_LOCAL_SUBGAME

        seed = int(cfg.seed) ^ int(crc32(str(infoset_str).encode("utf-8")) & 0xFFFFFFFF)
        search_engine = LocalSubgameSearchEngine(
            game = self._game,
            policy_model = policy_model,
            belief_model = self._belief_model,
            critic_model = self._critic_model,
            critic_device = self._critic_device,
            policy_query_service = policy_query_service,
            belief_temp = float(self.cfg.belief_temp),
            search_beta = float(cfg.search_beta),
            search_budget = int(cfg.rollout_budget),
            search_floor = cfg.search_floor,
            stability_margin = float(cfg.stability_margin),
            search_temp = float(self.cfg.search_temp),
            action_temp = float(self._action_temp),
            seed = seed,
        )
        return search_engine.compute_policy_action_prob(
            infoset_str = infoset_str,
            pending_action = str(pending_action).strip(),
        )

    def _lookup_update_equivalent_services(
        self,
    ) -> tuple[RolloutActionValueEstimator, PendingPolicyQueryService]:
        policy_model = self._policy_model
        belief_model = self._belief_model
        if policy_model is None:
            raise RuntimeError(
                "Dou Dizhu MARVEL model is not configured. "
                "Use DoudizhuInferenceEngine.bind_marvel_model(...) or pass policy_model explicitly."
            )
        if belief_model is None:
            raise ValueError("Dou Dizhu update-equivalent mode requires a Yomi belief connector.")
        rollout_value_estimator = self._cached_rollout_value_estimator
        policy_query_service = self._cached_policy_query_service
        if (
            rollout_value_estimator is not None
            and policy_query_service is not None
            and self._cached_search_policy_model is policy_model
            and self._cached_search_belief_model is belief_model
            and self._cached_search_routine == SEARCH_ROUTINE_UPDATE_EQUIVALENT
        ):
            return rollout_value_estimator, policy_query_service

        rollout_value_estimator = RolloutActionValueEstimator(
            game = self._game,
            policy_model = policy_model,
            belief_model = belief_model,
            belief_temp = float(self.cfg.belief_temp),
            rollout_temp = float(self.cfg.search_temp),
            rollout_budget = int(self.cfg.update_equivalent.rollout_budget),
        )
        policy_query_service = PendingPolicyQueryService(game = self._game, policy_model = policy_model)
        self._cached_search_policy_model = policy_model
        self._cached_search_belief_model = belief_model
        self._cached_search_critic_model = None
        self._cached_search_routine = SEARCH_ROUTINE_UPDATE_EQUIVALENT
        self._cached_rollout_value_estimator = rollout_value_estimator
        self._cached_policy_query_service = policy_query_service
        return rollout_value_estimator, policy_query_service

    @staticmethod
    def _validate_config(cfg: DoudizhuInferenceEngineConfig) -> None:
        validate_temperature(float(cfg.belief_temp), name = "belief_temp")
        validate_temperature(float(cfg.search_temp), name = "search_temp")
        validate_temperature(float(cfg.action_temp), name = "action_temp")

        ue = cfg.update_equivalent
        if int(ue.rollout_budget) <= 0:
            raise ValueError("update_equivalent.rollout_budget must be positive.")
        validate_temperature(float(ue.search_beta), name = "update_equivalent.search_beta")
        routine = normalize_search_routine(ue.routine)
        if routine == SEARCH_ROUTINE_UPDATE_EQUIVALENT and ue.search_floor is not None:
            raise ValueError("update_equivalent.search_floor is only valid for routine='local-subgame'.")
        validate_search_floor(ue.search_floor, search_budget = int(ue.rollout_budget), name = "update_equivalent.search_floor")
        if float(ue.stability_margin) < 0.0:
            raise ValueError("update_equivalent.stability_margin must be non-negative.")

    def _normalized_search_routine(self) -> str:
        return normalize_search_routine(self.cfg.update_equivalent.routine)


__all__ = [
    "DoudizhuInferenceEngine",
    "DoudizhuInferenceEngineConfig",
]
