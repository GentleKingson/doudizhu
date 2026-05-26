"""Game-agnostic search components for Kagami internal inference improvements."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable

__all__ = [
    "LocalSearchSnapshot",
    "LocalSubgameSearchEngine",
    "CriticActionValueEstimator",
    "DEFAULT_LOCAL_SEARCH_FLOOR_CAP",
    "DEFAULT_LOCAL_SEARCH_STABILITY_MARGIN",
    "PendingPolicyAction",
    "PendingPolicyEval",
    "PendingPolicyQueryService",
    "PolicyActionValueEstimator",
    "PolicyAnchoredPcfrSolver",
    "PolicyRegularizedNoRegretSolver",
    "RolloutActionValueEstimator",
    "SEARCH_ROUTINE_LOCAL_SUBGAME",
    "SEARCH_ROUTINE_UPDATE_EQUIVALENT",
    "SEARCH_ROUTINES",
    "SearchRolloutSession",
    "UpdateEquivalentSearchEngine",
    "build_local_budget_ladder",
    "default_local_search_floor",
    "normalize_search_routine",
    "validate_search_budget",
    "validate_search_floor",
]


_ATTR_MODULES: Dict[str, tuple[str, str]] = {
    "LocalSearchSnapshot": ("kagami.search.local_subgame", "LocalSearchSnapshot"),
    "LocalSubgameSearchEngine": ("kagami.search.local_subgame", "LocalSubgameSearchEngine"),
    "CriticActionValueEstimator": ("kagami.search.critic_value_estimator", "CriticActionValueEstimator"),
    "DEFAULT_LOCAL_SEARCH_FLOOR_CAP": ("kagami.search.routines", "DEFAULT_LOCAL_SEARCH_FLOOR_CAP"),
    "DEFAULT_LOCAL_SEARCH_STABILITY_MARGIN": (
        "kagami.search.routines",
        "DEFAULT_LOCAL_SEARCH_STABILITY_MARGIN",
    ),
    "PendingPolicyAction": ("kagami.search.update_equivalent", "PendingPolicyAction"),
    "PendingPolicyEval": ("kagami.search.pending_policy", "PendingPolicyEval"),
    "PendingPolicyQueryService": ("kagami.search.pending_policy", "PendingPolicyQueryService"),
    "PolicyActionValueEstimator": ("kagami.search.value_estimator", "PolicyActionValueEstimator"),
    "PolicyAnchoredPcfrSolver": ("kagami.search.local_subgame", "PolicyAnchoredPcfrSolver"),
    "PolicyRegularizedNoRegretSolver": ("kagami.search.local_subgame", "PolicyRegularizedNoRegretSolver"),
    "RolloutActionValueEstimator": ("kagami.search.rollout_value_estimator", "RolloutActionValueEstimator"),
    "SEARCH_ROUTINE_LOCAL_SUBGAME": ("kagami.search.routines", "SEARCH_ROUTINE_LOCAL_SUBGAME"),
    "SEARCH_ROUTINE_UPDATE_EQUIVALENT": ("kagami.search.routines", "SEARCH_ROUTINE_UPDATE_EQUIVALENT"),
    "SEARCH_ROUTINES": ("kagami.search.routines", "SEARCH_ROUTINES"),
    "SearchRolloutSession": ("kagami.search.rollout_session", "SearchRolloutSession"),
    "UpdateEquivalentSearchEngine": ("kagami.search.update_equivalent", "UpdateEquivalentSearchEngine"),
    "build_local_budget_ladder": ("kagami.search.routines", "build_local_budget_ladder"),
    "default_local_search_floor": ("kagami.search.routines", "default_local_search_floor"),
    "normalize_search_routine": ("kagami.search.routines", "normalize_search_routine"),
    "validate_search_budget": ("kagami.search.routines", "validate_search_budget"),
    "validate_search_floor": ("kagami.search.routines", "validate_search_floor"),
}


def __getattr__(name: str) -> Any:
    target = _ATTR_MODULES.get(name)
    if target is None:
        raise AttributeError(f"module 'kagami.search' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> Iterable[str]:
    return sorted(set(globals().keys()) | set(__all__))
