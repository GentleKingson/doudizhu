"""Shared Kagami search-routine validation and budget helpers."""

from __future__ import annotations


SEARCH_ROUTINE_UPDATE_EQUIVALENT = "update-equiv"
SEARCH_ROUTINE_LOCAL_SUBGAME = "local-subgame"
SEARCH_ROUTINES = (
    SEARCH_ROUTINE_UPDATE_EQUIVALENT,
    SEARCH_ROUTINE_LOCAL_SUBGAME,
)
DEFAULT_LOCAL_SEARCH_STABILITY_MARGIN = 0.05
DEFAULT_LOCAL_SEARCH_FLOOR_CAP = 16


def normalize_search_routine(value: str | None) -> str:
    routine = SEARCH_ROUTINE_LOCAL_SUBGAME if value is None else str(value).strip().lower()
    if routine not in SEARCH_ROUTINES:
        choices = ", ".join(SEARCH_ROUTINES)
        raise ValueError(f"search routine must be one of: {choices}.")
    return routine


def default_local_search_floor(search_budget: int) -> int:
    budget = int(search_budget)
    validate_search_budget(budget)
    return min(DEFAULT_LOCAL_SEARCH_FLOOR_CAP, budget)


def validate_search_budget(value: int, *, name: str = "search_budget") -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_search_floor(
    search_floor: int | None,
    *,
    search_budget: int,
    name: str = "search_floor",
) -> None:
    if search_floor is None:
        return
    floor = int(search_floor)
    budget = int(search_budget)
    validate_search_budget(budget)
    if floor <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    if floor > budget:
        raise ValueError(f"{name} must be <= search_budget.")


def build_local_budget_ladder(*, search_floor: int, search_budget: int) -> tuple[int, ...]:
    floor = int(search_floor)
    budget = int(search_budget)
    validate_search_floor(floor, search_budget = budget)

    ladder = [floor]
    current = floor
    while current < budget:
        current = min(int(current * 2), budget)
        if current != ladder[-1]:
            ladder.append(current)
    if ladder[-1] != budget:
        ladder.append(budget)
    return tuple(ladder)


__all__ = [
    "DEFAULT_LOCAL_SEARCH_FLOOR_CAP",
    "DEFAULT_LOCAL_SEARCH_STABILITY_MARGIN",
    "SEARCH_ROUTINE_LOCAL_SUBGAME",
    "SEARCH_ROUTINE_UPDATE_EQUIVALENT",
    "SEARCH_ROUTINES",
    "build_local_budget_ladder",
    "default_local_search_floor",
    "normalize_search_routine",
    "validate_search_budget",
    "validate_search_floor",
]
