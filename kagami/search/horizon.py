"""Shared game-horizon validation for search components."""

from __future__ import annotations

from typing import Any


def resolve_max_game_horizon(game: Any) -> int:
    raw = getattr(game, "MAX_GAME_HORIZON", None)
    if raw is None:
        raise RuntimeError("Game module is missing required MAX_GAME_HORIZON.")
    horizon = int(raw)
    if horizon <= 0:
        raise RuntimeError(f"MAX_GAME_HORIZON must be positive, got {horizon}.")
    return horizon


__all__ = ["resolve_max_game_horizon"]
