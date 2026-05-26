"""Train-time evaluation hooks for the Dou Dizhu-only MARVEL package."""

from __future__ import annotations

from marvel.eval.hooks import (
    TrainingEvaluator,
    available,
    evaluate_checkpoint,
    resolve,
)

__all__ = [
    "TrainingEvaluator",
    "available",
    "evaluate_checkpoint",
    "resolve",
]
