"""Marvel algorithm components."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable

__all__ = [
    "TrainConfig",
    "Agent",
    "CentralCritic",
    "AgentOutput",
    "ActiveAgentOutput",
    "CriticOutput",
    "elementwise_kl",
    "entropy",
    "EnvOutcome",
    "EnvStepBatch",
    "TrajectoryBuffer",
    "BufferFieldSpec",
    "main",
]


_ATTR_MODULES: Dict[str, tuple[str, str]] = {
    "TrainConfig": ("marvel.config", "TrainConfig"),
    "Agent": ("marvel.model", "Agent"),
    "CentralCritic": ("marvel.model", "CentralCritic"),
    "AgentOutput": ("marvel.model", "AgentOutput"),
    "ActiveAgentOutput": ("marvel.model", "ActiveAgentOutput"),
    "CriticOutput": ("marvel.model", "CriticOutput"),
    "elementwise_kl": ("marvel.utils", "elementwise_kl"),
    "entropy": ("marvel.utils", "entropy"),
    "EnvOutcome": ("marvel.env_batch", "EnvOutcome"),
    "EnvStepBatch": ("marvel.env_batch", "EnvStepBatch"),
    "TrajectoryBuffer": ("marvel.buffer", "TrajectoryBuffer"),
    "BufferFieldSpec": ("marvel.buffer", "BufferFieldSpec"),
    "main": ("marvel.train", "main"),
}


def __getattr__(name: str) -> Any:
    target = _ATTR_MODULES.get(name)
    if target is None:
        raise AttributeError(f"module 'marvel' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> Iterable[str]:
    return sorted(set(globals().keys()) | set(__all__))
