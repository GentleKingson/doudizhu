from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable

__all__ = [
    "Agent",
    "BaseMarvelModule",
    "CheckpointableModule",
    "CentralCritic",
    "KVCache",
    "load_checkpoint_payload_file",
    "save_checkpoint_payload_file",
    "TrainableModule",
    "TensorDict",
    "AgentOutput",
    "ActiveAgentOutput",
    "CriticOutput",
]


_ATTR_MODULES: Dict[str, tuple[str, str]] = {
    "Agent": ("marvel.model.agent", "Agent"),
    "BaseMarvelModule": ("marvel.model.base", "BaseMarvelModule"),
    "CheckpointableModule": ("marvel.model.checkpointable", "CheckpointableModule"),
    "CentralCritic": ("marvel.model.critic", "CentralCritic"),
    "KVCache": ("marvel.model.cache", "KVCache"),
    "load_checkpoint_payload_file": ("marvel.model.checkpointable", "load_checkpoint_payload_file"),
    "save_checkpoint_payload_file": ("marvel.model.checkpointable", "save_checkpoint_payload_file"),
    "TrainableModule": ("marvel.model.trainable", "TrainableModule"),
    "TensorDict": ("marvel.model.tensordict", "TensorDict"),
    "AgentOutput": ("marvel.model.types", "AgentOutput"),
    "ActiveAgentOutput": ("marvel.model.types", "ActiveAgentOutput"),
    "CriticOutput": ("marvel.model.types", "CriticOutput"),
}


def __getattr__(name: str) -> Any:
    target = _ATTR_MODULES.get(name)
    if target is None:
        raise AttributeError(f"module 'marvel.model' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> Iterable[str]:
    return sorted(set(globals().keys()) | set(__all__))
