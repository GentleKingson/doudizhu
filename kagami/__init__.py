"""Kagami inference/evaluation engine layer."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable

__all__ = [
    "DoudizhuInferenceEngine",
    "DoudizhuInferenceEngineConfig",
    "InferenceEngine",
    "PolicySpec",
    "PolicySpecParser",
]


_ATTR_MODULES: Dict[str, tuple[str, str]] = {
    "DoudizhuInferenceEngine": ("kagami.games.doudizhu", "DoudizhuInferenceEngine"),
    "DoudizhuInferenceEngineConfig": ("kagami.games.doudizhu", "DoudizhuInferenceEngineConfig"),
    "InferenceEngine": ("kagami.core.interfaces", "InferenceEngine"),
    "PolicySpec": ("kagami.core.specs", "PolicySpec"),
    "PolicySpecParser": ("kagami.core.specs", "PolicySpecParser"),
}


def __getattr__(name: str) -> Any:
    target = _ATTR_MODULES.get(name)
    if target is None:
        raise AttributeError(f"module 'kagami' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> Iterable[str]:
    return sorted(set(globals().keys()) | set(__all__))
