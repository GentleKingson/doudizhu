from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable, Tuple


TrainerClass = type[Any]


class AlgorithmRegistry:
    """Registry mapping algorithm identifiers to trainer classes."""

    def __init__(self) -> None:
        self._handlers: Dict[str, TrainerClass] = {}
        self._builtin_handlers: Dict[str, tuple[str, str]] = {}

    def register(self, name: str, trainer_cls: TrainerClass) -> None:
        key = name.lower()
        if key in self._handlers or key in self._builtin_handlers:
            raise ValueError(f"Algorithm '{name}' is already registered.")
        self._handlers[key] = trainer_cls

    def register_builtin(self, name: str, module_name: str, attr_name: str) -> None:
        key = name.lower()
        if key in self._handlers or key in self._builtin_handlers:
            raise ValueError(f"Algorithm '{name}' is already registered.")
        self._builtin_handlers[key] = (module_name, attr_name)

    def resolve(self, name: str) -> TrainerClass:
        key = name.lower()
        trainer_cls = self._handlers.get(key)
        if trainer_cls is not None:
            return trainer_cls
        target = self._builtin_handlers.get(key)
        if target is None:
            available = ", ".join(sorted(set(self._handlers) | set(self._builtin_handlers))) or "none"
            raise KeyError(f"Unknown algorithm '{name}'. Available algorithms: {available}.")
        module_name, attr_name = target
        module = import_module(module_name)
        trainer_cls = getattr(module, attr_name)
        self._handlers[key] = trainer_cls
        globals()[attr_name] = trainer_cls
        return trainer_cls

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        key = name.lower()
        return key in self._handlers or key in self._builtin_handlers

    def available(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self._handlers) | set(self._builtin_handlers)))


registry = AlgorithmRegistry()


def register_algorithm(name: str, trainer_cls: TrainerClass) -> None:
    registry.register(name, trainer_cls)


def available_algorithms() -> Tuple[str, ...]:
    return registry.available()

_ATTR_MODULES: Dict[str, tuple[str, str]] = {
    "IPPOTrainer": ("marvel.training.ippo", "IPPOTrainer"),
    "VRPOTrainer": ("marvel.training.vrpo", "VRPOTrainer"),
}

registry.register_builtin("ippo", "marvel.training.ippo", "IPPOTrainer")
registry.register_builtin("vrpo", "marvel.training.vrpo", "VRPOTrainer")

__all__ = [
    "AlgorithmRegistry",
    "TrainerClass",
    "registry",
    "register_algorithm",
    "available_algorithms",
    "IPPOTrainer",
    "VRPOTrainer",
]


def __getattr__(name: str) -> Any:
    target = _ATTR_MODULES.get(name)
    if target is None:
        raise AttributeError(f"module 'marvel.training' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> Iterable[str]:
    return sorted(set(globals().keys()) | set(__all__))
