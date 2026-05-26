"""Runtime subsystem modules for MARVEL."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable

__all__ = [
    "CheckpointStore",
    "CheckpointManager",
    "CheckpointState",
    "WandbCheckpointRef",
    "ModuleCompiler",
    "TorchCompileSettings",
    "DistributedContext",
    "DistributedRuntime",
    "NullProgressBar",
    "SamplerConfig",
    "Sampler",
    "TrainingRuntime",
    "MetricSerializer",
    "MetricSinks",
    "WandbSession",
    "RuntimeLoggingSession",
    "ParamStatsLogger",
    "TrainingLogger",
]


_ATTR_MODULES: Dict[str, tuple[str, str]] = {
    "CheckpointStore": ("marvel.runtime.checkpoints", "CheckpointStore"),
    "CheckpointManager": ("marvel.runtime.checkpoints", "CheckpointManager"),
    "CheckpointState": ("marvel.runtime.checkpoints", "CheckpointState"),
    "WandbCheckpointRef": ("marvel.runtime.checkpoints", "WandbCheckpointRef"),
    "ModuleCompiler": ("marvel.runtime.compilation", "ModuleCompiler"),
    "TorchCompileSettings": ("marvel.runtime.compilation", "TorchCompileSettings"),
    "DistributedContext": ("marvel.runtime.distributed", "DistributedContext"),
    "DistributedRuntime": ("marvel.runtime.distributed", "DistributedRuntime"),
    "NullProgressBar": ("marvel.runtime.distributed", "NullProgressBar"),
    "SamplerConfig": ("marvel.runtime.sampler", "SamplerConfig"),
    "Sampler": ("marvel.runtime.sampler", "Sampler"),
    "TrainingRuntime": ("marvel.runtime.orchestra", "TrainingRuntime"),
    "MetricSerializer": ("marvel.runtime.logger", "MetricSerializer"),
    "MetricSinks": ("marvel.runtime.logger", "MetricSinks"),
    "WandbSession": ("marvel.runtime.logger", "WandbSession"),
    "RuntimeLoggingSession": ("marvel.runtime.logger", "RuntimeLoggingSession"),
    "ParamStatsLogger": ("marvel.runtime.logger", "ParamStatsLogger"),
    "TrainingLogger": ("marvel.runtime.logger", "TrainingLogger"),
}


def __getattr__(name: str) -> Any:
    target = _ATTR_MODULES.get(name)
    if target is None:
        raise AttributeError(f"module 'marvel.runtime' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> Iterable[str]:
    return sorted(set(globals().keys()) | set(__all__))
