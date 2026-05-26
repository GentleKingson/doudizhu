"""Yomi belief modeling package."""

from __future__ import annotations

from typing import Any, Iterable


__all__ = [
    "BeliefTrainer",
    "BeliefSeqBuffer",
    "BeliefDecoderConcat",
    "BeliefTokenOutput",
    "CompletedEpisodeRootCollector",
    "ModelConfig",
    "TrainConfig",
    "YomiTrainingWorkflow",
]


_ATTR_TO_MODULE: dict[str, tuple[str, str]] = {
    "BeliefTrainer": ("yomi.training", "BeliefTrainer"),
    "BeliefSeqBuffer": ("yomi.buffer", "BeliefSeqBuffer"),
    "BeliefDecoderConcat": ("yomi.model", "BeliefDecoderConcat"),
    "BeliefTokenOutput": ("yomi.model", "BeliefTokenOutput"),
    "CompletedEpisodeRootCollector": ("yomi.runtime", "CompletedEpisodeRootCollector"),
    "ModelConfig": ("yomi.config", "ModelConfig"),
    "TrainConfig": ("yomi.config", "TrainConfig"),
    "YomiTrainingWorkflow": ("yomi.training", "YomiTrainingWorkflow"),
}


def __getattr__(name: str) -> Any:
    target = _ATTR_TO_MODULE.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = __import__(module_name, fromlist = [attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> Iterable[str]:
    return sorted(set(globals()) | set(__all__))
