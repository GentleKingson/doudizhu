from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


QueueEventKind = Literal["status", "progress", "episode", "result"]


@dataclass(frozen = True)
class JobWorkload:
    total_units: int
    unit_name: str
    shardable: bool = True

    def __post_init__(self) -> None:
        if int(self.total_units) <= 0:
            raise ValueError("JobWorkload.total_units must be positive.")
        if not str(self.unit_name).strip():
            raise ValueError("JobWorkload.unit_name must be non-empty.")


@dataclass(frozen = True)
class QueueEvent:
    kind: QueueEventKind
    job_id: str
    body: dict[str, object]


@dataclass(frozen = True)
class EvaluationJob:
    job_id: str
    scenario_name: str
    mode: str
    artifacts_dir: Path
    requires_gpu: bool
    logical_job_id: str = ""
    config: dict[str, object] = field(default_factory = dict)

    def __post_init__(self) -> None:
        if not str(self.logical_job_id).strip():
            object.__setattr__(self, "logical_job_id", str(self.job_id))


@dataclass(frozen = True)
class DecisionRequest:
    controller_id: str
    acting_entity_id: str
    episode_id: str
    decision_index: int
    payload: object


@dataclass(frozen = True)
class JobResult:
    job_id: str
    ok: bool
    payload: dict[str, object]
    artifacts: dict[str, str] = field(default_factory = dict)


@dataclass(frozen = True)
class WorkerSlot:
    worker_id: int
    gpu_token: str | None = None


@dataclass(frozen = True)
class DecisionBatch:
    batch_id: str
    controller_id: str
    requests: tuple[DecisionRequest, ...]


@dataclass(frozen = True)
class StatusUpdate:
    body: dict[str, object]


@dataclass(frozen = True)
class ProgressUpdate:
    body: dict[str, object]


@dataclass(frozen = True)
class EpisodeUpdate:
    body: dict[str, object]


@dataclass(frozen = True)
class Finished:
    pass


HostMessage = DecisionBatch | StatusUpdate | ProgressUpdate | EpisodeUpdate | Finished


__all__ = [
    "DecisionBatch",
    "DecisionRequest",
    "EpisodeUpdate",
    "EvaluationJob",
    "Finished",
    "HostMessage",
    "JobWorkload",
    "JobResult",
    "ProgressUpdate",
    "QueueEvent",
    "QueueEventKind",
    "StatusUpdate",
    "WorkerSlot",
]
