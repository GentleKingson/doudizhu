from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from evaluations.core.types import EvaluationJob, JobWorkload, WorkerSlot


@dataclass(frozen = True)
class EvaluationRunPlan:
    logdir: Path
    logical_jobs: tuple[EvaluationJob, ...]
    workloads: tuple[JobWorkload, ...]
    slots: tuple[WorkerSlot, ...]
    summary_writer: object
    runtime_factory: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "logdir", Path(self.logdir).resolve())
        object.__setattr__(self, "logical_jobs", tuple(self.logical_jobs))
        object.__setattr__(self, "workloads", tuple(self.workloads))
        object.__setattr__(self, "slots", tuple(self.slots))
        if not self.logical_jobs:
            raise ValueError("EvaluationRunPlan.logical_jobs must contain at least one job.")
        if len(self.logical_jobs) != len(self.workloads):
            raise ValueError("EvaluationRunPlan requires one workload per logical job.")
        if not self.slots:
            raise ValueError("EvaluationRunPlan.slots must contain at least one worker slot.")

    @classmethod
    def create(
        cls,
        *,
        logdir: Path,
        logical_jobs: Sequence[EvaluationJob],
        workloads: Sequence[JobWorkload],
        slots: Sequence[WorkerSlot],
        summary_writer: object,
        runtime_factory: object,
    ) -> EvaluationRunPlan:
        return cls(
            logdir = Path(logdir).resolve(),
            logical_jobs = tuple(logical_jobs),
            workloads = tuple(workloads),
            slots = tuple(slots),
            summary_writer = summary_writer,
            runtime_factory = runtime_factory,
        )

    def close(self) -> None:
        errors: list[str] = []
        for label, item in (
            ("summary_writer", self.summary_writer),
            ("runtime_factory", self.runtime_factory),
        ):
            close_fn = getattr(item, "close", None)
            if close_fn is None:
                continue
            try:
                close_fn()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{label}: {exc}")
        if errors:
            raise RuntimeError("Failed to close evaluation run plan resources: " + "; ".join(errors))


__all__ = ["EvaluationRunPlan"]
