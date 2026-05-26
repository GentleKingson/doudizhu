from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, Sequence

from evaluations.core.dispatch import WorkerAssignment
from evaluations.core.types import (
    DecisionBatch,
    DecisionRequest,
    HostMessage,
    JobResult,
    QueueEvent,
)

if TYPE_CHECKING:
    from evaluations.core.plan import EvaluationRunPlan


class HostSession(Protocol):
    def open(self) -> None:
        ...

    def next_message(self) -> HostMessage:
        ...

    def submit_actions(self, batch_id: str, actions: Sequence[object]) -> None:
        ...

    def close(self) -> None:
        ...


class PolicyController(Protocol):
    def open(self) -> None:
        ...

    def act_batch(self, requests: Sequence[DecisionRequest]) -> Sequence[object]:
        ...

    def close(self) -> None:
        ...


class ResultAggregator(Protocol):
    def record_episode(self, body: dict[str, object]) -> None:
        ...

    def record_status(self, body: dict[str, object]) -> None:
        ...

    def finalize(self) -> JobResult:
        ...

    def build_failed_result(self, exc: Exception) -> JobResult:
        ...


class SummaryWriter(Protocol):
    def record_event(self, event: QueueEvent) -> None:
        ...

    def format_event(self, event: QueueEvent) -> str | None:
        ...

    def write_job(self, result: JobResult) -> Path:
        ...

    def write_run(self, results: Sequence[JobResult]) -> Path:
        ...

    def render_report(self, results: Sequence[JobResult]) -> str | None:
        ...


class JobRuntimeSession(Protocol):
    def open(self) -> None:
        ...

    def next_message(self) -> HostMessage:
        ...

    def resolve_decision_batch(self, batch: DecisionBatch) -> None:
        ...

    def record_status(self, body: dict[str, object]) -> None:
        ...

    def record_progress(self, update) -> None:  # noqa: ANN001
        ...

    def record_episode(self, body: dict[str, object]) -> None:
        ...

    def finalize(self) -> JobResult:
        ...

    def build_failed_result(self, exc: Exception) -> JobResult:
        ...

    def close(self) -> None:
        ...


class JobRuntimeFactory(Protocol):
    def create_session(
        self,
        *,
        assignment: WorkerAssignment,
        emit: Callable[[QueueEvent], None],
    ) -> JobRuntimeSession:
        ...


class EvaluationScenario(Protocol):
    name: str

    def configure_parser(self, parser) -> None:
        ...

    def plan_run(self, args) -> EvaluationRunPlan:
        ...


class QueueEventEmitterAware(Protocol):
    def bind_queue_event_emitter(self, emitter) -> None:
        ...


__all__ = [
    "EvaluationScenario",
    "HostSession",
    "JobRuntimeFactory",
    "JobRuntimeSession",
    "PolicyController",
    "QueueEventEmitterAware",
    "ResultAggregator",
    "SummaryWriter",
]
