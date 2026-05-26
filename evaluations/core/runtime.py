from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from evaluations.core.dispatch import WorkerAssignment
from evaluations.core.controller_pool import WorkerLocalControllerPool
from evaluations.core.types import (
    DecisionBatch,
    JobResult,
    ProgressUpdate,
    QueueEvent,
)


class HostedJobRuntimeSession:
    def __init__(
        self,
        *,
        assignment: WorkerAssignment,
        host: object,
        controllers: dict[str, object],
        aggregator: object,
        emit: Callable[[QueueEvent], None],
    ) -> None:
        self._assignment = assignment
        self._host = host
        self._controllers = dict(controllers)
        self._aggregator = aggregator
        self._emit = emit

    def open(self) -> None:
        self._bind_emitters([self._host, self._aggregator, *self._controllers.values()])
        for controller_id, controller in self._controllers.items():
            controller.open()
            self._emit_status(
                stage = "controller_opened",
                message = f"Opened controller {controller_id}.",
                controller_id = controller_id,
            )
        self._host.open()
        self._emit_status(
            stage = "host_opened",
            message = "Host session opened.",
        )

    def next_message(self):
        return self._host.next_message()

    def resolve_decision_batch(self, batch: DecisionBatch) -> None:
        controller = self._controllers.get(batch.controller_id)
        if controller is None:
            raise RuntimeError(
                "Missing controller for controller_id="
                f"{batch.controller_id!r} in assignment {self._assignment.assignment_id}."
            )
        actions = list(controller.act_batch(batch.requests))
        if len(actions) != len(batch.requests):
            raise RuntimeError(
                "PolicyController.act_batch returned a result count mismatch: "
                f"expected {len(batch.requests)} got {len(actions)}."
            )
        self._host.submit_actions(batch.batch_id, actions)

    def record_status(self, body: Mapping[str, object]) -> None:
        self._aggregator.record_status(dict(body))

    def record_progress(self, update: ProgressUpdate) -> None:
        self._aggregator.record_status(dict(update.body))

    def record_episode(self, body: Mapping[str, object]) -> None:
        self._aggregator.record_episode(dict(body))

    def finalize(self) -> JobResult:
        return self._aggregator.finalize()

    def build_failed_result(self, exc: Exception) -> JobResult:
        return self._aggregator.build_failed_result(exc)

    def close(self) -> None:
        self._safe_close(self._host, label = "host")
        for controller in reversed(tuple(self._controllers.values())):
            self._safe_close(controller, label = "controller")

    def _bind_emitters(self, objects: Sequence[object]) -> None:
        def _forward(event: QueueEvent) -> None:
            if event.kind in {"status", "progress"}:
                record_status = getattr(self._aggregator, "record_status", None)
                if callable(record_status):
                    record_status(event.body)
            self._emit(event)

        for item in objects:
            binder = getattr(item, "bind_queue_event_emitter", None)
            if callable(binder):
                binder(_forward)

    def _emit_status(self, **body: object) -> None:
        payload = self._status_context()
        payload.update(body)
        self._emit(QueueEvent(kind = "status", job_id = self._assignment.assignment_id, body = payload))

    def _status_context(self) -> dict[str, object]:
        context = self._assignment.logical_job.config.get("status_context")
        if not isinstance(context, Mapping):
            payload: dict[str, object] = {}
        else:
            payload = {}
            for key, value in context.items():
                payload[str(key)] = value
        payload["logical_job_id"] = str(self._assignment.logical_job.logical_job_id)
        payload["worker_index"] = int(self._assignment.worker_index)
        payload["worker_count"] = int(self._assignment.worker_count)
        return payload

    def _safe_close(self, item: object, *, label: str) -> None:
        close_fn = getattr(item, "close", None)
        if close_fn is None:
            return
        try:
            close_fn()
        except Exception as exc:  # noqa: BLE001
            self._emit_status(
                stage = "close_failed",
                message = f"Failed to close {label}: {exc}",
                level = "warning",
                error = str(exc),
                target = label,
            )


class CallbackJobRuntimeFactory:
    def __init__(
        self,
        *,
        create_host,
        create_controllers,
        create_aggregator,
    ) -> None:
        self._create_host = create_host
        self._create_controllers = create_controllers
        self._create_aggregator = create_aggregator

    def create_session(
        self,
        *,
        assignment: WorkerAssignment,
        emit: Callable[[QueueEvent], None],
    ) -> HostedJobRuntimeSession:
        return HostedJobRuntimeSession(
            assignment = assignment,
            host = self._create_host(assignment),
            controllers = dict(self._create_controllers(assignment)),
            aggregator = self._create_aggregator(assignment),
            emit = emit,
        )


class RunScopedControllerRuntimeFactory:
    def __init__(self, *, controller_pool: WorkerLocalControllerPool) -> None:
        self._controller_pool = controller_pool

    @property
    def controller_pool(self) -> WorkerLocalControllerPool:
        return self._controller_pool

    def create_session(
        self,
        *,
        assignment: WorkerAssignment,
        emit: Callable[[QueueEvent], None],
    ) -> HostedJobRuntimeSession:
        runtime_factory = CallbackJobRuntimeFactory(
            create_host = self._create_host,
            create_controllers = self._create_controllers,
            create_aggregator = self._create_aggregator,
        )
        return runtime_factory.create_session(assignment = assignment, emit = emit)

    def close(self) -> None:
        self._controller_pool.close()

    def _create_host(self, assignment):  # noqa: ANN001
        raise NotImplementedError

    def _create_controllers(self, assignment):  # noqa: ANN001
        raise NotImplementedError

    def _create_aggregator(self, assignment):  # noqa: ANN001
        raise NotImplementedError


__all__ = ["CallbackJobRuntimeFactory", "HostedJobRuntimeSession", "RunScopedControllerRuntimeFactory"]
