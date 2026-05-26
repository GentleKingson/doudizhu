from __future__ import annotations

from collections.abc import Callable, Mapping
import threading

from evaluations.core.dispatch import WorkerAssignment
from evaluations.core.protocols import JobRuntimeFactory
from evaluations.core.types import (
    DecisionBatch,
    EpisodeUpdate,
    Finished,
    JobResult,
    ProgressUpdate,
    QueueEvent,
    StatusUpdate,
)


class JobCancelledError(Exception):
    pass


class JobExecutor:
    def __init__(
        self,
        *,
        assignment: WorkerAssignment,
        runtime_factory: JobRuntimeFactory,
        emit: Callable[[QueueEvent], None],
    ) -> None:
        self._assignment = assignment
        self._runtime_factory = runtime_factory
        self._emit = emit
        self._cancel_requested = threading.Event()
        self._runtime_lock = threading.Lock()
        self._session: object | None = None

    def cancel(self) -> None:
        self._cancel_requested.set()
        self._close_runtime()

    def run(self) -> JobResult:
        session = None
        job_display_name = self._job_display_name()
        has_status_context = self._has_user_status_context()
        self._emit_status(
            stage = "job_started",
            message = "Starting job." if has_status_context else f"Starting job {job_display_name}.",
            mode = self._assignment.logical_job.mode,
            worker_id = int(self._assignment.slot.worker_id),
            gpu_token = self._assignment.slot.gpu_token,
        )
        try:
            session = self._runtime_factory.create_session(
                assignment = self._assignment,
                emit = self._emit,
            )
            with self._runtime_lock:
                self._session = session
            self._raise_if_cancelled()
            session.open()

            while True:
                self._raise_if_cancelled()
                message = session.next_message()
                if isinstance(message, DecisionBatch):
                    self._raise_if_cancelled()
                    session.resolve_decision_batch(message)
                    continue
                if isinstance(message, StatusUpdate):
                    session.record_status(message.body)
                    self._emit(
                        QueueEvent(
                            kind = "status",
                            job_id = self._assignment.assignment_id,
                            body = dict(message.body),
                        )
                    )
                    continue
                if isinstance(message, ProgressUpdate):
                    session.record_progress(message)
                    self._emit(
                        QueueEvent(
                            kind = "progress",
                            job_id = self._assignment.assignment_id,
                            body = dict(message.body),
                        )
                    )
                    continue
                if isinstance(message, EpisodeUpdate):
                    session.record_episode(message.body)
                    self._emit(
                        QueueEvent(
                            kind = "episode",
                            job_id = self._assignment.assignment_id,
                            body = dict(message.body),
                        )
                    )
                    continue
                if isinstance(message, Finished):
                    break
                raise RuntimeError(f"Unsupported HostMessage type: {type(message).__name__}.")

            self._raise_if_cancelled()
            result = session.finalize()
        except JobCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            if self._cancel_requested.is_set():
                raise JobCancelledError(f"Assignment {self._assignment.assignment_id} cancelled.") from None
            self._emit_status(
                stage = "job_failed",
                message = f"Job failed: {exc}" if has_status_context else f"Job {job_display_name} failed: {exc}",
                level = "error",
                error = str(exc),
            )
            if session is None:
                result = JobResult(
                    job_id = self._assignment.assignment_id,
                    ok = False,
                    payload = {
                        "job_id": self._assignment.assignment_id,
                        "logical_job_id": self._assignment.logical_job.logical_job_id,
                        "scenario_name": self._assignment.logical_job.scenario_name,
                        "mode": self._assignment.logical_job.mode,
                        "error": str(exc),
                        "worker_index": int(self._assignment.worker_index),
                        "worker_count": int(self._assignment.worker_count),
                    },
                    artifacts = {},
                )
            else:
                result = session.build_failed_result(exc)
        finally:
            self._close_runtime()

        self._emit(
            QueueEvent(
                kind = "result",
                job_id = self._assignment.assignment_id,
                body = {
                    "ok": bool(result.ok),
                    "payload": dict(result.payload),
                    "artifacts": dict(result.artifacts),
                },
            )
        )
        return result

    def _emit_status(self, **body: object) -> None:
        payload = self._status_context()
        payload.update(body)
        self._emit(QueueEvent(kind = "status", job_id = self._assignment.assignment_id, body = payload))

    def _job_display_name(self) -> str:
        display_name = self._assignment.logical_job.config.get("display_name")
        if display_name is None:
            return str(self._assignment.logical_job.job_id)
        return str(display_name)

    def _status_context(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        context = self._assignment.logical_job.config.get("status_context")
        if isinstance(context, Mapping):
            for key, value in context.items():
                payload[str(key)] = value
        payload["logical_job_id"] = str(self._assignment.logical_job.logical_job_id)
        payload["worker_index"] = int(self._assignment.worker_index)
        payload["worker_count"] = int(self._assignment.worker_count)
        return payload

    def _has_user_status_context(self) -> bool:
        return isinstance(self._assignment.logical_job.config.get("status_context"), Mapping)

    def _raise_if_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise JobCancelledError(f"Assignment {self._assignment.assignment_id} cancelled.")

    def _close_runtime(self) -> None:
        with self._runtime_lock:
            session = self._session
            self._session = None
        if session is not None:
            self._safe_close(session, label = "runtime")

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


__all__ = ["JobCancelledError", "JobExecutor"]
