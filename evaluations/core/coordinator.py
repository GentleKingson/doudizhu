from __future__ import annotations

import queue
import sys
import threading
from typing import Sequence

from evaluations.core.dispatch import LogicalJobDispatcher, WorkerAssignment
from evaluations.core.executor import JobCancelledError, JobExecutor
from evaluations.core.plan import EvaluationRunPlan
from evaluations.core.recording import RunEventRecorder
from evaluations.core.types import JobResult, QueueEvent, WorkerSlot


class EvaluationWorker(threading.Thread):
    def __init__(
        self,
        *,
        runtime_factory: object,
        slot: WorkerSlot,
        dispatcher: LogicalJobDispatcher,
        event_queue: "queue.Queue[QueueEvent]",
    ) -> None:
        super().__init__(name = f"EvaluationWorker[{slot.worker_id}]", daemon = True)
        self._runtime_factory = runtime_factory
        self._slot = slot
        self._dispatcher = dispatcher
        self._event_queue = event_queue
        self._executor_lock = threading.Lock()
        self._executor: JobExecutor | None = None
        self._stop_requested = threading.Event()

    def cancel(self) -> None:
        self._stop_requested.set()
        with self._executor_lock:
            executor = self._executor
        if executor is not None:
            executor.cancel()

    def run(self) -> None:
        while True:
            assignment: WorkerAssignment | None = None
            release_required = False
            try:
                if self._stop_requested.is_set():
                    return
                assignment = self._dispatcher.claim_assignment(slot = self._slot)
                if assignment is None:
                    return
                executor = JobExecutor(
                    assignment = assignment,
                    runtime_factory = self._runtime_factory,
                    emit = self._event_queue.put,
                )
                with self._executor_lock:
                    self._executor = executor
                release_required = True
                result = executor.run()
                self._dispatcher.finish_assignment(assignment, ok = bool(result.ok))
                release_required = False
            except JobCancelledError:
                if assignment is not None and release_required:
                    self._dispatcher.finish_assignment(assignment, ok = False)
                if self._stop_requested.is_set():
                    return
                raise
            except Exception as exc:  # noqa: BLE001
                if assignment is not None and release_required:
                    self._dispatcher.finish_assignment(assignment, ok = False)
                self._event_queue.put(
                    QueueEvent(
                        kind = "status",
                        job_id = "unknown" if assignment is None else assignment.assignment_id,
                        body = {
                            "stage": "worker_failed",
                            "message": f"Worker {self._slot.worker_id} failed: {exc}",
                            "level": "error",
                            "worker_id": int(self._slot.worker_id),
                            "error": str(exc),
                            **(
                                {}
                                if assignment is None
                                else {
                                    "logical_job_id": str(assignment.logical_job.logical_job_id),
                                    "worker_index": int(assignment.worker_index),
                                    "worker_count": int(assignment.worker_count),
                                }
                            ),
                        },
                    )
                )
                if assignment is not None:
                    self._event_queue.put(
                        QueueEvent(
                            kind = "result",
                            job_id = assignment.assignment_id,
                            body = {
                                "ok": False,
                                "payload": {
                                    "job_id": assignment.assignment_id,
                                    "logical_job_id": str(assignment.logical_job.logical_job_id),
                                    "scenario_name": assignment.logical_job.scenario_name,
                                    "mode": assignment.logical_job.mode,
                                    "error": str(exc),
                                    "worker_index": int(assignment.worker_index),
                                    "worker_count": int(assignment.worker_count),
                                },
                                "artifacts": {},
                            },
                        )
                    )
            finally:
                with self._executor_lock:
                    self._executor = None


class EvaluationCoordinator:
    def __init__(
        self,
        *,
        plan: EvaluationRunPlan,
    ) -> None:
        self._plan = plan

    def run(self) -> list[JobResult]:
        logical_jobs = self._plan.logical_jobs
        slots = self._plan.slots
        if not logical_jobs:
            raise ValueError("Expected at least one evaluation job.")
        if not slots:
            raise ValueError("Expected at least one worker slot.")

        event_queue: "queue.Queue[QueueEvent]" = queue.Queue()
        recorder: RunEventRecorder | None = None
        dispatcher = LogicalJobDispatcher(
            logical_jobs = logical_jobs,
            workloads = self._plan.workloads,
            slots = slots,
        )
        workers = [
            EvaluationWorker(
                runtime_factory = self._plan.runtime_factory,
                slot = slot,
                dispatcher = dispatcher,
                event_queue = event_queue,
            )
            for slot in slots
        ]
        started_workers: list[EvaluationWorker] = []

        try:
            recorder = RunEventRecorder(
                logdir = self._plan.logdir,
                summary_writer = self._plan.summary_writer,
            )
            for worker in workers:
                worker.start()
                started_workers.append(worker)

            while any(worker.is_alive() for worker in workers) or not event_queue.empty():
                try:
                    event = event_queue.get(timeout = 0.1)
                except queue.Empty:
                    continue
                try:
                    recorder.record(event)
                finally:
                    event_queue.task_done()

            self._join_workers(workers)
            return recorder.finalize(logical_jobs)
        except BaseException:
            live_workers = [worker for worker in started_workers if worker.is_alive()]
            if live_workers:
                self._cancel_workers(workers = live_workers)
            self._join_workers(started_workers)
            raise
        finally:
            cleanup_errors: list[str] = []
            if recorder is not None:
                self._safe_close(recorder, cleanup_errors = cleanup_errors, label = "run recorder")
            self._safe_close(self._plan, cleanup_errors = cleanup_errors, label = "run plan")
            if cleanup_errors:
                cleanup_message = "Evaluation cleanup failed: " + "; ".join(cleanup_errors)
                active_exc = sys.exc_info()[1]
                if active_exc is not None:
                    active_exc.add_note(cleanup_message)
                else:
                    raise RuntimeError(cleanup_message)

    @staticmethod
    def _cancel_workers(
        *,
        workers: Sequence[EvaluationWorker],
    ) -> None:
        for worker in workers:
            worker.cancel()

    @staticmethod
    def _join_workers(workers: Sequence[EvaluationWorker]) -> None:
        for worker in workers:
            worker.join(timeout = 5.0)

    @staticmethod
    def _safe_close(item: object, *, cleanup_errors: list[str], label: str) -> None:
        close_fn = getattr(item, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(f"{label}: {exc}")


__all__ = ["EvaluationCoordinator", "EvaluationWorker"]
