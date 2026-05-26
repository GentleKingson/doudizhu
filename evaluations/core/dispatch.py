from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Sequence

from evaluations.core.types import EvaluationJob, JobWorkload, WorkerSlot


class SharedUnitAllocator:
    def __init__(self, *, total_units: int) -> None:
        if int(total_units) <= 0:
            raise ValueError("SharedUnitAllocator.total_units must be positive.")
        self._total_units = int(total_units)
        self._next_unit_id = 1
        self._lock = threading.Lock()

    @property
    def total_units(self) -> int:
        return int(self._total_units)

    def claim_unit_ids(self, *, max_count: int) -> tuple[int, ...]:
        if int(max_count) <= 0:
            raise ValueError("SharedUnitAllocator.max_count must be positive.")
        with self._lock:
            if self._next_unit_id > self._total_units:
                return ()
            start = int(self._next_unit_id)
            stop = min(int(self._total_units), int(start) + int(max_count) - 1)
            self._next_unit_id = int(stop) + 1
        return tuple(range(int(start), int(stop) + 1))

    def all_units_claimed(self) -> bool:
        with self._lock:
            return int(self._next_unit_id) > int(self._total_units)

    def has_unclaimed_units(self) -> bool:
        return not self.all_units_claimed()


@dataclass(frozen = True)
class WorkerAssignment:
    assignment_id: str
    logical_job: EvaluationJob
    slot: WorkerSlot
    worker_index: int
    worker_count: int
    work_allocator: SharedUnitAllocator


@dataclass
class _LogicalJobState:
    logical_job: EvaluationJob
    workload: JobWorkload
    worker_count: int
    work_allocator: SharedUnitAllocator
    available_worker_indices: list[int]
    active_count: int = 0
    failed: bool = False


class LogicalJobDispatcher:
    def __init__(
        self,
        *,
        logical_jobs: Sequence[EvaluationJob],
        workloads: Sequence[JobWorkload],
        slots: Sequence[WorkerSlot],
    ) -> None:
        if len(logical_jobs) != len(workloads):
            raise ValueError("LogicalJobDispatcher requires one workload per logical job.")
        if not logical_jobs:
            raise ValueError("LogicalJobDispatcher requires at least one logical job.")
        if not slots:
            raise ValueError("LogicalJobDispatcher requires at least one worker slot.")
        self._lock = threading.Lock()
        self._states = [
            _LogicalJobState(
                logical_job = job,
                workload = workload,
                worker_count = _compute_worker_count(workload = workload, slot_count = len(slots)),
                work_allocator = SharedUnitAllocator(total_units = int(workload.total_units)),
                available_worker_indices = list(
                    range(_compute_worker_count(workload = workload, slot_count = len(slots)))
                ),
            )
            for job, workload in zip(logical_jobs, workloads, strict = False)
        ]

    def claim_assignment(self, *, slot: WorkerSlot) -> WorkerAssignment | None:
        with self._lock:
            for state in self._states:
                if state.failed:
                    continue
                if state.active_count >= int(state.worker_count):
                    continue
                if not state.work_allocator.has_unclaimed_units():
                    continue
                worker_index = state.available_worker_indices.pop(0)
                state.active_count += 1
                assignment_id = _build_assignment_id(
                    logical_job_id = state.logical_job.logical_job_id,
                    worker_index = int(worker_index),
                    worker_count = int(state.worker_count),
                )
                return WorkerAssignment(
                    assignment_id = assignment_id,
                    logical_job = state.logical_job,
                    slot = slot,
                    worker_index = int(worker_index),
                    worker_count = int(state.worker_count),
                    work_allocator = state.work_allocator,
                )
        return None

    def finish_assignment(self, assignment: WorkerAssignment, *, ok: bool) -> None:
        logical_job_id = str(assignment.logical_job.logical_job_id)
        with self._lock:
            for state in self._states:
                if str(state.logical_job.logical_job_id) != logical_job_id:
                    continue
                state.active_count -= 1
                if state.active_count < 0:
                    raise RuntimeError(
                        "LogicalJobDispatcher.finish_assignment() active_count underflow "
                        f"for logical_job_id={logical_job_id!r}."
                    )
                state.available_worker_indices.append(int(assignment.worker_index))
                state.available_worker_indices.sort()
                if not ok:
                    state.failed = True
                return
        raise RuntimeError(f"Unknown logical_job_id in finish_assignment(): {logical_job_id!r}.")


def _compute_worker_count(*, workload: JobWorkload, slot_count: int) -> int:
    if not bool(workload.shardable):
        return 1
    return min(int(workload.total_units), int(slot_count))


def _build_assignment_id(*, logical_job_id: str, worker_index: int, worker_count: int) -> str:
    return (
        f"{str(logical_job_id)}_worker"
        f"{int(worker_index) + 1:03d}of{int(worker_count):03d}"
    )


__all__ = ["LogicalJobDispatcher", "SharedUnitAllocator", "WorkerAssignment"]
