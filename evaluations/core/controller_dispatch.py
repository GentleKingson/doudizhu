from __future__ import annotations

import collections
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from evaluations.core.types import DecisionRequest


class ControllerRuntime(Protocol):
    def act_batch(self, requests: Sequence[DecisionRequest]) -> Sequence[object]:
        ...

    def close(self) -> None:
        ...


class ControllerDispatchPolicy(Protocol):
    def create_runtime(
        self,
        *,
        runtime_factory: Callable[[], ControllerRuntime],
    ) -> ControllerRuntime:
        ...


class DirectDispatchPolicy:
    def create_runtime(
        self,
        *,
        runtime_factory: Callable[[], ControllerRuntime],
    ) -> ControllerRuntime:
        return runtime_factory()


@dataclass
class _QueuedActCall:
    requests: tuple[DecisionRequest, ...]
    done: threading.Event = field(default_factory = threading.Event)
    actions: list[object] | None = None
    error: Exception | None = None


class _CoalescingDispatchRuntime:
    def __init__(
        self,
        *,
        runtime_factory: Callable[[], ControllerRuntime],
        max_batch_size: int,
        coalesce_ms: float,
        failure_label: str,
    ) -> None:
        self._runtime = runtime_factory()
        self._max_batch_size = max(1, int(max_batch_size))
        self._coalesce_s = max(0.0, float(coalesce_ms) / 1000.0)
        self._failure_label = str(failure_label)
        self._pending: collections.deque[_QueuedActCall] = collections.deque()
        self._closed = False
        self._worker_error: Exception | None = None
        self._condition = threading.Condition()
        self._worker = threading.Thread(
            target = self._run,
            name = "CoalescingDispatchRuntime",
            daemon = True,
        )
        self._worker.start()

    def act_batch(self, requests: Sequence[DecisionRequest]) -> list[object]:
        request_batch = tuple(requests)
        if not request_batch:
            return []
        queued = _QueuedActCall(requests = request_batch)
        with self._condition:
            self._raise_if_unavailable_locked()
            self._pending.append(queued)
            self._condition.notify()
        queued.done.wait()
        if queued.error is not None:
            raise RuntimeError(self._failure_label) from queued.error
        if queued.actions is None:
            raise RuntimeError("Coalesced controller batch completed without actions.")
        return list(queued.actions)

    def close(self) -> None:
        with self._condition:
            pending = list(self._pending)
            self._pending.clear()
            self._closed = True
            self._condition.notify_all()
        for queued in pending:
            queued.error = RuntimeError("Coalesced controller runtime closed before request completion.")
            queued.done.set()
        self._worker.join(timeout = 5.0)
        if self._worker.is_alive():
            raise RuntimeError("Timed out waiting for coalesced controller worker to stop.")
        self._runtime.close()

    def _run(self) -> None:
        active_calls: list[_QueuedActCall] = []
        try:
            while True:
                active_calls = self._take_ready_calls()
                if not active_calls:
                    return
                actions = self._run_microbatch(active_calls)
                self._fulfill_calls(active_calls, actions = actions)
                active_calls = []
        except Exception as exc:  # noqa: BLE001
            self._fail_calls(active_calls, error = exc)
            with self._condition:
                self._worker_error = exc
                pending = list(self._pending)
                self._pending.clear()
                self._condition.notify_all()
            self._fail_calls(pending, error = exc)

    def _take_ready_calls(self) -> list[_QueuedActCall]:
        with self._condition:
            while not self._pending and not self._closed and self._worker_error is None:
                self._condition.wait()
            self._raise_if_unavailable_locked()
            if self._closed and not self._pending:
                return []

            calls = [self._pending.popleft()]
            total_requests = len(calls[0].requests)
            if total_requests >= self._max_batch_size or self._coalesce_s <= 0.0:
                return calls

            deadline = time.perf_counter() + self._coalesce_s
            while total_requests < self._max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                if not self._pending:
                    self._condition.wait(timeout = remaining)
                while self._pending:
                    next_call = self._pending[0]
                    next_size = len(next_call.requests)
                    if total_requests + next_size > self._max_batch_size:
                        return calls
                    calls.append(self._pending.popleft())
                    total_requests += next_size
                    if total_requests >= self._max_batch_size:
                        return calls
            return calls

    def _run_microbatch(self, calls: Sequence[_QueuedActCall]) -> list[object]:
        flat_requests: list[DecisionRequest] = []
        for queued in calls:
            flat_requests.extend(queued.requests)
        actions = list(self._runtime.act_batch(flat_requests))
        if len(actions) != len(flat_requests):
            raise RuntimeError(
                "Coalesced controller returned an action count mismatch: "
                f"expected {len(flat_requests)} got {len(actions)}."
            )
        return actions

    @staticmethod
    def _fulfill_calls(calls: Sequence[_QueuedActCall], *, actions: Sequence[object]) -> None:
        offset = 0
        for queued in calls:
            next_offset = offset + len(queued.requests)
            queued.actions = list(actions[offset:next_offset])
            queued.done.set()
            offset = next_offset

    @staticmethod
    def _fail_calls(calls: Sequence[_QueuedActCall], *, error: Exception) -> None:
        for queued in calls:
            queued.error = error
            queued.done.set()

    def _raise_if_unavailable_locked(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("Coalesced controller worker failed.") from self._worker_error
        if self._closed:
            raise RuntimeError("Coalesced controller runtime is closed.")


class CoalescingDispatchPolicy:
    def __init__(
        self,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 1.0,
        failure_label: str = "Coalesced controller batch failed.",
    ) -> None:
        self._max_batch_size = int(max_batch_size)
        self._coalesce_ms = float(coalesce_ms)
        self._failure_label = str(failure_label)

    def create_runtime(
        self,
        *,
        runtime_factory: Callable[[], ControllerRuntime],
    ) -> ControllerRuntime:
        return _CoalescingDispatchRuntime(
            runtime_factory = runtime_factory,
            max_batch_size = self._max_batch_size,
            coalesce_ms = self._coalesce_ms,
            failure_label = self._failure_label,
        )


__all__ = [
    "CoalescingDispatchPolicy",
    "ControllerDispatchPolicy",
    "ControllerRuntime",
    "DirectDispatchPolicy",
]
