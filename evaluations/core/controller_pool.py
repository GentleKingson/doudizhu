from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from evaluations.core.controller_dispatch import ControllerRuntime
from evaluations.core.types import DecisionRequest


@dataclass(frozen = True)
class ControllerPoolKey:
    worker_id: int
    controller_id: str
    resource_id: str
    gpu_token: str | None = None


@dataclass
class _PoolEntry:
    runtime: ControllerRuntime
    lease_count: int = 0


TControllerRuntime = TypeVar("TControllerRuntime", bound = ControllerRuntime)


class WorkerLocalControllerPool:
    def __init__(self) -> None:
        self._entries: dict[ControllerPoolKey, _PoolEntry] = {}
        self._lock = threading.Lock()

    def acquire(
        self,
        key: ControllerPoolKey,
        *,
        factory: Callable[[], TControllerRuntime],
    ) -> TControllerRuntime:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                runtime = factory()
                entry = _PoolEntry(runtime = runtime, lease_count = 0)
                self._entries[key] = entry
            entry.lease_count += 1
            return entry.runtime  # type: ignore[return-value]

    def release(self, key: ControllerPoolKey) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.lease_count <= 0:
                raise RuntimeError(f"Controller pool release underflow for key={key!r}.")
            entry.lease_count -= 1

    def invalidate(self, key: ControllerPoolKey) -> None:
        with self._lock:
            entry = self._entries.pop(key, None)
        if entry is not None:
            entry.runtime.close()

    def close(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries = {}
        errors: list[str] = []
        for entry in entries:
            try:
                entry.runtime.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Failed to close controller pool runtimes: " + "; ".join(errors))


class PooledPolicyController:
    def __init__(
        self,
        *,
        pool: WorkerLocalControllerPool,
        key: ControllerPoolKey,
        factory: Callable[[], TControllerRuntime],
    ) -> None:
        self._pool = pool
        self._key = key
        self._factory = factory
        self._runtime: TControllerRuntime | None = None
        self._invalidated = False

    def open(self) -> None:
        if self._runtime is not None:
            return
        self._runtime = self._pool.acquire(self._key, factory = self._factory)
        self._invalidated = False

    def act_batch(self, requests: Sequence[DecisionRequest]) -> Sequence[object]:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("PooledPolicyController.open() must be called before act_batch().")
        try:
            return runtime.act_batch(requests)
        except Exception:
            if not self._invalidated:
                self._pool.invalidate(self._key)
                self._invalidated = True
            self._runtime = None
            raise

    def close(self) -> None:
        if self._runtime is None:
            return
        if not self._invalidated:
            self._pool.release(self._key)
        self._runtime = None
        self._invalidated = False


__all__ = [
    "ControllerPoolKey",
    "PooledPolicyController",
    "WorkerLocalControllerPool",
]
