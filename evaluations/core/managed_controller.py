from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from evaluations.core.controller_dispatch import (
    ControllerDispatchPolicy,
    ControllerRuntime,
    DirectDispatchPolicy,
)
from evaluations.core.controller_pool import ControllerPoolKey, PooledPolicyController, WorkerLocalControllerPool
from evaluations.core.socket_worker import (
    PolicyWorkerClientConfigProtocol,
    PolicyWorkerRequest,
    SocketSubprocessControllerRuntime,
    SubprocessSocketClientRuntime,
    build_controller_subprocess_env,
    worker_scoped_resource_dir,
)
from evaluations.core.types import DecisionRequest, QueueEvent

ControllerReuseScope = Literal["per_worker", "per_gpu_token"]


class ManagedPolicyController:
    def __init__(
        self,
        *,
        pool: WorkerLocalControllerPool,
        job_id: str,
        worker_id: int,
        controller_id: str,
        resource_id: str,
        gpu_token: str | None,
        runtime_factory: Callable[[], ControllerRuntime],
        reuse_scope: ControllerReuseScope = "per_worker",
        pool_controller_id: str | None = None,
        dispatch_policy: ControllerDispatchPolicy | None = None,
        status_context: Mapping[str, object] | None = None,
        ready_body_factory: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        self._job_id = str(job_id)
        self._controller_id = str(controller_id)
        self._status_context = dict(status_context or {})
        self._ready_body_factory = ready_body_factory
        self._emit: Callable[[QueueEvent], None] | None = None
        self._runtime_factory = runtime_factory
        self._dispatch_policy = DirectDispatchPolicy() if dispatch_policy is None else dispatch_policy
        self._pooled = PooledPolicyController(
            pool = pool,
            key = ControllerPoolKey(
                worker_id = _controller_owner_id(
                    worker_id = int(worker_id),
                    gpu_token = gpu_token,
                    reuse_scope = reuse_scope,
                ),
                controller_id = str(pool_controller_id or controller_id),
                resource_id = str(resource_id),
                gpu_token = None if gpu_token is None else str(gpu_token),
            ),
            factory = self._build_runtime,
        )

    def bind_queue_event_emitter(self, emitter: Callable[[QueueEvent], None]) -> None:
        self._emit = emitter

    def open(self) -> None:
        self._pooled.open()
        body: dict[str, object] = {
            "stage": "controller_ready",
            "message": f"Controller {self._controller_id} ready.",
            "controller_id": self._controller_id,
        }
        if self._ready_body_factory is not None:
            body.update(dict(self._ready_body_factory()))
        self._emit_status(**body)

    def act_batch(self, requests: Sequence[DecisionRequest]) -> Sequence[object]:
        return self._pooled.act_batch(requests)

    def close(self) -> None:
        self._pooled.close()

    def _build_runtime(self) -> ControllerRuntime:
        return self._dispatch_policy.create_runtime(runtime_factory = self._runtime_factory)

    def _emit_status(self, **body: object) -> None:
        if self._emit is None:
            return
        payload = dict(self._status_context)
        payload.update(body)
        self._emit(QueueEvent(kind = "status", job_id = self._job_id, body = payload))


class ManagedSocketPolicyController:
    def __init__(
        self,
        *,
        pool: WorkerLocalControllerPool,
        logdir: Path,
        job_id: str,
        worker_id: int,
        controller_id: str,
        resource_id: str,
        gpu_token: str | None,
        repo_root: Path,
        command_factory: Callable[[Path, str], Sequence[str]],
        socket_name: str,
        connect_timeout_s: float,
        client_cfg: PolicyWorkerClientConfigProtocol,
        runtime_label: str,
        encode_requests: Callable[[Sequence[DecisionRequest]], Sequence[PolicyWorkerRequest]],
        format_failure: Callable[[str], str],
        reuse_scope: ControllerReuseScope = "per_worker",
        pool_controller_id: str | None = None,
        dispatch_policy: ControllerDispatchPolicy | None = None,
        status_context: Mapping[str, object] | None = None,
        ready_body_factory: Callable[[Path, str], Mapping[str, object]] | None = None,
        client_name: str | None = None,
        client_runtime_factory: Callable[..., object] | None = None,
    ) -> None:
        self._controller_id = str(controller_id)
        self._gpu_token = None if gpu_token is None else str(gpu_token)
        self._repo_root = Path(repo_root).resolve()
        self._command_factory = command_factory
        self._socket_name = str(socket_name)
        self._connect_timeout_s = float(connect_timeout_s)
        self._client_cfg = client_cfg
        self._client_name = str(client_name or controller_id)
        self._runtime_label = str(runtime_label)
        self._encode_requests = encode_requests
        self._format_failure = format_failure
        self._client_runtime_factory = client_runtime_factory
        self._env, self._subprocess_device = build_controller_subprocess_env(
            pythonpath_root = self._repo_root,
            gpu_token = self._gpu_token,
        )
        worker_dir = worker_scoped_resource_dir(
            logdir = logdir,
            worker_id = int(worker_id),
            controller_id = str(pool_controller_id or controller_id),
            resource_id = str(resource_id),
        )
        self._log_path = worker_dir / "controller.log"
        self._managed = ManagedPolicyController(
            pool = pool,
            job_id = job_id,
            worker_id = int(worker_id),
            controller_id = self._controller_id,
            resource_id = resource_id,
            gpu_token = self._gpu_token,
            runtime_factory = self._build_runtime,
            reuse_scope = reuse_scope,
            pool_controller_id = pool_controller_id,
            dispatch_policy = dispatch_policy,
            status_context = status_context,
            ready_body_factory = (
                None
                if ready_body_factory is None
                else lambda: dict(ready_body_factory(self._log_path, self._subprocess_device))
            ),
        )

    def bind_queue_event_emitter(self, emitter: Callable[[QueueEvent], None]) -> None:
        self._managed.bind_queue_event_emitter(emitter)

    def open(self) -> None:
        self._managed.open()

    def act_batch(self, requests: Sequence[DecisionRequest]) -> Sequence[object]:
        return self._managed.act_batch(requests)

    def close(self) -> None:
        self._managed.close()

    def _build_runtime(self) -> SocketSubprocessControllerRuntime:
        return SocketSubprocessControllerRuntime(
            command_factory = lambda socket_path: self._command_factory(socket_path, self._subprocess_device),
            cwd = self._repo_root,
            env = self._env,
            socket_name = self._socket_name,
            log_path = self._log_path,
            connect_timeout_s = self._connect_timeout_s,
            client_cfg = self._client_cfg,
            client_name = self._client_name,
            runtime_label = self._runtime_label,
            encode_requests = self._encode_requests,
            format_failure = self._format_failure,
            client_runtime_factory = self._client_runtime_factory,
        )


def _controller_owner_id(
    *,
    worker_id: int,
    gpu_token: str | None,
    reuse_scope: ControllerReuseScope,
) -> int:
    if reuse_scope == "per_worker":
        return int(worker_id)
    if gpu_token is None:
        return -1
    digest = hashlib.sha1(str(gpu_token).encode("utf-8")).hexdigest()[:8]
    return -int(digest, 16)


__all__ = [
    "ControllerReuseScope",
    "ManagedPolicyController",
    "ManagedSocketPolicyController",
]
