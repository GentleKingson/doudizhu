from __future__ import annotations

# ── Python 3.7 compatibility ──────────────────────────────────────────
# The PerfectDou agent subprocess loads this module under Python 3.7.
# Only the server-side routines (run_policy_worker_server,
# PolicyWorkerRequest, send_json, recv_json, …) execute there; client
# classes require 3.10+.
#
# Guarded constructs:
#   • typing.Protocol            (3.8+)   → ``Protocol = object`` stand-in
#   • Callable[…] subscript      (3.9+)   → TYPE_CHECKING only
#   • zip(strict=…)              (3.10+)  → omitted (False is the default)
#   • evaluations.core.types /
#     controller_dispatch        (3.10+)  → TYPE_CHECKING only
# ──────────────────────────────────────────────────────────────────────

import collections
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Protocol

    from evaluations.core.controller_dispatch import ControllerRuntime  # noqa: F401
    from evaluations.core.types import DecisionRequest  # noqa: F401

    PolicyWorker = Callable[[Sequence["PolicyWorkerRequest"]], list[str]]
else:
    Protocol = object

PROTOCOL_VERSION = 1


@dataclass(frozen = True)
class PolicyWorkerRequest:
    request_id: int
    infoset_str: str
    other_state_strings: dict[str, str] | None = None


@dataclass(frozen = True)
class PolicyWorkerAction:
    request_id: int
    action: str


class PolicyWorkerClientConfigProtocol(Protocol):
    connect_timeout_s: float
    request_timeout_s: float
    max_batch_size: int
    max_delay_ms: int


class PolicyWorkerServerConfigProtocol(Protocol):
    socket_timeout_s: float


def send_json(conn: socket.socket, obj: dict[str, object]) -> None:
    data = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack("!I", len(data)) + data)


def recv_json(conn: socket.socket) -> dict[str, object]:
    header = conn.recv(4)
    if len(header) < 4:
        raise ConnectionError("Socket closed before header received.")
    (length,) = struct.unpack("!I", header)
    payload = b""
    while len(payload) < length:
        chunk = conn.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("Socket closed mid-message.")
        payload += chunk
    return json.loads(payload.decode("utf-8"))


def parse_policy_worker_requests(obj: Mapping[str, object]) -> list[PolicyWorkerRequest]:
    raw = obj.get("requests", [])
    if not isinstance(raw, list):
        raise ValueError("requests must be a list")
    out: list[PolicyWorkerRequest] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("requests items must be objects")
        request_id = int(item.get("id"))
        infoset_str = str(item.get("infoset_str", ""))
        if not infoset_str:
            raise ValueError("request missing infoset_str")
        other_raw = item.get("other_state_strings")
        other_state_strings = None
        if other_raw is not None:
            if not isinstance(other_raw, dict):
                raise ValueError("other_state_strings must be an object")
            other_state_strings = {str(key): str(value) for key, value in other_raw.items()}
        out.append(
            PolicyWorkerRequest(
                request_id = request_id,
                infoset_str = infoset_str,
                other_state_strings = other_state_strings,
            )
        )
    return out


def format_policy_worker_actions(
    requests: Sequence[PolicyWorkerRequest],
    actions: Sequence[str],
) -> dict[str, object]:
    if len(actions) != len(requests):
        raise ValueError(f"Expected {len(requests)} actions, got {len(actions)}.")
    return {
        "ok": True,
        "actions": [
            {"id": request.request_id, "action": str(action)}
            for request, action in zip(requests, actions)
        ],
    }


@dataclass(frozen = True)
class _Pending:
    request: PolicyWorkerRequest
    future: concurrent.futures.Future[str]


class PolicyWorkerClient:
    def __init__(
        self,
        socket_path: Path | str,
        *,
        cfg: PolicyWorkerClientConfigProtocol,
        name: str = "",
    ) -> None:
        self._socket_path = Path(str(socket_path)).resolve()
        self._name = str(name)
        self._connect_timeout_s = float(cfg.connect_timeout_s)
        self._request_timeout_s = float(cfg.request_timeout_s)
        self._max_batch_size = int(cfg.max_batch_size)
        self._max_delay_s = float(cfg.max_delay_ms) / 1000.0

        self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._conn.settimeout(self._connect_timeout_s)
        self._conn.connect(str(self._socket_path))
        self._conn.settimeout(self._request_timeout_s)

        send_json(self._conn, {"type": "hello"})
        resp = recv_json(self._conn)
        if not resp.get("ok"):
            raise RuntimeError(f"Handshake failed: {resp.get('error')}")

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._io_lock = threading.Lock()
        self._queue: collections.deque[_Pending] = collections.deque()
        self._closed = False
        self._worker_error: Exception | None = None
        self._worker = threading.Thread(
            target = self._run,
            name = f"PolicyWorkerClient[{self._name}]",
            daemon = True,
        )
        self._worker.start()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        io_lock_acquired = self._io_lock.acquire(blocking = False)
        if io_lock_acquired:
            try:
                send_json(self._conn, {"type": "shutdown"})
            except Exception:
                pass
            finally:
                self._io_lock.release()
        try:
            self._conn.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        self._conn.close()
        self._worker.join(timeout = 1.0)

    def submit(
        self,
        request_id: int,
        infoset_str: str,
        other_state_strings: dict[str, str] | None = None,
    ) -> concurrent.futures.Future[str]:
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        pending = _Pending(
            request = PolicyWorkerRequest(
                request_id = int(request_id),
                infoset_str = str(infoset_str),
                other_state_strings = other_state_strings,
            ),
            future = future,
        )
        with self._cv:
            if self._closed:
                raise RuntimeError("Client already closed.")
            self._queue.append(pending)
            self._cv.notify_all()
        return future

    def query(
        self,
        request_id: int,
        infoset_str: str,
        other_state_strings: dict[str, str] | None = None,
    ) -> str:
        future = self.submit(
            request_id,
            infoset_str,
            other_state_strings = other_state_strings,
        )
        return future.result(timeout = self._request_timeout_s)

    def act_batch(self, requests: Sequence[PolicyWorkerRequest]) -> list[str]:
        with self._io_lock:
            send_json(
                self._conn,
                {
                    "type": "act_batch",
                    "requests": [
                        {
                            "id": int(request.request_id),
                            "infoset_str": str(request.infoset_str),
                            **(
                                {"other_state_strings": dict(request.other_state_strings)}
                                if request.other_state_strings
                                else {}
                            ),
                        }
                        for request in requests
                    ],
                },
            )
            resp = recv_json(self._conn)
        if not resp.get("ok"):
            raise RuntimeError(str(resp.get("error")))
        actions = resp.get("actions", [])
        if not isinstance(actions, list) or len(actions) != len(requests):
            raise RuntimeError("Policy worker response batch size mismatch.")
        out: list[str] = []
        for expected, item in zip(requests, actions):
            if not isinstance(item, dict):
                raise RuntimeError("Invalid action payload.")
            if int(item.get("id")) != int(expected.request_id):
                raise RuntimeError("Action id mismatch.")
            out.append(str(item.get("action", "")))
        return out

    def _run(self) -> None:
        try:
            while True:
                batch: list[_Pending] = []
                deadline = None
                with self._cv:
                    while not self._queue and not self._closed:
                        self._cv.wait(timeout = 0.1)
                    if self._closed and not self._queue:
                        return
                    if self._queue:
                        deadline = time.monotonic() + self._max_delay_s
                    while self._queue and len(batch) < self._max_batch_size:
                        batch.append(self._queue.popleft())
                        if len(batch) >= self._max_batch_size:
                            break
                        now = time.monotonic()
                        if deadline is not None and now >= deadline:
                            break
                        remaining = 0.0 if deadline is None else max(0.0, deadline - now)
                        if remaining > 0.0 and len(batch) < self._max_batch_size and not self._queue:
                            self._cv.wait(timeout = remaining)

                if not batch:
                    continue

                requests = [item.request for item in batch]
                try:
                    actions = self.act_batch(requests)
                except Exception as exc:  # noqa: BLE001
                    for item in batch:
                        item.future.set_exception(exc)
                    if self._closed:
                        return
                    raise RuntimeError(
                        f"PolicyWorkerClient[{self._name}] act_batch failed for {len(batch)} request(s)."
                    ) from exc

                for item, action in zip(batch, actions):
                    item.future.set_result(action)
        except Exception as exc:  # noqa: BLE001
            if self._closed:
                return
            self._worker_error = exc
            with self._cv:
                self._closed = True
                while self._queue:
                    item = self._queue.popleft()
                    item.future.set_exception(exc)
            raise


def run_policy_worker_server(
    socket_path: Path | str,
    policy: PolicyWorker,
    *,
    cfg: PolicyWorkerServerConfigProtocol,
    server_label: str = "policy worker socket server",
    exit_after_disconnect: bool = False,
) -> None:
    resolved_socket_path = Path(str(socket_path)).resolve()
    _ensure_socket_dir(resolved_socket_path)
    _unlink_if_exists(resolved_socket_path)

    timeout_s = float(cfg.socket_timeout_s)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(resolved_socket_path))
    server.listen(1)
    try:
        while True:
            conn, _ = server.accept()
            conn.settimeout(timeout_s)
            try:
                _serve_policy_worker_connection(conn, policy = policy)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc(file = sys.stderr)
                try:
                    send_json(conn, {"ok": False, "error": str(exc)})
                except Exception:
                    pass
                raise RuntimeError(f"Connection handling failed in {server_label}.") from exc
            finally:
                conn.close()
            if bool(exit_after_disconnect):
                return
    finally:
        server.close()
        _unlink_if_exists(resolved_socket_path)


class SocketSubprocessControllerRuntime:
    def __init__(
        self,
        *,
        command_factory: Callable[[Path], Sequence[str]],
        cwd: Path,
        env: Mapping[str, str],
        socket_name: str,
        log_path: Path,
        connect_timeout_s: float,
        client_cfg: PolicyWorkerClientConfigProtocol,
        client_name: str,
        runtime_label: str,
        encode_requests: Callable[[Sequence[DecisionRequest]], Sequence[PolicyWorkerRequest]],
        format_failure: Callable[[str], str],
        client_runtime_factory: Callable[..., object] | None = None,
    ) -> None:
        self._encode_requests = encode_requests
        self._format_failure = format_failure
        runtime_factory = SubprocessSocketClientRuntime if client_runtime_factory is None else client_runtime_factory
        self._runtime = runtime_factory(
            command_factory = command_factory,
            cwd = cwd,
            env = dict(env),
            socket_name = socket_name,
            log_path = log_path,
            connect_timeout_s = connect_timeout_s,
            client_factory = lambda socket_path: PolicyWorkerClient(
                socket_path,
                cfg = client_cfg,
                name = client_name,
            ),
            runtime_label = runtime_label,
        )

    def act_batch(self, requests: Sequence[DecisionRequest]) -> Sequence[object]:
        payloads = list(self._encode_requests(requests))
        client = self._runtime.client
        try:
            return list(client.act_batch(payloads))
        except Exception as exc:
            message = self._format_failure(self._runtime.failure_context())
            error_text = str(exc).strip()
            if error_text and error_text not in message:
                message = f"{message}\nCause: {error_text}"
            raise RuntimeError(message) from exc

    def close(self) -> None:
        self._runtime.close()


class SubprocessSocketClientRuntime:
    def __init__(
        self,
        *,
        command_factory: Callable[[Path], Sequence[str]],
        cwd: Path,
        env: dict[str, str],
        socket_name: str,
        log_path: Path,
        connect_timeout_s: float,
        client_factory: Callable[[Path], object],
        runtime_label: str,
    ) -> None:
        self._command_factory = command_factory
        self._cwd = Path(cwd).resolve()
        self._env = dict(env)
        self._socket_name = _bounded_socket_name(socket_name)
        self._log_path = Path(log_path).resolve()
        self._connect_timeout_s = float(connect_timeout_s)
        self._client_factory = client_factory
        self._runtime_label = str(runtime_label)

        self._socket_dir: Path | None = None
        self._socket_path: Path | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._client: object | None = None
        self._log_handle = None
        self._log_thread: threading.Thread | None = None
        self._log_done = threading.Event()
        self._tail: collections.deque[str] = collections.deque(maxlen = 40)
        self._tail_lock = threading.Lock()

        self._open()

    @property
    def client(self) -> object:
        if self._client is None:
            raise RuntimeError(f"{self._runtime_label} client is not open.")
        return self._client

    def failure_context(self) -> str:
        proc_state = "process already closed"
        if self._proc is not None:
            exit_code = self._proc.poll()
            if exit_code is None:
                proc_state = "process still running"
            else:
                proc_state = f"process exited with code {int(exit_code)}"
        return (
            f"Log: {self._log_path}\n"
            f"State: {proc_state}\n"
            f"Recent output:\n{self._tail_text()}"
        )

    def close(self) -> None:
        if self._client is not None:
            close_client = getattr(self._client, "close", None)
            if callable(close_client):
                close_client()
            self._client = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout = 1.0)
            except subprocess.TimeoutExpired:
                _kill_process_tree(self._proc)
                try:
                    self._proc.wait(timeout = 1.0)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f"Timed out waiting for {self._runtime_label} process to exit."
                    ) from exc
            self._proc = None
        if self._log_thread is not None:
            self._log_done.set()
            self._log_thread.join(timeout = 1.0)
            self._log_thread = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if self._socket_path is not None and self._socket_path.exists():
            self._socket_path.unlink(missing_ok = True)
        if self._socket_dir is not None:
            shutil.rmtree(self._socket_dir, ignore_errors = True)
            self._socket_dir = None
            self._socket_path = None

    def _open(self) -> None:
        self._log_path.parent.mkdir(parents = True, exist_ok = True)
        socket_tmpdir = _preferred_socket_tmpdir()
        self._socket_dir = Path(
            tempfile.mkdtemp(
                prefix = f"{self._socket_name}_",
                dir = None if socket_tmpdir is None else str(socket_tmpdir),
            )
        )
        self._socket_path = self._socket_dir / f"{self._socket_name}.sock"
        self._log_handle = self._log_path.open("w", encoding = "utf-8")
        command = [str(part) for part in self._command_factory(self._socket_path)]
        self._proc = subprocess.Popen(
            command,
            cwd = self._cwd,
            env = self._env,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            bufsize = 1,
            start_new_session = True,
        )
        self._log_thread = threading.Thread(
            target = self._forward_logs,
            name = f"SubprocessSocketRuntime[{self._runtime_label}]",
            daemon = True,
        )
        self._log_thread.start()
        try:
            _wait_for_socket(self._socket_path, timeout_s = self._connect_timeout_s)
            self._client = self._client_factory(self._socket_path)
        except Exception as exc:
            cleanup_error: Exception | None = None
            startup_context = self.failure_context()
            try:
                self.close()
            except Exception as close_exc:  # noqa: BLE001
                cleanup_error = close_exc
            message = f"Failed to start {self._runtime_label}.\n{startup_context}"
            if cleanup_error is not None:
                message += f"\nCleanup error: {cleanup_error}"
            raise RuntimeError(message) from exc

    def _forward_logs(self) -> None:
        if self._proc is None or self._proc.stdout is None or self._log_handle is None:
            return
        try:
            for raw_line in self._proc.stdout:
                line = str(raw_line)
                with self._tail_lock:
                    self._tail.append(line.rstrip("\n"))
                self._log_handle.write(line)
                self._log_handle.flush()
                if self._log_done.is_set():
                    break
        finally:
            self._log_done.set()

    def _tail_text(self) -> str:
        with self._tail_lock:
            tail = [line for line in self._tail if line.strip()]
        if tail:
            return "\n".join(tail)
        file_tail = self._log_tail_from_file()
        if file_tail:
            return "\n".join(file_tail)
        return "<no process output>"

    def _log_tail_from_file(self) -> list[str]:
        if not self._log_path.exists():
            return []
        try:
            with self._log_path.open("r", encoding = "utf-8") as handle:
                tail = collections.deque((line.rstrip("\n") for line in handle), maxlen = 40)
        except OSError:
            return []
        return [line for line in tail if line.strip()]


def worker_scoped_resource_dir(
    *,
    logdir: Path,
    worker_id: int,
    controller_id: str,
    resource_id: str,
) -> Path:
    return (
        Path(logdir).resolve()
        / "workers"
        / f"worker_{int(worker_id):03d}"
        / "controllers"
        / _bounded_resource_dir_component(controller_id)
        / _bounded_resource_dir_component(resource_id)
    )


def build_controller_subprocess_env(
    *,
    pythonpath_root: Path,
    gpu_token: str | None,
    base_env: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], str]:
    env = os.environ.copy() if base_env is None else dict(base_env)
    env["PYTHONPATH"] = f"{Path(pythonpath_root).resolve()}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if gpu_token is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
        return env, "cpu"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_token)
    return env, "cuda"


def _ensure_socket_dir(socket_path: Path) -> None:
    socket_path.parent.mkdir(parents = True, exist_ok = True)


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _serve_policy_worker_connection(conn: socket.socket, *, policy: PolicyWorker) -> None:
    hello = recv_json(conn)
    if hello.get("type") != "hello":
        send_json(conn, {"ok": False, "error": "expected hello"})
        return
    send_json(conn, {"ok": True, "protocol": PROTOCOL_VERSION})

    while True:
        try:
            req = recv_json(conn)
        except ConnectionError:
            return
        msg_type = req.get("type")
        if msg_type == "shutdown":
            try:
                send_json(conn, {"ok": True})
            except OSError:
                pass
            return
        if msg_type != "act_batch":
            send_json(conn, {"ok": False, "error": "expected act_batch"})
            return

        requests = parse_policy_worker_requests(req)
        actions = policy(requests)
        send_json(conn, format_policy_worker_actions(requests, actions))


def _sanitize_name_component(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return normalized or "resource"


def _bounded_resource_dir_component(text: str, *, max_len: int = 96) -> str:
    normalized = _sanitize_name_component(text)
    if len(normalized) <= int(max_len):
        return normalized
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    prefix_len = max(1, int(max_len) - len(digest) - 1)
    prefix = normalized[:prefix_len].rstrip("._-")
    if not prefix:
        prefix = "resource"
    return f"{prefix}_{digest}"


def _bounded_socket_name(text: str, *, max_len: int = 24) -> str:
    normalized = _sanitize_name_component(text)
    if len(normalized) <= int(max_len):
        return normalized
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    prefix_len = max(1, int(max_len) - len(digest) - 1)
    prefix = normalized[:prefix_len].rstrip("._-")
    if not prefix:
        prefix = "sock"
    return f"{prefix}_{digest}"


def _preferred_socket_tmpdir() -> Path | None:
    if os.name == "posix":
        candidate = Path("/tmp")
        if candidate.is_dir() and os.access(candidate, os.W_OK | os.X_OK):
            return candidate
    return None


def _wait_for_socket(path: Path, *, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        if stat.S_ISSOCK(mode):
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for socket: {path}")


def _kill_process_tree(proc: subprocess.Popen[str], *, timeout_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


__all__ = [
    "PROTOCOL_VERSION",
    "PolicyWorkerAction",
    "PolicyWorkerClient",
    "PolicyWorkerClientConfigProtocol",
    "PolicyWorkerRequest",
    "PolicyWorkerServerConfigProtocol",
    "SocketSubprocessControllerRuntime",
    "SubprocessSocketClientRuntime",
    "build_controller_subprocess_env",
    "format_policy_worker_actions",
    "parse_policy_worker_requests",
    "recv_json",
    "run_policy_worker_server",
    "send_json",
    "worker_scoped_resource_dir",
]
