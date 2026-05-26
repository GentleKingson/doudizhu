from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen = True)
class WorkerLayout:
    workers: int
    slot_tokens: tuple[str | None, ...]
    usable_gpu_tokens: tuple[str, ...]
    workers_per_gpu: int | None = None

    def __post_init__(self) -> None:
        if int(self.workers) <= 0:
            raise ValueError("WorkerLayout.workers must be positive.")
        if len(self.slot_tokens) != int(self.workers):
            raise ValueError("WorkerLayout.slot_tokens must contain one entry per worker.")
        if self.workers_per_gpu is not None and int(self.workers_per_gpu) <= 0:
            raise ValueError("WorkerLayout.workers_per_gpu must be positive when provided.")

    def uses_gpu_slots(self) -> bool:
        return any(token is not None for token in self.slot_tokens)

    def resolved_device(self) -> str:
        return "cuda" if self.uses_gpu_slots() else "cpu"


class WorkerLayoutPlanner:
    def __init__(
        self,
        *,
        discover_visible_tokens = None,
        probe_token = None,
    ) -> None:
        self._discover_visible_tokens = (
            discover_visible_gpu_tokens if discover_visible_tokens is None else discover_visible_tokens
        )
        self._probe_token = probe_gpu_token if probe_token is None else probe_token

    def discover_visible_gpu_tokens(self) -> list[str]:
        return [str(token) for token in self._discover_visible_tokens()]

    def discover_usable_gpu_tokens(self) -> list[str]:
        return [
            str(token)
            for token in self.discover_visible_gpu_tokens()
            if bool(self._probe_token(str(token)))
        ]

    def build(
        self,
        *,
        workers: int,
        prefer_gpu: bool,
        require_even_gpu_distribution: bool = False,
    ) -> WorkerLayout:
        return build_worker_layout(
            workers = int(workers),
            usable_tokens = self.discover_usable_gpu_tokens(),
            prefer_gpu = bool(prefer_gpu),
            require_even_gpu_distribution = bool(require_even_gpu_distribution),
        )

    def build_for_workers_per_gpu(
        self,
        *,
        workers_per_gpu: int,
        prefer_gpu: bool,
    ) -> WorkerLayout:
        return build_worker_layout_for_workers_per_gpu(
            workers_per_gpu = int(workers_per_gpu),
            usable_tokens = self.discover_usable_gpu_tokens(),
            prefer_gpu = bool(prefer_gpu),
        )


def discover_visible_gpu_tokens() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        text = str(raw).strip()
        if text == "":
            return []
        return [item.strip() for item in text.split(",") if item.strip()]
    if not torch.cuda.is_available():
        return []
    return [str(idx) for idx in range(int(torch.cuda.device_count()))]


def probe_gpu_token(token: str) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(token)
    cmd = [
        sys.executable,
        "-c",
        "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)",
    ]
    try:
        proc = subprocess.run(
            cmd,
            env = env,
            check = False,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        )
    except OSError:
        return False
    return proc.returncode == 0


def discover_usable_gpu_tokens() -> list[str]:
    return [str(token) for token in discover_visible_gpu_tokens() if probe_gpu_token(str(token))]


def resolve_worker_slot_device(
    *,
    gpu_token: str | None,
    visible_devices_env: str | None = None,
) -> str:
    if gpu_token is None:
        return "cpu"

    token_text = str(gpu_token)
    visible_text = visible_devices_env if visible_devices_env is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_text is None:
        if not token_text.isdigit():
            raise RuntimeError(
                "Non-numeric gpu_token requires CUDA_VISIBLE_DEVICES to map worker-local CUDA ordinals: "
                f"gpu_token={token_text!r}"
            )
        local_index = int(token_text)
    else:
        visible_tokens = [item.strip() for item in str(visible_text).split(",") if item.strip()]
        if not visible_tokens:
            raise RuntimeError("Worker slot requested CUDA placement but CUDA_VISIBLE_DEVICES is empty.")
        try:
            local_index = visible_tokens.index(token_text)
        except ValueError as exc:
            raise RuntimeError(
                "Worker slot gpu_token is not present in the current CUDA_VISIBLE_DEVICES mapping: "
                f"gpu_token={token_text!r} visible={visible_tokens!r}"
            ) from exc
    return f"cuda:{local_index}"


def build_worker_layout(
    *,
    workers: int,
    usable_tokens: Sequence[str],
    prefer_gpu: bool,
    require_even_gpu_distribution: bool = False,
) -> WorkerLayout:
    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError("workers must be a positive integer.")

    normalized_tokens = tuple(str(token) for token in usable_tokens)
    if (not prefer_gpu) or len(normalized_tokens) == 0:
        return WorkerLayout(
            workers = worker_count,
            slot_tokens = tuple(None for _ in range(worker_count)),
            usable_gpu_tokens = normalized_tokens,
            workers_per_gpu = None,
        )

    if require_even_gpu_distribution and worker_count % len(normalized_tokens) != 0:
        raise ValueError(
            "workers must be divisible by number of usable GPUs "
            f"(workers={worker_count}, usable_gpus={len(normalized_tokens)})."
        )

    slot_tokens = tuple(
        str(normalized_tokens[idx % len(normalized_tokens)])
        for idx in range(worker_count)
    )
    workers_per_gpu = None
    if require_even_gpu_distribution:
        workers_per_gpu = worker_count // len(normalized_tokens)
    return WorkerLayout(
        workers = worker_count,
        slot_tokens = slot_tokens,
        usable_gpu_tokens = normalized_tokens,
        workers_per_gpu = workers_per_gpu,
    )


def build_worker_layout_for_workers_per_gpu(
    *,
    workers_per_gpu: int,
    usable_tokens: Sequence[str],
    prefer_gpu: bool,
) -> WorkerLayout:
    workers_per_gpu_count = int(workers_per_gpu)
    if workers_per_gpu_count <= 0:
        raise ValueError("workers_per_gpu must be a positive integer.")

    normalized_tokens = tuple(str(token) for token in usable_tokens)
    if (not prefer_gpu) or len(normalized_tokens) == 0:
        return build_worker_layout(
            workers = 1,
            usable_tokens = normalized_tokens,
            prefer_gpu = False,
        )

    return build_worker_layout(
        workers = workers_per_gpu_count * len(normalized_tokens),
        usable_tokens = normalized_tokens,
        prefer_gpu = True,
        require_even_gpu_distribution = True,
    )


__all__ = [
    "WorkerLayout",
    "WorkerLayoutPlanner",
    "build_worker_layout",
    "build_worker_layout_for_workers_per_gpu",
    "discover_usable_gpu_tokens",
    "discover_visible_gpu_tokens",
    "probe_gpu_token",
    "resolve_worker_slot_device",
]
