"""Distributed training helpers for MARVEL."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen = True)
class DistributedContext:
    is_initialized: bool
    is_primary: bool
    world_size: int
    rank: int
    local_rank: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.is_initialized and self.world_size > 1

    def wrap_model(
        self,
        model: torch.nn.Module,
        *,
        find_unused_parameters: bool = False,
    ) -> torch.nn.Module:
        if not self.enabled:
            return model
        if self.device.type != "cuda":
            raise RuntimeError("Distributed training requires CUDA.")
        return DDP(
            model,
            device_ids = [self.device.index],
            output_device = self.device,
            find_unused_parameters = bool(find_unused_parameters),
        )

    def _reduce_scalar(
        self,
        value: float | int | torch.Tensor | None,
        *,
        average: bool,
    ) -> float | int | None:
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone()
        else:
            tensor = torch.tensor(value)

        target_device = torch.device("cpu")
        if self.enabled and dist.get_backend() == "nccl":
            # NCCL reductions require CUDA tensors.
            target_device = self.device
        if tensor.device != target_device:
            tensor = tensor.to(target_device)

        if self.enabled:
            dist.all_reduce(tensor, op = dist.ReduceOp.SUM)
            if average:
                tensor = tensor / self.world_size

        result = tensor.item()
        if average or tensor.is_floating_point():
            return float(result)
        return int(result)

    def reduce_mean(self, value: float | int | torch.Tensor | None) -> float | None:
        return self._reduce_scalar(value, average = True)

    def reduce_sum(self, value: float | int | torch.Tensor | None) -> float | int | None:
        return self._reduce_scalar(value, average = False)


class NullProgressBar:
    """Stub progress bar used on non-primary ranks."""

    def set_postfix(self, **_kwargs: Any) -> None:
        return None

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def close(self) -> None:  # pragma: no cover - trivial
        return None


class DistributedRuntime:
    """Object-oriented setup/execution wrapper for distributed training context."""

    def __init__(self, context: DistributedContext) -> None:
        self.context = context

    @property
    def enabled(self) -> bool:
        return self.context.enabled

    @property
    def is_primary(self) -> bool:
        return self.context.is_primary

    @property
    def world_size(self) -> int:
        return self.context.world_size

    @property
    def rank(self) -> int:
        return self.context.rank

    @property
    def local_rank(self) -> int:
        return self.context.local_rank

    @property
    def device(self) -> torch.device:
        return self.context.device

    def wrap_model(
        self,
        model: torch.nn.Module,
        *,
        find_unused_parameters: bool = False,
    ) -> torch.nn.Module:
        return self.context.wrap_model(
            model,
            find_unused_parameters = find_unused_parameters,
        )

    def reduce_mean(self, value: float | int | torch.Tensor | None) -> float | None:
        return self.context.reduce_mean(value)

    def reduce_sum(self, value: float | int | torch.Tensor | None) -> float | int | None:
        return self.context.reduce_sum(value)

    @staticmethod
    def _available_cpu_count() -> int:
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return max(os.cpu_count() or 1, 1)
        except OSError:
            return max(os.cpu_count() or 1, 1)

    @classmethod
    def _context_from_env(cls, cfg_device: str) -> DistributedContext:
        """Initialise torch.distributed using environment variables if available."""
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        env_rank = int(os.environ.get("RANK", "0"))
        env_local_rank = int(os.environ.get("LOCAL_RANK", env_rank))

        want_distributed = env_world_size > 1
        device = torch.device(cfg_device)

        if device.type == "cuda" and not torch.cuda.is_available():
            if env_rank != 0:
                raise SystemExit(0)
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

        if want_distributed and not torch.cuda.is_available():
            if env_rank != 0:
                raise SystemExit(0)
            want_distributed = False
        if want_distributed:
            # MARVEL distributed training is CUDA-only; force CUDA device selection.
            device = torch.device("cuda", env_local_rank)
            torch.cuda.set_device(device)

        if want_distributed and not dist.is_initialized():
            backend = "nccl" if device.type == "cuda" else "gloo"
            dist.init_process_group(backend = backend, init_method = "env://")

        is_initialized = dist.is_initialized()
        rank = dist.get_rank() if is_initialized else 0
        world_size = dist.get_world_size() if is_initialized else 1
        local_rank = env_local_rank if is_initialized else 0

        # Explicit values, including 1, are part of the execution configuration.
        explicit_threads = os.environ.get("OMP_NUM_THREADS")
        omp_threads = int(explicit_threads) if explicit_threads is not None else 0
        if explicit_threads is not None and omp_threads <= 0:
            raise ValueError("OMP_NUM_THREADS must be a positive integer.")
        if rank == 0 and explicit_threads is None:
            omp_threads = cls._available_cpu_count()
            os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        if omp_threads > 0:
            torch.set_num_threads(omp_threads)

        return DistributedContext(
            is_initialized = is_initialized,
            is_primary = rank == 0,
            world_size = world_size,
            rank = rank,
            local_rank = local_rank,
            device = device,
        )

    @classmethod
    def from_env(cls, cfg_device: str) -> DistributedRuntime:
        return cls(cls._context_from_env(cfg_device))

    @classmethod
    def init_from_env(cls, cfg_device: str) -> DistributedContext:
        """Compatibility helper returning the raw distributed context."""
        return cls._context_from_env(cfg_device)

__all__ = [
    "DistributedContext",
    "DistributedRuntime",
    "NullProgressBar",
]
