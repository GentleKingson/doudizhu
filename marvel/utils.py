from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch

from marvel.config import TrainConfig

# NOTE: keep this module light on training dependencies; avoid importing training.* here.


def get_game_attr(game: object, name: str):
    """Return a game attribute, accepting either snake_case or UPPER_CASE."""
    if hasattr(game, name):
        return getattr(game, name)
    alt = name.upper()
    if hasattr(game, alt):
        return getattr(game, alt)
    raise AttributeError(f"Game module missing expected attribute '{name}' or '{alt}'.")


def elementwise_kl(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute KL divergence for each element pair in ``a`` and ``b``."""
    a_safe = a + eps
    b_safe = b + eps
    a_log = torch.log(a_safe)
    b_log = torch.log(b_safe)
    kl_values = (a_safe * (a_log - b_log)).sum(dim = -1)
    return kl_values


def set_global_seed(seed: int, *, offset: int = 0) -> None:
    seed = int(seed) + int(offset)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(cfg: TrainConfig, *, log: Callable[[str], None] | None = None) -> None:
    """Apply global torch runtime settings based on TrainConfig.

    Note: If you need CUDA-level determinism for GEMMs (mm/bmm/matmul), set
    `CUBLAS_WORKSPACE_CONFIG` before launching Python.
    """
    if log is None:
        log = print

    deterministic = bool(getattr(cfg, "deterministic", False))
    disable_tf32 = bool(getattr(cfg, "deterministic_disable_tf32", False))

    # Match the previous default behaviour (fast matmuls) unless determinism requests otherwise.
    if deterministic and disable_tf32:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    if not deterministic:
        return

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device_type = torch.device(getattr(cfg, "device", "cpu")).type
    if device_type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        raise RuntimeError(
            "Deterministic mode on CUDA requires `CUBLAS_WORKSPACE_CONFIG` to be set before "
            "launching Python (for example, `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8`)."
        )
    if os.environ.get("PYTHONHASHSEED") is None:
        log("[Determinism] Set `PYTHONHASHSEED=0` before launching Python for stable hashing.")


def entropy(a: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute the entropy of ``a`` along the last dimension."""
    a_safe = a + eps
    a_log = torch.log(a_safe)
    return -(a_safe * a_log).sum(dim = -1)


def unwrap_distributed_module(module: torch.nn.Module | None) -> torch.nn.Module | None:
    """Return the innermost module when wrapped by compile/DDP/DataParallel layers."""
    current = module
    while isinstance(current, torch.nn.Module):
        inner_orig = getattr(current, "_orig_mod", None)
        if isinstance(inner_orig, torch.nn.Module):
            current = inner_orig
            continue
        inner_module = getattr(current, "module", None)
        if isinstance(inner_module, torch.nn.Module):
            current = inner_module
            continue
        return current
    return current


def compute_commit_step_counts(
    avail_is_commit: torch.Tensor,
    action_taken: torch.Tensor,
    num_steps: torch.Tensor,
) -> torch.Tensor:
    """Count committed player-decision steps per trajectory.

    A time step is counted when the chosen action has ``is_commit=True``.
    Environment-controlled steps should expose ``is_commit=False`` for all
    available actions.
    """
    time = torch.arange(avail_is_commit.shape[1], device = avail_is_commit.device)
    active_mask = time.unsqueeze(0) < num_steps.unsqueeze(1)
    taken_commit = avail_is_commit.gather(dim = 2, index = action_taken.unsqueeze(-1)).squeeze(-1)
    return (taken_commit.to(torch.bool) & active_mask).sum(dim = 1)
