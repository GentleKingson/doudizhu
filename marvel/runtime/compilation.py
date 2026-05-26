from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable

import torch
import torch.nn as nn

from marvel.config import TrainConfig


LogFn = Callable[[str], None]


@dataclass(frozen = True)
class TorchCompileSettings:
    enabled: bool
    backend: str
    dynamic_shapes: bool

    @classmethod
    def from_cfg(cls, cfg: TrainConfig) -> TorchCompileSettings:
        enabled = bool(getattr(cfg, "use_compile", False))
        backend = str(getattr(cfg, "compile_backend", "inductor") or "inductor")
        dynamic_shapes = bool(getattr(cfg, "compile_dynamic_shapes", False))
        return cls(
            enabled = enabled,
            backend = backend,
            dynamic_shapes = dynamic_shapes,
        )


class ModuleCompiler:
    """Compile a set of rollout modules based on TrainConfig settings."""

    def __init__(
        self,
        settings: TorchCompileSettings,
        *,
        log: LogFn | None = None,
    ) -> None:
        self.settings = settings
        self._log = log
        self._logged = False

    @classmethod
    def from_cfg(
        cls,
        cfg: TrainConfig,
        *,
        log: LogFn | None = None,
    ) -> ModuleCompiler:
        return cls(TorchCompileSettings.from_cfg(cfg), log = log)

    def compile(self, module: nn.Module) -> nn.Module:
        if not self.settings.enabled:
            return module

        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile unavailable; disable --use-compile or install torch>=2.0.")

        compile_kwargs = {"backend": self.settings.backend}
        if "dynamic" in inspect.signature(torch.compile).parameters and not self.settings.dynamic_shapes:
            compile_kwargs["dynamic"] = False
        compiled = torch.compile(module, **compile_kwargs)
        self._log_enabled_once()
        return compiled

    def _log_enabled_once(self) -> None:
        if self._log is None or self._logged:
            return

        self._log(
            "[Compile] Enabled torch.compile "
            f"(backend={self.settings.backend}, dynamic_batch_dims={self.settings.dynamic_shapes})."
        )
        self._logged = True
