from __future__ import annotations

try:
    from functools import cached_property
except ImportError:  # Python 3.7
    class cached_property:  # type: ignore[no-redef]
        def __init__(self, func):
            self.func = func
            self.__doc__ = getattr(func, "__doc__")
            self.attrname = getattr(func, "__name__", "cached_property")

        def __get__(self, obj, objtype = None):
            if obj is None:
                return self
            value = self.func(obj)
            setattr(obj, self.attrname, value)
            return value

import time

import torch
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from marvel.game_identity import (
    resolve_nanopokerlib_game_identifier,
    resolve_nanopokerlib_game_module,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()
_SUPPORTED_MODEL_ARCHITECTURES = frozenset({"v1", "v1.1", "v1.2", "v1.3"})
_LEGACY_MODEL_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "history_dim": ("history_embd",),
    "feature_dim": ("feature_embd",),
    "decoder_layers": ("decoder_n_layers",),
    "decoder_heads": ("decoder_n_heads",),
    "state_layers": ("n_state_layers",),
    "action_layers": ("n_action_layers",),
    "fuse_layers": ("n_fuse_layers",),
}

@dataclass
class PPOConfig:
    base_lr: float = 4.0e-4
    agent_epoch: int = 4
    gae_lambda: float = 0.95
    base_clip_coef: float = 0.02
    base_reg_coef: float = 0.1


@dataclass
class VRPOConfig:
    base_lr: float = 4.0e-4
    gae_lambda: float = 0.95
    aux_epoch: int = 4
    replay_buffer_device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ScheduleConfig:
    agent_lr_decay_power: float = 1.0
    critic_lr_decay_power: float = 0.5
    clip_coef_decay_power: float = 1.0
    reg_coef_decay_power: float = 0.5


@dataclass
class OptimizerConfig:
    use_muon: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    muon_beta1: float = 0.95
    weight_decay: float = 1e-2
    grad_max_norm: float = 1.0


@dataclass
class ModelConfig:
    architecture: str = "v1.3"
    history_dim: int = 128
    feature_dim: int = 256
    decoder_layers: int = 4
    decoder_heads: int = 4
    state_layers: int = 3
    action_layers: int = 2
    fuse_layers: int = 2


def _canonical_key(key: object) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _lookup_mapping_value(
    data: Mapping[str, Any],
    *,
    field_name: str,
    aliases: tuple[str, ...] = (),
) -> Any:
    candidate_keys = {_canonical_key(field_name)}
    candidate_keys.update(_canonical_key(alias) for alias in aliases)
    for raw_key, value in data.items():
        if _canonical_key(raw_key) in candidate_keys:
            return value
    return _MISSING


def _deserialize_checkpoint_model_config(payload: Any) -> ModelConfig:
    if payload is None:
        return ModelConfig(architecture = "v1")
    if isinstance(payload, ModelConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported train_config.model payload type: {type(payload)}")

    model_kwargs: dict[str, Any] = {}
    for model_field in fields(ModelConfig):
        field_value = _lookup_mapping_value(
            payload,
            field_name = model_field.name,
            aliases = _LEGACY_MODEL_FIELD_ALIASES.get(model_field.name, ()),
        )
        if field_value is not _MISSING:
            model_kwargs[model_field.name] = field_value

    architecture = str(model_kwargs.get("architecture", "v1"))
    if architecture not in _SUPPORTED_MODEL_ARCHITECTURES:
        raise ValueError(
            "Unsupported model architecture "
            f"'{architecture}'. Expected one of: v1, v1.1, v1.2, v1.3."
        )
    model_kwargs["architecture"] = architecture
    return ModelConfig(**model_kwargs)


@dataclass
class TrainConfig:
    """Hyperparameters used for training."""

    project_name: str = "marvel"
    run_name: str | None = None
    algo_name: str = "vrpo"
    game_name: str = "doudizhu"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logging_dir: str = ""
    debug: bool = True

    seed: int = 0
    deterministic: bool = False
    deterministic_disable_tf32: bool = False

    use_compile: bool = True
    compile_backend: str = "inductor"

    benchmark: bool = False
    benchmark_contiguous: bool = False
    profile: bool = False
    profile_trace_dir: str | None = None
    profile_warmup_iterations: int = 3

    resume_from: str | None = None
    resume_optimizer: bool = True
    full_state_checkpoints: bool = False
    checkpoint_wallclock_seconds: float = 900.0

    total_iterations: int = 2000
    checkpoint_interval: int = 5000
    evaluation_interval: int = 0
    evaluation_routines: tuple[str, ...] | None = None
    param_stats_interval: int = 1

    rollout_batch_size: int = 512
    collect_batch_size: int = 2048
    replay_buffer_scale: int = 64
    mini_batches: int = 4

    lr_stable_iterations: int = 500
    reg_stable_iterations: int = 500
    discount_gamma: float = 1.0
    agent_merging_beta: float = 0.0
    value_loss_coef: float = 0.5

    optimizer: OptimizerConfig = field(default_factory = OptimizerConfig)
    ppo: PPOConfig = field(default_factory = PPOConfig)
    vrpo: VRPOConfig = field(default_factory = VRPOConfig)
    schedule: ScheduleConfig = field(default_factory = ScheduleConfig)
    model: ModelConfig = field(default_factory = ModelConfig)

    def __post_init__(self) -> None:
        if not self.run_name:
            timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
            self.run_name = f"{self.game_identifier}_{timestamp}"
        if not self.logging_dir:
            self.logging_dir = str(_REPO_ROOT / "runs" / self.run_name)

    @property
    def game_identifier(self) -> str:
        return resolve_nanopokerlib_game_identifier(self.game_name)

    @cached_property
    def game(self) -> Any:
        return resolve_nanopokerlib_game_module(self.game_name)


def deserialize_train_config(payload: Any) -> TrainConfig | None:
    """Load the inference-required checkpoint metadata into a TrainConfig."""
    if payload is None:
        return None
    if isinstance(payload, TrainConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported train_config payload type: {type(payload)}")

    game_name = _lookup_mapping_value(payload, field_name = "game_name")
    if game_name is _MISSING:
        raise KeyError("Checkpoint train_config is missing required inference field 'game_name'.")

    model_payload = _lookup_mapping_value(payload, field_name = "model")
    optimizer_payload = _lookup_mapping_value(payload, field_name = "optimizer")
    kwargs: dict[str, Any] = {
        "game_name": str(game_name),
        "model": _deserialize_checkpoint_model_config(
            None if model_payload is _MISSING else model_payload
        ),
    }
    if optimizer_payload is not _MISSING:
        if isinstance(optimizer_payload, OptimizerConfig):
            kwargs["optimizer"] = optimizer_payload
        elif isinstance(optimizer_payload, Mapping):
            optimizer_fields = {f.name for f in fields(OptimizerConfig)}
            kwargs["optimizer"] = OptimizerConfig(
                **{k: v for k, v in optimizer_payload.items() if k in optimizer_fields}
            )
    return TrainConfig(**kwargs)
