from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from marvel.config import OptimizerConfig, TrainConfig as MarvelTrainConfig
from marvel.game_identity import (
    resolve_nanopokerlib_game_identifier,
    resolve_nanopokerlib_game_module,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MARVEL_TRAIN_DEFAULTS = MarvelTrainConfig()


@dataclass
class ModelConfig:
    architecture: str = "v1.3"
    # Match MARVEL default decoder sizing.
    decoder_layers: int = 4
    decoder_heads: int = 4
    history_dim: int = 256
    # Number of layers in the belief output-row encoder MLP.
    # For v1.1: must satisfy 2 + 4k structured convention.
    # For v1.2: counts only the gated blocks.
    # Must be positive.
    belief_layers: int = 2
    enable_reward_to_go_head: bool = True


@dataclass
class TrainConfig:
    project_name: str = "yomi"
    run_name: str | None = None
    game_name: str = "doudizhu"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    resume_from: str | None = None
    resume_optimizer: bool = True

    policy_checkpoint_path: str = ""

    total_iterations: int = 40000
    checkpoint_interval: int = 5000
    validation_interval: int = 100
    param_stats_interval: int = 1
    rollout_batch_size: int = 128
    collect_interval: int | None = None
    collect_warmup_steps: int | None = None

    rollout_temperature: float = 0.0

    base_lr: float = _MARVEL_TRAIN_DEFAULTS.ppo.base_lr
    reward_to_go_loss_weight: float = 1.0

    lr_stable_fraction: float = 0.9

    optimizer: OptimizerConfig = field(default_factory = OptimizerConfig)

    use_compile: bool = True
    compile_backend: str = "inductor"
    compile_dynamic_shapes: bool = True

    logging_dir: str = ""
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

    @property
    def game(self) -> Any:
        return resolve_nanopokerlib_game_module(self.game_name)


def deserialize_train_config(payload: Any, *, strict: bool = False) -> TrainConfig | None:
    """Reconstruct a Yomi TrainConfig from either a mapping or an existing instance."""
    if payload is None:
        return None
    if isinstance(payload, TrainConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported yomi train_config payload type: {type(payload)}")

    def _canonical_key(key: object) -> str:
        return "".join(ch for ch in str(key).lower() if ch.isalnum())

    def _filter_payload_keys(
        data: Mapping[str, Any],
        *,
        allowed: set[str],
        where: str,
        aliases: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        aliases = aliases or {}
        canonical_to_allowed: dict[str, list[str]] = {}
        for name in allowed:
            canonical_to_allowed.setdefault(_canonical_key(name), []).append(name)

        filtered: dict[str, Any] = {}
        unknown: list[str] = []
        for raw_key, value in data.items():
            key = str(raw_key)
            resolved_key = aliases.get(key, key)
            if resolved_key in allowed:
                if resolved_key not in filtered or resolved_key == key:
                    filtered[resolved_key] = value
                continue

            candidates = canonical_to_allowed.get(_canonical_key(resolved_key), [])
            if len(candidates) == 1:
                filtered.setdefault(candidates[0], value)
                continue
            unknown.append(key)

        if unknown and strict:
            raise ValueError(f"Unexpected keys for {where}: {sorted(unknown)}")
        return filtered

    allowed_keys = set(TrainConfig.__dataclass_fields__.keys())
    cfg_payload = _filter_payload_keys(payload, allowed = allowed_keys, where = "yomi.TrainConfig")

    if "model" in cfg_payload and isinstance(cfg_payload["model"], Mapping):
        model_allowed = {field_info.name for field_info in ModelConfig.__dataclass_fields__.values()}
        model_payload = _filter_payload_keys(
            cfg_payload["model"],
            allowed = model_allowed,
            where = "yomi.ModelConfig",
            aliases = {
                "history_embd": "history_dim",
                "decoder_n_layers": "decoder_layers",
                "decoder_n_heads": "decoder_heads",
                "n_belief_layers": "belief_layers",
            },
        )
        model_payload.setdefault("enable_reward_to_go_head", False)
        cfg_payload["model"] = ModelConfig(**model_payload)

    if "optimizer" in cfg_payload and isinstance(cfg_payload["optimizer"], Mapping):
        optimizer_allowed = {f.name for f in OptimizerConfig.__dataclass_fields__.values()}
        optimizer_payload = _filter_payload_keys(
            cfg_payload["optimizer"],
            allowed = optimizer_allowed,
            where = "OptimizerConfig",
        )
        cfg_payload["optimizer"] = OptimizerConfig(**optimizer_payload)

    return TrainConfig(**cfg_payload)
