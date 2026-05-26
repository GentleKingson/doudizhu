from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch
from marvel.env_batch import EnvOutcome, EnvStepBatch
from marvel.model.tensordict import TensorDict

from marvel.utils import get_game_attr

FieldFactory = Callable[[int, torch.device], torch.Tensor]
FieldSpecLike = Union[FieldFactory, "BufferFieldSpec", torch.Tensor]

__all__ = [
    "BufferConfig",
    "EnvOutcome",
    "EnvStepBatch",
    "FieldSpecLike",
    "BufferFieldSpec",
    "TrajectoryBuffer",
    "trajectory_field_specs",
]


def trajectory_field_specs(game: object) -> dict[str, FieldSpecLike]:
    """Return the standard trajectory field specs for a game."""
    T = int(get_game_attr(game, "max_game_horizon"))
    A = int(get_game_attr(game, "max_available_actions"))
    P = int(get_game_attr(game, "num_players"))
    L_obs = int(get_game_attr(game, "max_observation_tokens"))
    L_act = int(get_game_attr(game, "max_action_tokens"))
    C = int(get_game_attr(game, "num_channels"))
    return {
        "obs_tokens": lambda B, device: torch.zeros(
            [B, T, P, L_obs], dtype = torch.long, device = device
        ),
        "obs_channels": lambda B, device: torch.zeros(
            [B, T, P, C], dtype = torch.uint8, device = device
        ),
        "player_ids": lambda B, device: torch.zeros([B, T], dtype = torch.long, device = device),
        "avail_tokens": lambda B, device: torch.zeros(
            [B, T, A, L_act], dtype = torch.long, device = device
        ),
        "avail_weights": lambda B, device: torch.zeros([B, T, A], device = device),
        "avail_is_commit": lambda B, device: torch.zeros([B, T, A], dtype = torch.bool, device = device),
        "avail_probs": lambda B, device: torch.zeros([B, T, A], device = device),
        "action_taken": lambda B, device: torch.zeros([B, T], dtype = torch.long, device = device),
        "rewards": lambda B, device: torch.zeros([B, T, P], device = device),
        "is_terminal": lambda B, device: torch.zeros([B, T], dtype = torch.bool, device = device),
    }


@dataclass(frozen = True)
class BufferConfig:
    """Configuration describing how buffers should interpret game layouts."""

    game: Any
    device: torch.device | str
    field_specs: Mapping[str, FieldSpecLike] = dataclasses.field(default_factory = dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(self, "field_specs", dict(self.field_specs))


@dataclass(frozen = True)
class BufferFieldSpec:
    """Specification for optional buffer fields."""

    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    fill: float | int | bool = 0.0

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))


class TrajectoryBuffer:
    """TensorDict-backed buffer managing trajectory storage with optional cyclic replay.

    When used as a rollout accumulator (sampler path), use ``add_step()`` and ``reset()``.
    When used as a replay buffer (trainer path), use ``add()`` to cyclically insert whole
    trajectories and ``sample()`` to draw uniform random batches.
    """

    def __init__(self, batch_size: int, config: BufferConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.batch_size = int(batch_size)

        self._field_factories: Dict[str, FieldFactory] = {
            "num_steps": lambda B, device: torch.zeros([B], dtype = torch.long, device = device),
        }
        for name, spec in config.field_specs.items():
            self._field_factories[name] = self._make_field_factory(spec)

        self._storage = self._allocate(self.batch_size, self.device)

        # Cyclic replay state — always present, inert unless add()/sample() are called.
        self._ptr = 0
        self._size = 0

    def _make_field_factory(self, spec: FieldSpecLike) -> FieldFactory:
        if callable(spec):
            return spec  # type: ignore[return-value]
        if isinstance(spec, BufferFieldSpec):
            shape = tuple(spec.shape)

            def factory(batch: int, device: torch.device, *, _spec = spec, _shape = shape):
                full_shape = (batch,) + _shape
                return torch.full(
                    full_shape,
                    _spec.fill,
                    dtype = _spec.dtype,
                    device = device,
                )

            return factory
        if isinstance(spec, torch.Tensor):
            base = spec.detach().clone()

            def factory(batch: int, device: torch.device, *, _base = base):
                tensor = _base.to(device)
                if tensor.dim() == 0:
                    return tensor.repeat(batch).clone()
                expanded = tensor.unsqueeze(0).expand((batch,) + tuple(tensor.shape))
                return expanded.clone()

            return factory
        raise TypeError(
            "Field specification must be a callable, BufferFieldSpec, or torch.Tensor prototype."
        )

    def _allocate(self, batch_size: int, device: torch.device) -> TensorDict:
        fields = {
            name: factory(batch_size, device)
            for name, factory in self._field_factories.items()
        }
        return TensorDict(fields, batch_size = [batch_size], device = device)

    # ------------------------------------------------------------------
    # Common accessors
    # ------------------------------------------------------------------

    def __getitem__(self, item):
        return self._storage[item]

    def __setitem__(self, key, value) -> None:
        self._storage[key] = value

    def __len__(self) -> int:
        return self._size

    @property
    def storage(self) -> TensorDict:
        return self._storage

    # ------------------------------------------------------------------
    # Step-level accumulation (sampler path)
    # ------------------------------------------------------------------

    def add_step(self, batch_indices: torch.Tensor, transition: TensorDict) -> None:
        """Commit a batch of transitions at the current per-environment step index."""
        if transition.batch_size[0] != batch_indices.shape[0]:
            raise ValueError(
                "Transition batch size must match number of batch indices. "
                f"Got transition batch {transition.batch_size[0]} and indices {batch_indices.shape[0]}."
            )

        transition = transition.to(self.device)
        time_indices = self._storage["num_steps"][batch_indices]

        storage_keys = set(self._storage.keys())
        for key, value in transition.items():
            if key not in storage_keys:
                raise KeyError(
                    f"Field '{key}' is not registered in TrajectoryBuffer. "
                    "Include it in BufferConfig.field_specs before constructing the buffer."
                )
            target = self._storage.get(key)
            target[batch_indices, time_indices] = value
            self._storage.set_(key, target)

        self._storage["num_steps"][batch_indices] = time_indices + 1

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        """Reset the whole buffer or a subset identified by a boolean mask."""
        if mask is None:
            self._storage = self._allocate(self.batch_size, self.device)
            return

        mask = mask.to(device = self.device, dtype = torch.bool)
        if mask.numel() != self.batch_size:
            raise ValueError("Mask size must match buffer batch size.")
        num_resets = int(mask.sum().item())
        if num_resets == 0:
            return
        replacement = self._allocate(num_resets, self.device)
        self._storage[mask] = replacement

    def empty(self, *, batch_size: Optional[int] = None, device: Optional[torch.device] = None) -> TensorDict:
        """Allocate a fresh TensorDict with the registered fields."""
        resolved_batch = self.batch_size if batch_size is None else int(batch_size)
        resolved_device = self.device if device is None else torch.device(device)
        return self._allocate(resolved_batch, resolved_device)

    # ------------------------------------------------------------------
    # Cyclic replay (trainer path)
    # ------------------------------------------------------------------

    def add(self, batch: TensorDict) -> None:
        """Insert a batch of complete trajectories cyclically, overwriting the oldest."""
        if batch.batch_size[0] > self.batch_size:
            raise ValueError("Cannot add more trajectories than the buffer capacity in a single call.")

        trajectories = batch.to(self.device)
        count = trajectories.batch_size[0]

        indices = (
            torch.arange(self._ptr, self._ptr + count, device = self.device, dtype = torch.long)
            % self.batch_size
        )

        self._storage[indices] = trajectories

        self._ptr = (self._ptr + count) % self.batch_size
        self._size = min(self.batch_size, self._size + count)

    def sample(
        self,
        batch_size: int,
        *,
        keys: tuple[str, ...] | None = None,
    ) -> TensorDict:
        """Sample trajectories uniformly at random without replacement."""
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if batch_size > self._size:
            raise ValueError("Cannot sample more items than are stored.")

        indices = torch.randperm(self._size, device = self.device)[:batch_size]

        source = self._storage
        if keys is not None:
            source = source.select(*keys)
        return source[indices]
