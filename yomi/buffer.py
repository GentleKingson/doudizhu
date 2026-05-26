from __future__ import annotations

import torch
from marvel.buffer import BufferConfig, FieldSpecLike, TrajectoryBuffer
from marvel.model.tensordict import TensorDict
from marvel.utils import get_game_attr


def belief_seq_field_specs(game: object) -> dict[str, FieldSpecLike]:
    """Return the field specs for a belief-sequence buffer."""
    T = int(get_game_attr(game, "max_game_horizon"))
    L_obs = int(get_game_attr(game, "max_observation_tokens"))
    C = int(get_game_attr(game, "num_channels"))
    S = int(get_game_attr(game, "max_belief_rows"))
    E = int(get_game_attr(game, "max_belief_tokens"))
    A = int(get_game_attr(game, "max_belief_candidates"))
    pad = int(get_game_attr(game, "pad_token"))
    return {
        "obs_prefix_tokens": lambda B, device: torch.full(
            [B, T, L_obs], fill_value = pad, dtype = torch.long, device = device
        ),
        "obs_prefix_mask": lambda B, device: torch.zeros([B, T], dtype = torch.long, device = device),
        "state_channels": lambda B, device: torch.zeros([B, C], dtype = torch.float32, device = device),
        "belief_input_tokens": lambda B, device: torch.full(
            [B, S, E], fill_value = pad, dtype = torch.long, device = device
        ),
        "candidate_tokens": lambda B, device: torch.full(
            [B, S, A, E], fill_value = pad, dtype = torch.long, device = device
        ),
        "candidate_valid": lambda B, device: torch.zeros([B, S, A], dtype = torch.long, device = device),
        "belief_target_index": lambda B, device: torch.full([B, S], fill_value = -1, dtype = torch.long, device = device),
        "belief_length": lambda B, device: torch.zeros([B], dtype = torch.long, device = device),
        "belief_target_tokens": lambda B, device: torch.full(
            [B, S, E], fill_value = pad, dtype = torch.long, device = device
        ),
        "belief_target_mask": lambda B, device: torch.zeros([B, S], dtype = torch.long, device = device),
        "reward_to_go": lambda B, device: torch.zeros([B], dtype = torch.float32, device = device),
        "reward_to_go_mask": lambda B, device: torch.zeros([B], dtype = torch.long, device = device),
        "player_ids": lambda B, device: torch.full([B], fill_value = -1, dtype = torch.long, device = device),
    }


class BeliefSeqBuffer(TrajectoryBuffer):
    """TensorDict-backed queue for belief-sequence samples."""

    _REQUIRED_KEYS = (
        "obs_prefix_tokens",
        "obs_prefix_mask",
        "state_channels",
        "belief_input_tokens",
        "candidate_tokens",
        "candidate_valid",
        "belief_target_index",
        "belief_length",
        "belief_target_tokens",
        "belief_target_mask",
        "reward_to_go",
        "reward_to_go_mask",
        "player_ids",
    )

    def __init__(self, batch_size: int, config: BufferConfig):
        super().__init__(batch_size, config)
        self.capacity = int(self.batch_size)
        self.size = 0

    def append(self, chunk: TensorDict) -> int:
        missing = [key for key in self._REQUIRED_KEYS if key not in chunk.keys()]
        if missing:
            raise KeyError(f"BeliefSeqBuffer append missing required keys: {missing}")
        if len(chunk.batch_size) == 0:
            raise ValueError("BeliefSeqBuffer append expects a batched TensorDict.")

        chunk = chunk.to(self.device)
        chunk_size = int(chunk.batch_size[0])
        if chunk_size <= 0 or self.size >= self.capacity:
            return 0

        copy_n = min(self.capacity - self.size, chunk_size)
        start = int(self.size)
        end = int(start + copy_n)
        self._storage[start:end] = chunk[:copy_n].clone()
        self.size = end
        return copy_n

    def consume(self, count: int) -> TensorDict:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}.")
        if count > self.size:
            raise ValueError("Cannot consume more rows than currently stored.")

        consumed = self._storage[:count].clone()
        if count == 0:
            return consumed

        remaining = int(self.size - count)
        if remaining > 0:
            self._storage[:remaining] = self._storage[count:self.size].clone()
        self.size = remaining
        return consumed
