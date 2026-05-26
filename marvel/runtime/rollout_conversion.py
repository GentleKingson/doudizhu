from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from marvel.model.tensordict import TensorDict

HistoryPrefixMode = Literal["ignore", "from_action_taken"]


def history_batch_to_rollout(
    *,
    history_batch: Any,
    device: torch.device,
    pad_token_id: int,
    prefix_mode: HistoryPrefixMode = "ignore",
    target_batch_size: int | None = None,
    target_time_dim: int | None = None,
) -> tuple[TensorDict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a model-ready rollout TensorDict from endpoint HistoryBatch-like data."""
    num_steps_np = np.asarray(history_batch.num_steps, dtype = np.int64)
    action_taken_np = np.asarray(history_batch.action_taken_tokens, dtype = np.int64)
    player_ids_hist_np = np.asarray(history_batch.player_ids, dtype = np.int64)
    last_avail = history_batch.last_available_actions
    last_avail_tokens_np = np.asarray(last_avail.tokens, dtype = np.int64)
    last_avail_weights_np = np.asarray(last_avail.weights, dtype = np.float32)
    last_avail_is_commit_np = np.asarray(last_avail.is_commit, dtype = np.uint8)
    last_avail_player_ids_np = np.asarray(last_avail.player_ids, dtype = np.int64)

    if action_taken_np.ndim != 3:
        raise ValueError("HistoryBatch.action_taken_tokens must have shape [B, T, MAX_ACTION_TOKENS].")
    if last_avail_tokens_np.ndim != 3:
        raise ValueError("HistoryBatch.last_available_actions.tokens must have shape [B, A, MAX_ACTION_TOKENS].")
    if last_avail_weights_np.ndim != 2:
        raise ValueError("HistoryBatch.last_available_actions.weights must have shape [B, A].")
    if last_avail_is_commit_np.ndim != 2:
        raise ValueError("HistoryBatch.last_available_actions.is_commit must have shape [B, A].")

    obs = history_batch.observations
    obs_tokens_np = np.asarray(getattr(obs, "tokens", obs), dtype = np.int64)
    obs_channels_raw = getattr(obs, "channels", None)
    if obs_channels_raw is None:
        obs_channels_np = np.zeros(obs_tokens_np.shape[:3] + (0,), dtype = np.int64)
    else:
        obs_channels_np = np.asarray(obs_channels_raw, dtype = np.int64)

    if obs_tokens_np.ndim != 4:
        raise ValueError("HistoryBatch.observations.tokens must have shape [B, T, P, L_obs].")
    if obs_channels_np.ndim != 4:
        raise ValueError("HistoryBatch.observations.channels must have shape [B, T, P, C].")

    batch, time_dim_hist, num_players, obs_width = obs_tokens_np.shape
    if player_ids_hist_np.ndim != 2:
        raise ValueError("HistoryBatch.player_ids must have shape [B, T].")
    if obs_channels_np.shape[:3] != (batch, time_dim_hist, num_players):
        raise ValueError("HistoryBatch observations.tokens/channels leading dimensions are misaligned.")
    if action_taken_np.shape[0] != batch or action_taken_np.shape[1] != time_dim_hist:
        raise ValueError("HistoryBatch.action_taken_tokens must align with observations along [B, T].")
    if int(num_steps_np.shape[0]) != batch:
        raise ValueError("HistoryBatch.num_steps must have shape [B].")
    if int(last_avail_tokens_np.shape[0]) != batch:
        raise ValueError("HistoryBatch.last_available_actions.tokens must align with batch size [B].")
    if int(last_avail_player_ids_np.shape[0]) != batch:
        raise ValueError("HistoryBatch.last_available_actions.player_ids must have shape [B].")
    if player_ids_hist_np.shape != (batch, time_dim_hist):
        raise ValueError("HistoryBatch.player_ids must align with observations along [B, T].")

    max_actions = int(last_avail_tokens_np.shape[1])
    max_action_tokens = int(last_avail_tokens_np.shape[2])
    if action_taken_np.shape[2] != max_action_tokens:
        raise ValueError(
            "HistoryBatch.action_taken_tokens trailing action-token dim must match "
            "HistoryBatch.last_available_actions.tokens."
        )
    if last_avail_weights_np.shape != (batch, max_actions):
        raise ValueError("HistoryBatch.last_available_actions.weights must align with [B, A].")
    if last_avail_is_commit_np.shape != (batch, max_actions):
        raise ValueError("HistoryBatch.last_available_actions.is_commit must align with [B, A].")

    if np.any(num_steps_np <= 0):
        raise ValueError("HistoryBatch.num_steps must be positive.")
    if np.any(num_steps_np > time_dim_hist):
        raise ValueError("HistoryBatch.num_steps contains out-of-range timesteps.")

    batch_size_out = int(batch if target_batch_size is None else target_batch_size)
    if batch_size_out < batch:
        raise ValueError("target_batch_size cannot be smaller than the input HistoryBatch batch size.")

    time_dim_out = int(time_dim_hist if target_time_dim is None else target_time_dim)
    if time_dim_out < time_dim_hist:
        raise ValueError("target_time_dim cannot be smaller than the input HistoryBatch time dimension.")

    num_channels = int(obs_channels_np.shape[-1])
    obs_tokens_out = np.full(
        (batch_size_out, time_dim_out, num_players, obs_width),
        int(pad_token_id),
        dtype = np.int64,
    )
    obs_channels_out = np.zeros(
        (batch_size_out, time_dim_out, num_players, num_channels),
        dtype = np.int64,
    )
    avail_tokens_out = np.full(
        (batch_size_out, time_dim_out, max_actions, max_action_tokens),
        int(pad_token_id),
        dtype = np.int64,
    )
    avail_weights_out = np.zeros((batch_size_out, time_dim_out, max_actions), dtype = np.float32)
    player_ids_out = np.full((batch_size_out, time_dim_out), -1, dtype = np.int64)
    num_steps_out = np.zeros((batch_size_out,), dtype = np.int64)
    is_commit_out = np.zeros((batch_size_out, max_actions), dtype = np.uint8)

    obs_tokens_out[:batch, :time_dim_hist] = obs_tokens_np
    obs_channels_out[:batch, :time_dim_hist] = obs_channels_np
    num_steps_out[:batch] = num_steps_np
    is_commit_out[:batch] = last_avail_is_commit_np
    player_ids_out[:batch, :time_dim_hist] = player_ids_hist_np

    last_steps = num_steps_np - 1
    batch_idx = np.arange(batch, dtype = np.int64)
    if not np.array_equal(player_ids_hist_np[batch_idx, last_steps], last_avail_player_ids_np):
        raise ValueError("HistoryBatch.player_ids final step must match HistoryBatch.last_available_actions.player_ids.")
    avail_tokens_out[batch_idx, last_steps, :, :] = last_avail_tokens_np
    avail_weights_out[batch_idx, last_steps, :] = last_avail_weights_np
    player_ids_out[batch_idx, last_steps] = last_avail_player_ids_np

    if prefix_mode == "from_action_taken":
        if batch > 0 and time_dim_hist > 0:
            time_range = np.arange(time_dim_hist)
            prefix_mask = time_range[np.newaxis, :] < last_steps[:batch, np.newaxis]
            not_pad = np.any(action_taken_np[:batch] != int(pad_token_id), axis = 2)
            valid = prefix_mask & not_pad
            b_idx, t_idx = np.where(valid)
            if b_idx.size > 0:
                avail_tokens_out[b_idx, t_idx, 0, :] = action_taken_np[b_idx, t_idx]
                avail_weights_out[b_idx, t_idx, 0] = 1.0
    elif prefix_mode != "ignore":
        raise ValueError(f"Unsupported prefix_mode: {prefix_mode!r}")

    num_steps_tensor = torch.from_numpy(num_steps_out).to(device = device)
    time_mask = torch.arange(time_dim_out, device = device).unsqueeze(0) < num_steps_tensor.unsqueeze(1)

    rollout = TensorDict(
        {
            "num_steps": num_steps_tensor,
            "player_ids": torch.from_numpy(player_ids_out).to(device = device),
            "obs_tokens": torch.from_numpy(obs_tokens_out).to(device = device),
            "obs_channels": torch.from_numpy(obs_channels_out).to(device = device),
            "avail_tokens": torch.from_numpy(avail_tokens_out).to(device = device),
            "avail_weights": torch.from_numpy(avail_weights_out).to(device = device)
            * time_mask.unsqueeze(-1).to(dtype = torch.float32),
        },
        batch_size = [batch_size_out],
        device = device,
    )
    return rollout, num_steps_out, player_ids_out, is_commit_out, avail_weights_out


__all__ = ["history_batch_to_rollout"]
