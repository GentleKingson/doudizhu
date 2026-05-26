"""Internal shared helpers for bootstrapping a MARVEL Sampler from semantic roots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

from marvel.env_batch import EnvStepBatch
from marvel.runtime.rollout_conversion import history_batch_to_rollout
from marvel.runtime.sampler import Sampler, SamplerConfig


def build_sampler(
    *,
    game: Any,
    policy_model: Any,
    rollout_temp: float,
    envs: Any,
    envs_step: Any,
    collect_batch_size: int = 0,
) -> Sampler:
    rollout_device = torch.device(getattr(policy_model, "_device"))
    parsed_step = EnvStepBatch.from_env_step(envs_step, SimpleNamespace(device = rollout_device))
    return Sampler(
        config = SamplerConfig(
            game = game,
            device = rollout_device,
            rollout_batch_size = int(parsed_step.batch_size[0]),
            collect_batch_size = int(collect_batch_size),
            sampling_temperature = float(rollout_temp),
            env_seed = 0,
        ),
        envs = envs,
        parsed_envs_step = parsed_step,
    )


def prefill_sampler_rollout_buffer_from_infoset_strs(
    *,
    game: Any,
    sampler: Sampler,
    infoset_strs: Sequence[str],
    pending_actions: Sequence[str],
) -> None:
    """Prefill the Sampler's rollout buffer from a batch of infoset/pending pairs.

    CRITICAL INVARIANT — callers must preserve this or search rollouts silently
    degrade by ~3 reward points (Hanabi self-play):

        After this call, ``sampler.rollout_buffer["num_steps"][i]`` equals the number
        of trajectory positions preceding ``sampler.parsed_envs_step[i]``. In other
        words, ``parsed_envs_step[i]`` represents the game state AT trajectory
        position ``num_steps[i]`` — the next slot the Sampler will write.

    Practical consequence: the Sampler's ``envs`` must be AT the state described by
    ``(infoset_strs[i], pending_actions[i])`` when this is called. If ``envs`` has
    been stepped past that state before handoff (e.g. via a pre-call to
    ``envs.step(root_actions)``), the buffer and env drift by one position and every
    subsequent policy query operates on a misaligned state. This was the root cause
    of the search-vs-eval 3-point calibration gap; see
    ``kagami/search/rollout_session.py::SearchRolloutSession.rollout_selected_action_rows``
    for the correct pattern (drive the root action via ``forced_action_indices`` on
    the first ``rollout_step`` instead of pre-stepping).
    """
    parse_infoset_strs = getattr(game, "parse_infoset_strs", None)
    if not callable(parse_infoset_strs):
        raise RuntimeError("Env-native rollout requires game.parse_infoset_strs(...).")

    infoset_rows = [str(item) for item in infoset_strs]
    pending_rows = [str(item) for item in pending_actions]
    if len(pending_rows) != len(infoset_rows):
        raise RuntimeError("pending_actions must match infoset_strs during rollout-buffer prefill.")

    history_batch = parse_infoset_strs(infoset_rows, pending_rows)
    rollout_prefix, num_steps_np, _player_ids_np, _is_commit_np, _avail_weights_np = history_batch_to_rollout(
        history_batch = history_batch,
        device = sampler.device,
        pad_token_id = int(game.PAD_TOKEN),
        prefix_mode = "ignore",
    )
    _write_rollout_prefix_excluding_current_state(
        sampler = sampler,
        rollout_prefix = rollout_prefix,
        num_steps_np = num_steps_np,
    )


def _write_rollout_prefix_excluding_current_state(
    *,
    sampler: Sampler,
    rollout_prefix: Any,
    num_steps_np: np.ndarray,
) -> None:
    batch_count = int(num_steps_np.shape[0])
    prefix_lengths_np = np.asarray(num_steps_np, dtype = np.int64) - 1
    if np.any(prefix_lengths_np < 0):
        raise RuntimeError("history_batch_to_rollout returned an empty current-state history during rollout-buffer prefill.")

    prefix_lengths_t = torch.as_tensor(prefix_lengths_np, dtype = torch.long, device = sampler.device)
    sampler.rollout_buffer["num_steps"][:batch_count] = prefix_lengths_t
    sampler.rollout_buffer["avail_is_commit"].zero_()

    for row_idx in range(batch_count):
        prefix_len = int(prefix_lengths_np[row_idx])
        if prefix_len <= 0:
            continue
        sampler.rollout_buffer["player_ids"][row_idx, :prefix_len] = rollout_prefix["player_ids"][row_idx, :prefix_len]
        sampler.rollout_buffer["obs_tokens"][row_idx, :prefix_len] = rollout_prefix["obs_tokens"][row_idx, :prefix_len]
        sampler.rollout_buffer["obs_channels"][row_idx, :prefix_len] = rollout_prefix["obs_channels"][row_idx, :prefix_len]
        sampler.rollout_buffer["avail_tokens"][row_idx, :prefix_len] = rollout_prefix["avail_tokens"][row_idx, :prefix_len]
        sampler.rollout_buffer["avail_weights"][row_idx, :prefix_len] = rollout_prefix["avail_weights"][row_idx, :prefix_len]
        chance_mask = sampler.rollout_buffer["player_ids"][row_idx, :prefix_len] < 0
        if bool(chance_mask.any().item()):
            sampler.rollout_buffer["avail_tokens"][row_idx, :prefix_len][chance_mask] = 0
            sampler.rollout_buffer["avail_weights"][row_idx, :prefix_len][chance_mask] = 0.0
            sampler.rollout_buffer["avail_weights"][row_idx, :prefix_len][chance_mask, 0] = 1.0


def prefill_sampler_rollout_buffer_from_history_strs(
    *,
    game: Any,
    sampler: Sampler,
    history_strs: Sequence[str],
    pending_action: str,
) -> None:
    history_rows = [str(item) for item in history_strs]
    pending = str(pending_action)
    parse_history_strs = getattr(game, "parse_history_strs", None)
    if callable(parse_history_strs) and not pending:
        history_batch = parse_history_strs(history_rows)
        rollout_prefix, num_steps_np, _player_ids_np, _is_commit_np, _avail_weights_np = history_batch_to_rollout(
            history_batch = history_batch,
            device = sampler.device,
            pad_token_id = int(game.PAD_TOKEN),
            prefix_mode = "from_action_taken",
        )
        _write_rollout_prefix_excluding_current_state(
            sampler = sampler,
            rollout_prefix = rollout_prefix,
            num_steps_np = num_steps_np,
        )
        return

    infoset_str_from_history_strs = getattr(game, "infoset_str_from_history_strs", None)
    if not callable(infoset_str_from_history_strs):
        raise RuntimeError(
            "Env-native rollout requires either game.parse_history_strs(...) for concrete history prefill "
            "or game.infoset_str_from_history_strs(...) for infoset-derived prefill."
        )

    infoset_rows = [str(item) for item in infoset_str_from_history_strs(history_rows)]
    if len(infoset_rows) != len(history_rows):
        raise RuntimeError("infoset_str_from_history_strs returned an unexpected row count during rollout-buffer prefill.")
    prefill_sampler_rollout_buffer_from_infoset_strs(
        game = game,
        sampler = sampler,
        infoset_strs = infoset_rows,
        pending_actions = [pending] * len(infoset_rows),
    )


def validate_prefilled_rollout_timeline(*, sampler: Sampler, next_step: Any, max_horizon: int) -> None:
    batch_size = int(sampler.parsed_envs_step.batch_size[0])
    num_steps_t = sampler.rollout_buffer["num_steps"][:batch_size].detach().to(
        dtype = torch.long,
        device = torch.device("cpu"),
    )
    if int(num_steps_t.ndim) != 1 or int(num_steps_t.shape[0]) != batch_size:
        raise RuntimeError(
            "rollout_buffer.num_steps shape mismatch during rollout timeline validation: "
            f"expected {(batch_size,)}, got {tuple(num_steps_t.shape)}."
        )

    overflow = num_steps_t > int(max_horizon)
    if bool(overflow.any().item()):
        row_idx = int(torch.nonzero(overflow, as_tuple = False)[0].item())
        raise RuntimeError(
            "Prefilled rollout buffer exceeds MAX_GAME_HORIZON before env-native rollout: "
            f"row={row_idx} num_steps={int(num_steps_t[row_idx].item())} horizon={int(max_horizon)}."
        )

    terminal_np = np.asarray(next_step.is_terminal, dtype = bool)
    if tuple(terminal_np.shape) != (batch_size,):
        raise RuntimeError(
            "reconstruct_envs/env.step returned invalid is_terminal shape during rollout timeline validation: "
            f"expected {(batch_size,)}, got {tuple(terminal_np.shape)}."
        )
    at_horizon = np.asarray(num_steps_t.numpy() == int(max_horizon), dtype = bool)
    invalid_horizon = np.logical_and(at_horizon, np.logical_not(terminal_np))
    if bool(np.any(invalid_horizon)):
        row_idx = int(np.nonzero(invalid_horizon)[0][0])
        raise RuntimeError(
            "Encountered non-terminal env-native rollout state at MAX_GAME_HORIZON; refusing out-of-bounds step write: "
            f"row={row_idx} num_steps={int(num_steps_t[row_idx].item())} horizon={int(max_horizon)}."
        )


__all__ = [
    "build_sampler",
    "prefill_sampler_rollout_buffer_from_history_strs",
    "prefill_sampler_rollout_buffer_from_infoset_strs",
    "validate_prefilled_rollout_timeline",
]
