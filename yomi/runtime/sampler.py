from __future__ import annotations

from typing import Any

import numpy as np
import torch

from marvel.env_batch import EnvStepBatch
from marvel.model.tensordict import TensorDict
from marvel.runtime.rollout_conversion import history_batch_to_rollout


class CompletedEpisodeRootCollector:
    """Collect terminal-labeled supervised roots from a live no-commit sampler stream."""

    def __init__(
        self,
        *,
        sampler: Any,
        policy: torch.nn.Module,
        game: Any,
        pad_token: int,
    ) -> None:
        self.sampler = sampler
        self.policy = policy
        self.game = game
        self.pad_token = int(pad_token)

        self.device = torch.device(sampler.device)
        self.target = int(sampler.rollout_batch_size)
        if self.target <= 0:
            raise RuntimeError("Sampler rollout batch size must be positive for supervised root collection.")

        self.max_rows = int(game.MAX_BELIEF_ROWS)
        self.max_tokens = int(game.MAX_BELIEF_TOKENS)
        self.max_candidates = int(game.MAX_BELIEF_CANDIDATES)
        self.max_horizon = int(sampler.T)
        self.num_players = int(game.NUM_PLAYERS)
        self.max_obs_tokens = int(game.MAX_OBSERVATION_TOKENS)
        self.num_channels = int(game.NUM_CHANNELS)

        self.env_indices = torch.arange(self.target, dtype = torch.long, device = self.device)
        self.time_indices = torch.arange(self.max_horizon, dtype = torch.long, device = self.device).unsqueeze(0)

        self._pending_roots: list[list[TensorDict]] = [[] for _ in range(self.target)]
        self._episode_returns = torch.zeros((self.target, self.num_players), dtype = torch.float32, device = self.device)
        self._cache_prepared = False
        self._validate_game_endpoints()

    def _validate_game_endpoints(self) -> None:
        if not hasattr(self.sampler, "envs") or not hasattr(self.sampler.envs, "belief_supervision"):
            raise RuntimeError(
                "CompletedEpisodeRootCollector requires sampler.envs.belief_supervision() supervision endpoint."
            )

    def reset(self) -> None:
        """Reset collector pending roots, reward state, and sampler rollout history."""
        self._pending_roots = [[] for _ in range(self.target)]
        self._episode_returns.zero_()
        self.sampler.rollout_buffer.reset()
        if hasattr(self.sampler, "last_step_rewards"):
            self.sampler.last_step_rewards = None
        if hasattr(self.sampler, "last_step_is_terminal"):
            self.sampler.last_step_is_terminal = None
        self._cache_prepared = False

    def _advance_history_strings(self, history_strs: list[str]) -> list[str]:
        advance = getattr(self.sampler.envs, "advance_history_strs", None)
        if not callable(advance):
            raise RuntimeError(
                "Staggered Yomi collection warmup requires sampler.envs.advance_history_strs()."
            )
        if len(history_strs) != self.target:
            raise RuntimeError(
                f"history_strs must match sampler batch size: expected {self.target}, got {len(history_strs)}."
            )
        advanced = [str(item) for item in advance([str(item) for item in history_strs])]
        if len(advanced) != self.target:
            raise RuntimeError(
                f"advance_history_strs returned {len(advanced)} rows for sampler batch size {self.target}."
            )
        return advanced

    def _compute_stagger_offsets(
        self,
        *,
        warmup_steps: int,
        global_env_start: int,
        global_env_count: int,
    ) -> list[int]:
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}.")
        if global_env_count <= 0:
            raise ValueError(f"global_env_count must be positive, got {global_env_count}.")
        if global_env_start < 0:
            raise ValueError(f"global_env_start must be non-negative, got {global_env_start}.")
        if global_env_start + self.target > global_env_count:
            raise ValueError(
                "local sampler env range exceeds global_env_count: "
                f"start={global_env_start}, local={self.target}, global={global_env_count}."
            )

        offsets: list[int] = []
        for local_env_idx in range(self.target):
            global_env_idx = int(global_env_start) + int(local_env_idx)
            midpoint_numerator = (2 * global_env_idx + 1) * int(warmup_steps)
            offset = midpoint_numerator // (2 * int(global_env_count))
            offsets.append(min(max(int(offset), 0), int(warmup_steps) - 1))
        return offsets

    def _snapshot_ready_mask(self) -> list[bool]:
        player_ids = self.sampler.parsed_envs_step["player_ids"].to(dtype = torch.long, device = self.device)
        if tuple(player_ids.shape) != (self.target,):
            raise RuntimeError(f"player_ids shape mismatch: expected {(self.target,)}, got {tuple(player_ids.shape)}")
        return [bool(item) for item in (player_ids >= 0).detach().cpu().tolist()]

    def _save_due_snapshot_histories(
        self,
        *,
        history_strs: list[str],
        snapshot_histories: list[str | None],
        stagger_offsets: list[int],
        step_index: int,
        terminal_mask: list[bool] | None = None,
    ) -> int:
        if len(history_strs) != self.target:
            raise RuntimeError(
                f"history_strs must match sampler batch size: expected {self.target}, got {len(history_strs)}."
            )
        if len(snapshot_histories) != self.target:
            raise RuntimeError(
                "snapshot_histories must match sampler batch size: "
                f"expected {self.target}, got {len(snapshot_histories)}."
            )
        if len(stagger_offsets) != self.target:
            raise RuntimeError(
                f"stagger_offsets must match sampler batch size: expected {self.target}, got {len(stagger_offsets)}."
            )
        if terminal_mask is not None and len(terminal_mask) != self.target:
            raise RuntimeError(
                f"terminal_mask must match sampler batch size: expected {self.target}, got {len(terminal_mask)}."
            )

        ready_mask = self._snapshot_ready_mask()
        terminal = terminal_mask or [False] * self.target
        saved = 0
        for env_idx in range(self.target):
            if snapshot_histories[env_idx] is not None:
                continue
            if int(step_index) < int(stagger_offsets[env_idx]):
                continue
            if terminal[env_idx] or not ready_mask[env_idx]:
                continue
            snapshot_histories[env_idx] = str(history_strs[env_idx])
            saved += 1
        return saved

    def _reconstruct_sampler_from_histories(self, *, history_strs: list[str], seed: int) -> None:
        reconstruct = getattr(self.game, "reconstruct_envs", None)
        if not callable(reconstruct):
            raise RuntimeError(
                "Staggered Yomi collection warmup requires game.reconstruct_envs()."
            )
        if len(history_strs) != self.target:
            raise RuntimeError(
                f"reconstruct history count mismatch: expected {self.target}, got {len(history_strs)}."
            )

        reconstructed = reconstruct([str(item) for item in history_strs], [], int(seed))
        if not isinstance(reconstructed, tuple) or len(reconstructed) != 2:
            raise RuntimeError("game.reconstruct_envs() must return tuple[EnvBatch, StepBatch].")
        envs, envs_step = reconstructed
        self.sampler.envs = envs
        self.sampler.parsed_envs_step = EnvStepBatch.from_env_step(envs_step, self.sampler.config)

    def _prefill_rollout_buffer_from_history_strs(self, *, history_strs: list[str]) -> dict[str, int]:
        parse_history_strs = getattr(self.game, "parse_history_strs", None)
        if not callable(parse_history_strs):
            raise RuntimeError(
                "Staggered Yomi collection warmup requires game.parse_history_strs() so reconstructed "
                "mid-episode envs can be aligned with sampler rollout history."
            )
        if len(history_strs) != self.target:
            raise RuntimeError(
                f"prefill history count mismatch: expected {self.target}, got {len(history_strs)}."
            )

        history_batch = parse_history_strs([str(item) for item in history_strs])
        rollout_prefix, num_steps_np, _player_ids_np, _is_commit_np, _avail_weights_np = history_batch_to_rollout(
            history_batch = history_batch,
            device = self.device,
            pad_token_id = int(self.pad_token),
            prefix_mode = "from_action_taken",
            target_batch_size = self.target,
            target_time_dim = self.max_horizon,
        )
        num_steps_np = np.asarray(num_steps_np, dtype = np.int64)
        if tuple(num_steps_np.shape) != (self.target,):
            raise RuntimeError(
                "history_batch_to_rollout returned invalid num_steps shape during Yomi warmup prefill: "
                f"expected {(self.target,)}, got {tuple(num_steps_np.shape)}."
            )

        prefix_lengths_np = num_steps_np - 1
        if bool(np.any(prefix_lengths_np < 0)):
            raise RuntimeError("Yomi warmup prefill requires each reconstructed history to include a current state.")
        if bool(np.any(prefix_lengths_np >= self.max_horizon)):
            row_idx = int(np.nonzero(prefix_lengths_np >= self.max_horizon)[0][0])
            raise RuntimeError(
                "Yomi warmup prefill would leave no room for the reconstructed current state: "
                f"row={row_idx} prefix_steps={int(prefix_lengths_np[row_idx])} horizon={self.max_horizon}."
            )

        prefix_lengths = torch.as_tensor(prefix_lengths_np, dtype = torch.long, device = self.device)
        buffer = self.sampler.rollout_buffer
        buffer["num_steps"][: self.target] = prefix_lengths
        buffer["avail_is_commit"].zero_()

        fields = ("player_ids", "obs_tokens", "obs_channels", "avail_tokens", "avail_weights")
        for row_idx, prefix_len_raw in enumerate(prefix_lengths_np.tolist()):
            prefix_len = int(prefix_len_raw)
            if prefix_len <= 0:
                continue
            for field in fields:
                dst = buffer[field]
                src = rollout_prefix[field]
                dst[row_idx, :prefix_len] = src[row_idx, :prefix_len].to(dtype = dst.dtype, device = self.device)

            chance_mask = buffer["player_ids"][row_idx, :prefix_len] < 0
            if bool(chance_mask.any().item()):
                buffer["avail_tokens"][row_idx, :prefix_len][chance_mask] = 0
                buffer["avail_weights"][row_idx, :prefix_len][chance_mask] = 0.0
                buffer["avail_weights"][row_idx, :prefix_len][chance_mask, 0] = 1.0

        return {
            "warmup_min_prefix_steps": int(prefix_lengths_np.min()),
            "warmup_max_prefix_steps": int(prefix_lengths_np.max()),
        }

    def warm_start_staggered(
        self,
        *,
        warmup_steps: int,
        global_env_start: int = 0,
        global_env_count: int | None = None,
        seed: int = 0,
    ) -> dict[str, int]:
        """Desynchronize live env ages by rebuilding from stratified warmup snapshots."""
        warmup_steps = int(warmup_steps)
        global_env_start = int(global_env_start)
        global_env_count = self.target if global_env_count is None else int(global_env_count)
        self._validate_runtime_invariants()
        stagger_offsets = self._compute_stagger_offsets(
            warmup_steps = warmup_steps,
            global_env_start = global_env_start,
            global_env_count = global_env_count,
        )

        snapshot_histories: list[str | None] = [None for _ in range(self.target)]
        history_strs = self._advance_history_strings([""] * self.target)
        saved_count = self._save_due_snapshot_histories(
            history_strs = history_strs,
            snapshot_histories = snapshot_histories,
            stagger_offsets = stagger_offsets,
            step_index = 0,
        )

        sampler_steps = 0
        max_sampler_steps = int(warmup_steps) + int(self.max_horizon) * 2
        with torch.inference_mode():
            self._prepare_agent_cache()
            while saved_count < self.target:
                if sampler_steps >= max_sampler_steps:
                    missing = [idx for idx, item in enumerate(snapshot_histories) if item is None]
                    raise RuntimeError(
                        "Staggered Yomi collection warmup did not capture all live-env snapshots "
                        f"within {max_sampler_steps} sampler steps; missing env indices: {missing}."
                    )
                step_done = self.sampler.rollout_step(agent_model = self.policy).to(
                    dtype = torch.bool,
                    device = self.device,
                )
                sampler_steps += 1
                advanced_history_strs = self._advance_history_strings(history_strs)
                terminal_mask = [bool(item) for item in step_done.detach().cpu().tolist()]
                saved_count += self._save_due_snapshot_histories(
                    history_strs = advanced_history_strs,
                    snapshot_histories = snapshot_histories,
                    stagger_offsets = stagger_offsets,
                    step_index = sampler_steps,
                    terminal_mask = terminal_mask,
                )
                history_strs = [
                    "" if terminal_mask[env_idx] else str(advanced_history_strs[env_idx])
                    for env_idx in range(self.target)
                ]

        final_histories = [item for item in snapshot_histories if item is not None]
        if len(final_histories) != self.target:
            raise RuntimeError("Staggered Yomi collection warmup produced incomplete snapshot histories.")
        self._reconstruct_sampler_from_histories(history_strs = final_histories, seed = int(seed))
        self.reset()
        prefill_stats = self._prefill_rollout_buffer_from_history_strs(history_strs = final_histories)
        stats = {
            "warmup_sampler_steps": int(sampler_steps),
            "warmup_stagger_span": int(warmup_steps),
            "warmup_snapshots": int(len(final_histories)),
            "warmup_min_offset": int(min(stagger_offsets)),
            "warmup_max_offset": int(max(stagger_offsets)),
        }
        stats.update(prefill_stats)
        return stats

    def _validate_runtime_invariants(self) -> None:
        if int(self.sampler.rollout_batch_size) != self.target:
            raise RuntimeError(
                f"Sampler rollout batch size changed: expected {self.target}, got {int(self.sampler.rollout_batch_size)}."
            )
        if int(self.sampler.T) != self.max_horizon:
            raise RuntimeError(
                f"Sampler max_game_horizon changed: expected {self.max_horizon}, got {int(self.sampler.T)}."
            )
        if torch.device(self.sampler.device) != self.device:
            raise RuntimeError(
                f"Sampler device changed: expected {self.device}, got {torch.device(self.sampler.device)}."
            )
        rollout_num_steps = self.sampler.rollout_buffer["num_steps"]
        if tuple(rollout_num_steps.shape) != (self.target,):
            raise RuntimeError(
                f"Sampler rollout_buffer['num_steps'] shape mismatch: expected {(self.target,)}, got {tuple(rollout_num_steps.shape)}."
            )
        rollout_obs_tokens = self.sampler.rollout_buffer["obs_tokens"]
        expected_obs_shape = (self.target, self.max_horizon, self.num_players, self.max_obs_tokens)
        if tuple(rollout_obs_tokens.shape) != expected_obs_shape:
            raise RuntimeError(
                "Sampler rollout_buffer['obs_tokens'] shape mismatch: "
                f"expected {expected_obs_shape}, got {tuple(rollout_obs_tokens.shape)}."
            )
        rollout_obs_channels = self.sampler.rollout_buffer["obs_channels"]
        expected_obs_channels_shape = (self.target, self.max_horizon, self.num_players, self.num_channels)
        if tuple(rollout_obs_channels.shape) != expected_obs_channels_shape:
            raise RuntimeError(
                "Sampler rollout_buffer['obs_channels'] shape mismatch: "
                f"expected {expected_obs_channels_shape}, got {tuple(rollout_obs_channels.shape)}."
            )
        rollout_player_ids = self.sampler.rollout_buffer["player_ids"]
        expected_player_ids_shape = (self.target, self.max_horizon)
        if tuple(rollout_player_ids.shape) != expected_player_ids_shape:
            raise RuntimeError(
                "Sampler rollout_buffer['player_ids'] shape mismatch: "
                f"expected {expected_player_ids_shape}, got {tuple(rollout_player_ids.shape)}."
            )

    def _prepare_agent_cache(self) -> None:
        if self._cache_prepared:
            return
        self.sampler.prepare_agent_cache(agent_model = self.policy)
        self._cache_prepared = True

    def _parse_belief_table_supervision(
        self,
        raw: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(raw, dict):
            raise RuntimeError("envs.belief_supervision() must return a dict supervision payload.")
        required = ("candidate_tokens", "candidate_valid", "target_index", "belief_length")
        missing = [name for name in required if name not in raw]
        if missing:
            raise RuntimeError(f"envs.belief_supervision() missing required fields: {missing}")

        candidate_tokens_np = np.asarray(raw["candidate_tokens"], dtype = np.int32)
        candidate_valid_np = np.asarray(raw["candidate_valid"], dtype = np.uint8)
        target_index_np = np.asarray(raw["target_index"], dtype = np.int64)
        belief_length_np = np.asarray(raw["belief_length"], dtype = np.int64)

        expected_tokens_shape = (self.target, self.max_rows, self.max_candidates, self.max_tokens)
        if candidate_tokens_np.shape != expected_tokens_shape:
            raise RuntimeError(
                "envs.belief_supervision()['candidate_tokens'] shape mismatch: "
                f"expected {expected_tokens_shape}, got {candidate_tokens_np.shape}."
            )
        expected_valid_shape = (self.target, self.max_rows, self.max_candidates)
        if candidate_valid_np.shape != expected_valid_shape:
            raise RuntimeError(
                "envs.belief_supervision()['candidate_valid'] shape mismatch: "
                f"expected {expected_valid_shape}, got {candidate_valid_np.shape}."
            )
        expected_index_shape = (self.target, self.max_rows)
        if target_index_np.shape != expected_index_shape:
            raise RuntimeError(
                "envs.belief_supervision()['target_index'] shape mismatch: "
                f"expected {expected_index_shape}, got {target_index_np.shape}."
            )
        if belief_length_np.shape != (self.target,):
            raise RuntimeError(
                "envs.belief_supervision()['belief_length'] shape mismatch: "
                f"expected {(self.target,)}, got {belief_length_np.shape}."
            )

        if bool((belief_length_np < 0).any()) or bool((belief_length_np > self.max_rows).any()):
            raise RuntimeError("belief_length must be within [0, MAX_BELIEF_ROWS].")

        target_tokens_np = np.full(
            (self.target, self.max_rows, self.max_tokens),
            fill_value = int(self.pad_token),
            dtype = np.int32,
        )
        target_mask_np = np.zeros((self.target, self.max_rows), dtype = np.uint8)

        for b in range(self.target):
            active_rows = int(belief_length_np[b])
            for t in range(self.max_rows):
                idx = int(target_index_np[b, t])
                row_has_candidates = bool(np.any(candidate_valid_np[b, t] == 1))
                if t < active_rows:
                    if idx < 0 or idx >= self.max_candidates:
                        raise RuntimeError("Active belief rows must have target_index in candidate range.")
                    if not row_has_candidates:
                        raise RuntimeError("Active belief rows must have at least one valid candidate.")
                    if int(candidate_valid_np[b, t, idx]) != 1:
                        raise RuntimeError("Active target_index must point to a valid candidate.")
                    target_tokens_np[b, t] = candidate_tokens_np[b, t, idx]
                    target_mask_np[b, t] = 1
                else:
                    if idx != -1:
                        raise RuntimeError("Inactive belief rows must use target_index = -1.")
                    if row_has_candidates:
                        raise RuntimeError("Inactive belief rows must not expose valid candidates.")

        return (
            torch.as_tensor(candidate_tokens_np, dtype = torch.long, device = self.device),
            torch.as_tensor(candidate_valid_np, dtype = torch.long, device = self.device),
            torch.as_tensor(target_index_np, dtype = torch.long, device = self.device),
            torch.as_tensor(belief_length_np, dtype = torch.long, device = self.device),
            torch.as_tensor(target_tokens_np, dtype = torch.long, device = self.device),
            torch.as_tensor(target_mask_np, dtype = torch.long, device = self.device),
        )

    @staticmethod
    def _cat_chunks(chunks: list[TensorDict]) -> TensorDict:
        if not chunks:
            raise RuntimeError("Cannot concatenate an empty TensorDict chunk list.")
        keys = tuple(chunks[0].keys())
        data = {
            key: torch.cat([chunk[key] for chunk in chunks], dim = 0)
            for key in keys
        }
        batch_size = int(data[keys[0]].shape[0])
        return TensorDict(data, batch_size = [batch_size], device = chunks[0].device)

    def _build_root_chunk(
        self,
        *,
        root_env_indices: torch.Tensor,
        player_ids_step: torch.Tensor,
        obs_tokens_step: torch.Tensor,
        obs_channels_step: torch.Tensor,
        episode_step: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_valid: torch.Tensor,
        target_index: torch.Tensor,
        belief_length: torch.Tensor,
        target_tokens: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> TensorDict:
        selected_players = player_ids_step[root_env_indices]
        selected_steps = episode_step[root_env_indices]
        selected_history = self.sampler.rollout_buffer["obs_tokens"].index_select(0, root_env_indices).to(
            dtype = torch.long,
            device = self.device,
        )
        selected_obs_tokens_step = obs_tokens_step.index_select(0, root_env_indices)
        selected_obs_channels_step = obs_channels_step.index_select(0, root_env_indices)

        selected_candidate_tokens = candidate_tokens.index_select(0, root_env_indices)
        selected_candidate_valid = candidate_valid.index_select(0, root_env_indices)
        selected_target_index = target_index.index_select(0, root_env_indices)
        selected_belief_length = belief_length.index_select(0, root_env_indices)
        selected_belief = target_tokens.index_select(0, root_env_indices)
        selected_belief_mask = target_mask.index_select(0, root_env_indices)

        pid_index = selected_players.view(-1, 1, 1, 1).expand(
            -1,
            self.max_horizon,
            1,
            self.max_obs_tokens,
        )
        selected_obs_prefix_tokens = torch.gather(
            selected_history,
            dim = 2,
            index = pid_index,
        ).squeeze(2).clone()
        current_player_obs = torch.gather(
            selected_obs_tokens_step,
            dim = 1,
            index = selected_players.view(-1, 1, 1).expand(-1, 1, self.max_obs_tokens),
        ).squeeze(1)
        row_indices = torch.arange(int(selected_players.shape[0]), dtype = torch.long, device = self.device)
        selected_obs_prefix_tokens[row_indices, selected_steps, :] = current_player_obs
        selected_obs_prefix_mask = (
            self.time_indices < (selected_steps + 1).unsqueeze(1)
        ).to(dtype = torch.long)

        channel_index = selected_players.view(-1, 1, 1).expand(-1, 1, self.num_channels)
        selected_state_channels = torch.gather(
            selected_obs_channels_step,
            dim = 1,
            index = channel_index,
        ).squeeze(1).to(dtype = torch.float32)

        selected_belief_input = torch.full_like(selected_belief, fill_value = self.pad_token)
        if selected_belief.shape[1] > 1:
            selected_belief_input[:, 1:, :] = selected_belief[:, :-1, :]

        return_before_root = self._episode_returns[root_env_indices, selected_players].to(dtype = torch.float32)
        return TensorDict(
            {
                "obs_prefix_tokens": selected_obs_prefix_tokens,
                "obs_prefix_mask": selected_obs_prefix_mask,
                "state_channels": selected_state_channels,
                "belief_input_tokens": selected_belief_input,
                "candidate_tokens": selected_candidate_tokens,
                "candidate_valid": selected_candidate_valid,
                "belief_target_index": selected_target_index,
                "belief_length": selected_belief_length,
                "belief_target_tokens": selected_belief,
                "belief_target_mask": selected_belief_mask,
                "return_before_root": return_before_root,
                "player_ids": selected_players,
            },
            batch_size = [int(selected_players.shape[0])],
            device = self.device,
        )

    def _capture_current_roots(self) -> None:
        parsed_step = self.sampler.parsed_envs_step
        player_ids_step = parsed_step["player_ids"].to(dtype = torch.long, device = self.device)
        obs_tokens_step = parsed_step["obs_tokens"].to(dtype = torch.long, device = self.device)
        obs_channels_step = parsed_step["obs_channels"].to(dtype = torch.long, device = self.device)

        if player_ids_step.shape != (self.target,):
            raise RuntimeError(f"player_ids shape mismatch: expected {(self.target,)}, got {tuple(player_ids_step.shape)}")
        if obs_tokens_step.shape != (self.target, self.num_players, self.max_obs_tokens):
            raise RuntimeError(
                "obs_tokens shape mismatch: "
                f"expected {(self.target, self.num_players, self.max_obs_tokens)}, got {tuple(obs_tokens_step.shape)}"
            )
        if obs_channels_step.shape != (self.target, self.num_players, self.num_channels):
            raise RuntimeError(
                "obs_channels shape mismatch: "
                f"expected {(self.target, self.num_players, self.num_channels)}, got {tuple(obs_channels_step.shape)}"
            )
        episode_step = self.sampler.rollout_buffer["num_steps"].to(dtype = torch.long, device = self.device)
        if bool((episode_step < 0).any().item()) or bool((episode_step >= self.max_horizon).any().item()):
            raise RuntimeError(
                "Observed rollout_buffer num_steps outside valid range while collecting supervised roots."
            )

        (
            candidate_tokens,
            candidate_valid,
            target_index,
            belief_length,
            target_tokens,
            target_mask,
        ) = self._parse_belief_table_supervision(self.sampler.envs.belief_supervision())

        active = player_ids_step >= 0
        supervised = belief_length > 0
        if bool((supervised & ~active).any().item()):
            raise RuntimeError("envs.belief_supervision() emitted positive belief_length for an inactive state.")
        root_env_indices = torch.nonzero(active & supervised, as_tuple = False).squeeze(-1)
        if int(root_env_indices.numel()) <= 0:
            return

        root_chunk = self._build_root_chunk(
            root_env_indices = root_env_indices,
            player_ids_step = player_ids_step,
            obs_tokens_step = obs_tokens_step,
            obs_channels_step = obs_channels_step,
            episode_step = episode_step,
            candidate_tokens = candidate_tokens,
            candidate_valid = candidate_valid,
            target_index = target_index,
            belief_length = belief_length,
            target_tokens = target_tokens,
            target_mask = target_mask,
        )
        env_ids = root_env_indices.detach().cpu().tolist()
        for row, env_id in enumerate(env_ids):
            self._pending_roots[int(env_id)].append(root_chunk[row: row + 1].clone())

    def _read_last_step_rewards(self, step_done: torch.Tensor) -> torch.Tensor:
        rewards = getattr(self.sampler, "last_step_rewards", None)
        if rewards is None:
            raise RuntimeError(
                "CompletedEpisodeRootCollector requires sampler.last_step_rewards after rollout_step()."
            )
        rewards_tensor = rewards.to(dtype = torch.float32, device = self.device)
        expected_shape = (self.target, self.num_players)
        if tuple(rewards_tensor.shape) != expected_shape:
            raise RuntimeError(
                f"sampler.last_step_rewards shape mismatch: expected {expected_shape}, got {tuple(rewards_tensor.shape)}."
            )
        if step_done.shape != (self.target,):
            raise RuntimeError(f"step_done shape mismatch: expected {(self.target,)}, got {tuple(step_done.shape)}")
        return rewards_tensor

    def _flush_completed_envs(self, *, done_env_indices: torch.Tensor) -> list[TensorDict]:
        flushed: list[TensorDict] = []
        for env_id in done_env_indices.detach().cpu().tolist():
            env_idx = int(env_id)
            pending = self._pending_roots[env_idx]
            if pending:
                env_batch = self._cat_chunks(pending)
                players = env_batch["player_ids"].to(dtype = torch.long)
                final_returns = self._episode_returns[env_idx].index_select(dim = 0, index = players)
                reward_to_go = final_returns.to(dtype = torch.float32) - env_batch["return_before_root"].to(
                    dtype = torch.float32
                )
                env_batch.set("reward_to_go", reward_to_go)
                env_batch.set("reward_to_go_mask", torch.ones_like(reward_to_go, dtype = torch.long))
                flushed.append(env_batch)
            self._pending_roots[env_idx] = []
            self._episode_returns[env_idx].zero_()
        return flushed

    def _build_collected_batch(
        self,
        *,
        chunks: list[TensorDict],
        completed_episodes: int,
        env_steps: int,
        interval_chunks: int,
    ) -> TensorDict:
        batch = self._cat_chunks(chunks)
        batch_size = int(batch.batch_size[0])
        if batch_size <= 0:
            raise RuntimeError("Collected completed-root batch is empty.")

        prefix_lengths = batch["obs_prefix_mask"].sum(dim = 1)
        if bool((prefix_lengths <= 0).any().item()):
            raise RuntimeError("Completed-root samples must have non-empty observation prefixes.")
        if "reward_to_go" not in batch.keys() or "reward_to_go_mask" not in batch.keys():
            raise RuntimeError("Completed-root samples must include reward_to_go labels.")
        if bool((batch["reward_to_go_mask"].to(dtype = torch.bool) == 0).any().item()):
            raise RuntimeError("Completed-root samples must all have active reward_to_go labels.")

        batch.set(
            "collection_completed_episodes",
            torch.full((batch_size,), int(completed_episodes), dtype = torch.long, device = self.device),
        )
        batch.set(
            "collection_env_steps",
            torch.full((batch_size,), int(env_steps), dtype = torch.long, device = self.device),
        )
        batch.set(
            "collection_interval_chunks",
            torch.full((batch_size,), int(interval_chunks), dtype = torch.long, device = self.device),
        )
        return batch

    def collect(self, *, collect_interval: int) -> TensorDict:
        collect_interval = int(collect_interval)
        if collect_interval <= 0:
            raise ValueError("collect_interval must be positive.")
        self._validate_runtime_invariants()

        chunks: list[TensorDict] = []
        completed_episodes = 0
        env_steps = 0
        interval_chunks = 0
        sampler_steps_without_output = 0

        with torch.inference_mode():
            self._prepare_agent_cache()
            while not chunks:
                interval_chunks += 1
                for _ in range(collect_interval):
                    self._capture_current_roots()
                    step_done = self.sampler.rollout_step(agent_model = self.policy).to(
                        dtype = torch.bool,
                        device = self.device,
                    )
                    rewards = self._read_last_step_rewards(step_done)
                    self._episode_returns.add_(rewards)
                    env_steps += int(self.target)
                    sampler_steps_without_output += 1

                    done_env_indices = torch.nonzero(step_done, as_tuple = False).squeeze(-1)
                    if int(done_env_indices.numel()) > 0:
                        completed_episodes += int(done_env_indices.numel())
                        chunks.extend(self._flush_completed_envs(done_env_indices = done_env_indices))

                    if not chunks and sampler_steps_without_output >= self.max_horizon * 2:
                        raise RuntimeError(
                            "No completed root-bearing episodes produced for too many sampler steps."
                        )

        return self._build_collected_batch(
            chunks = chunks,
            completed_episodes = completed_episodes,
            env_steps = env_steps,
            interval_chunks = interval_chunks,
        )


__all__ = ["CompletedEpisodeRootCollector"]
