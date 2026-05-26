"""Critic-backed action-value estimation for update-equivalent search."""

from __future__ import annotations

from typing import Any, Sequence
from zlib import crc32

import numpy as np
import torch

from kagami.search.pending_policy import PendingPolicyEval
from marvel.runtime.rollout_conversion import history_batch_to_rollout


class CriticActionValueEstimator:
    """Yomi belief sampling plus direct MARVEL critic Q estimation."""

    def __init__(
        self,
        *,
        game: Any,
        belief_model: Any,
        critic_model: Any,
        device: str | torch.device,
        belief_temp: float,
        belief_sample_count: int,
    ) -> None:
        self._game = game
        self._belief_model = belief_model
        self._critic_model = critic_model
        self._device = torch.device(device)
        self._belief_temp = float(belief_temp)
        self._belief_sample_count = int(belief_sample_count)
        if self._belief_sample_count <= 0:
            raise ValueError("belief_sample_count must be positive.")
        output_mode = str(getattr(critic_model, "output_mode", "")).strip().lower()
        if output_mode != "qvalue":
            raise ValueError("CriticActionValueEstimator requires a qvalue CentralCritic.")
        self._critic_model.eval()

    def estimate_action_qvalues(
        self,
        *,
        infoset_str: str,
        root_player_id: int,
        pending_action: str,
        pending_eval: PendingPolicyEval,
        seed: int,
    ) -> dict[str, float]:
        action_strs = [str(action).strip() for action in pending_eval.action_strs]
        if any(not action for action in action_strs):
            raise RuntimeError("Critic action-value estimation received an empty action string.")
        action_token_rows = np.asarray(pending_eval.action_token_rows, dtype = np.int64)
        if int(action_token_rows.ndim) != 2 or int(action_token_rows.shape[0]) != len(action_strs):
            raise RuntimeError(
                "pending_eval.action_token_rows must have shape [num_actions, max_action_tokens] for critic search: "
                f"actions={len(action_strs)} shape={tuple(action_token_rows.shape)}."
            )

        belief_samples = self._belief_model.sample_determinized_histories(
            infoset_strs = [str(infoset_str)],
            num_samples = int(self._belief_sample_count),
            temperature = float(self._belief_temp),
            seed = self._mix_seed(int(seed), "belief"),
        )[0]
        history_rows = [str(sample.history_str) for sample in belief_samples]
        if not history_rows:
            raise RuntimeError("Critic action-value estimation requires at least one belief sample.")

        rollout, num_steps_np = self._build_critic_rollout(
            history_rows = history_rows,
            pending_action = str(pending_action).strip(),
            seed = self._mix_seed(int(seed), "critic_history"),
        )
        qvalues_np = self._evaluate_critic_qvalues(rollout = rollout, batch_size = len(history_rows))
        action_indices = self._select_legal_action_indices(
            rollout = rollout,
            num_steps_np = num_steps_np,
            action_token_rows = action_token_rows,
        )

        root = int(root_player_id)
        if root < 0 or root >= int(qvalues_np.shape[3]):
            raise RuntimeError(
                "root_player_id is out of range for critic qvalues: "
                f"root_player_id={root} qvalue_players={int(qvalues_np.shape[3])}."
            )

        row_idx = np.arange(len(history_rows), dtype = np.int64)
        last_steps = np.asarray(num_steps_np, dtype = np.int64) - 1
        if np.any(last_steps < 0) or np.any(last_steps >= int(qvalues_np.shape[1])):
            raise RuntimeError(
                "HistoryBatch.num_steps produced out-of-range critic timestep indices: "
                f"num_steps={num_steps_np.tolist()} qvalue_time_dim={int(qvalues_np.shape[1])}."
            )

        out: dict[str, float] = {}
        for action_idx, action_str in enumerate(action_strs):
            legal_indices = action_indices[:, int(action_idx)]
            values = qvalues_np[row_idx, last_steps, legal_indices, root]
            out[str(action_str)] = float(np.mean(values, dtype = np.float64))
        return out

    def _build_critic_rollout(
        self,
        *,
        history_rows: Sequence[str],
        pending_action: str,
        seed: int,
    ) -> tuple[Any, np.ndarray]:
        parse_history_strs = getattr(self._game, "parse_history_strs", None)
        if callable(parse_history_strs):
            history_batch = parse_history_strs([str(row) for row in history_rows])
            rollout, num_steps_np, _player_ids_np, _is_commit_np, _avail_weights_np = history_batch_to_rollout(
                history_batch = history_batch,
                device = self._device,
                pad_token_id = int(self._game.PAD_TOKEN),
                prefix_mode = "from_action_taken",
            )
            if str(pending_action):
                self._overwrite_current_rollout_step_from_reconstructed_env(
                    rollout = rollout,
                    num_steps_np = num_steps_np,
                    history_rows = history_rows,
                    pending_action = str(pending_action),
                    seed = int(seed),
                )
            return rollout, num_steps_np

        infoset_str_from_history_strs = getattr(self._game, "infoset_str_from_history_strs", None)
        parse_infoset_strs = getattr(self._game, "parse_infoset_strs", None)
        if not callable(infoset_str_from_history_strs) or not callable(parse_infoset_strs):
            raise RuntimeError(
                "Critic action-value estimation requires game.parse_history_strs(...) or "
                "game.infoset_str_from_history_strs(...) plus game.parse_infoset_strs(...)."
            )

        infoset_rows = [str(item) for item in infoset_str_from_history_strs([str(row) for row in history_rows])]
        if len(infoset_rows) != len(history_rows):
            raise RuntimeError("infoset_str_from_history_strs returned an unexpected row count for critic search.")
        history_batch = parse_infoset_strs(infoset_rows, [str(pending_action)] * len(infoset_rows))
        rollout, num_steps_np, _player_ids_np, _is_commit_np, _avail_weights_np = history_batch_to_rollout(
            history_batch = history_batch,
            device = self._device,
            pad_token_id = int(self._game.PAD_TOKEN),
            prefix_mode = "ignore",
        )
        return rollout, num_steps_np

    def _overwrite_current_rollout_step_from_reconstructed_env(
        self,
        *,
        rollout: Any,
        num_steps_np: np.ndarray,
        history_rows: Sequence[str],
        pending_action: str,
        seed: int,
    ) -> None:
        reconstruct_envs = getattr(self._game, "reconstruct_envs", None)
        if not callable(reconstruct_envs):
            raise RuntimeError(
                "Critic action-value estimation with a pending prefix requires game.reconstruct_envs(...)."
            )

        _envs, env_step = reconstruct_envs(
            [str(row) for row in history_rows],
            pending_actions = [str(pending_action)] * len(history_rows),
            seed = int(seed),
        )
        batch_size = len(history_rows)
        current_obs_tokens_np = np.asarray(env_step.observations.tokens, dtype = np.int64)
        current_obs_channels_np = np.asarray(env_step.observations.channels, dtype = np.int64)
        current_player_ids_np = np.asarray(env_step.player_ids, dtype = np.int64)
        current_avail_tokens_np = np.asarray(env_step.available_actions.tokens, dtype = np.int64)
        current_avail_weights_np = np.asarray(env_step.available_actions.weights, dtype = np.float32)

        if tuple(current_obs_tokens_np.shape) != tuple(rollout["obs_tokens"].shape[0:1] + rollout["obs_tokens"].shape[2:]):
            raise RuntimeError(
                "reconstruct_envs returned obs_tokens incompatible with the parsed critic rollout: "
                f"current={tuple(current_obs_tokens_np.shape)} rollout={tuple(rollout['obs_tokens'].shape)}."
            )
        if tuple(current_obs_channels_np.shape) != tuple(
            rollout["obs_channels"].shape[0:1] + rollout["obs_channels"].shape[2:]
        ):
            raise RuntimeError(
                "reconstruct_envs returned obs_channels incompatible with the parsed critic rollout: "
                f"current={tuple(current_obs_channels_np.shape)} rollout={tuple(rollout['obs_channels'].shape)}."
            )
        if tuple(current_avail_tokens_np.shape) != tuple(
            rollout["avail_tokens"].shape[0:1] + rollout["avail_tokens"].shape[2:]
        ):
            raise RuntimeError(
                "reconstruct_envs returned available-action tokens incompatible with the parsed critic rollout: "
                f"current={tuple(current_avail_tokens_np.shape)} rollout={tuple(rollout['avail_tokens'].shape)}."
            )
        if tuple(current_avail_weights_np.shape) != tuple(
            rollout["avail_weights"].shape[0:1] + rollout["avail_weights"].shape[2:]
        ):
            raise RuntimeError(
                "reconstruct_envs returned available-action weights incompatible with the parsed critic rollout: "
                f"current={tuple(current_avail_weights_np.shape)} rollout={tuple(rollout['avail_weights'].shape)}."
            )
        if tuple(current_player_ids_np.shape) != (batch_size,):
            raise RuntimeError(
                "reconstruct_envs returned invalid player_ids shape for critic search: "
                f"expected={(batch_size,)}, got={tuple(current_player_ids_np.shape)}."
            )

        row_idx_t = torch.arange(batch_size, dtype = torch.long, device = self._device)
        last_steps_t = torch.as_tensor(np.asarray(num_steps_np, dtype = np.int64) - 1, dtype = torch.long, device = self._device)
        if bool(torch.any(last_steps_t < 0).item()):
            raise RuntimeError("Critic rollout num_steps must be positive before pending-prefix overwrite.")

        rollout["obs_tokens"][row_idx_t, last_steps_t] = torch.as_tensor(
            current_obs_tokens_np,
            dtype = torch.long,
            device = self._device,
        )
        rollout["obs_channels"][row_idx_t, last_steps_t] = torch.as_tensor(
            current_obs_channels_np,
            dtype = rollout["obs_channels"].dtype,
            device = self._device,
        )
        rollout["player_ids"][row_idx_t, last_steps_t] = torch.as_tensor(
            current_player_ids_np,
            dtype = torch.long,
            device = self._device,
        )
        rollout["avail_tokens"][row_idx_t, last_steps_t] = torch.as_tensor(
            current_avail_tokens_np,
            dtype = torch.long,
            device = self._device,
        )
        rollout["avail_weights"][row_idx_t, last_steps_t] = torch.as_tensor(
            current_avail_weights_np,
            dtype = rollout["avail_weights"].dtype,
            device = self._device,
        )

    def _evaluate_critic_qvalues(self, *, rollout: Any, batch_size: int) -> np.ndarray:
        with torch.inference_mode():
            out = self._critic_model(rollout = rollout)
        qvalues = getattr(out, "qvalues", None)
        if qvalues is None:
            raise RuntimeError("qvalue CentralCritic did not return qvalues.")
        if int(qvalues.ndim) != 4 or int(qvalues.shape[0]) != int(batch_size):
            raise RuntimeError(
                "CentralCritic qvalues must have shape [B, T, A, P]: "
                f"batch_size={batch_size} shape={tuple(qvalues.shape)}."
            )
        return qvalues.detach().to(dtype = torch.float64, device = torch.device("cpu")).numpy()

    def _select_legal_action_indices(
        self,
        *,
        rollout: Any,
        num_steps_np: np.ndarray,
        action_token_rows: np.ndarray,
    ) -> np.ndarray:
        row_idx = np.arange(int(num_steps_np.shape[0]), dtype = np.int64)
        last_steps = np.asarray(num_steps_np, dtype = np.int64) - 1
        legal_tokens = (
            rollout["avail_tokens"]
            .detach()
            .to(device = torch.device("cpu"))
            .numpy()[row_idx, last_steps]
        )
        legal_weights = (
            rollout["avail_weights"]
            .detach()
            .to(dtype = torch.float64, device = torch.device("cpu"))
            .numpy()[row_idx, last_steps]
        )
        if int(legal_tokens.ndim) != 3:
            raise RuntimeError(
                "Critic rollout available-action tokens must have shape [B, A, MAX_ACTION_TOKENS]: "
                f"{tuple(legal_tokens.shape)}."
            )
        if legal_weights.shape[:2] != legal_tokens.shape[:2]:
            raise RuntimeError(
                "Critic rollout legal-action tensors are misaligned: "
                f"tokens={tuple(legal_tokens.shape)} weights={tuple(legal_weights.shape)}."
            )

        requested = np.asarray(action_token_rows, dtype = np.int64)
        if int(requested.ndim) != 2 or int(requested.shape[1]) != int(legal_tokens.shape[2]):
            raise RuntimeError(
                "action_token_rows shape mismatch for critic legal-action matching: "
                f"expected={(int(requested.shape[0]), int(legal_tokens.shape[2]))}, got={tuple(requested.shape)}."
            )

        action_indices = np.full((int(legal_tokens.shape[0]), int(requested.shape[0])), -1, dtype = np.int64)
        for sample_idx in range(int(legal_tokens.shape[0])):
            row_lookup = self._legal_action_lookup(
                legal_token_rows = legal_tokens[sample_idx],
                legal_weight_row = legal_weights[sample_idx],
            )
            for action_idx in range(int(requested.shape[0])):
                wanted = self._token_tuple(requested[action_idx])
                legal_idx = row_lookup.get(wanted, -1)
                if legal_idx < 0:
                    raise RuntimeError(
                        "Critic search action token row is not legal at sampled state: "
                        f"sample_idx={sample_idx} action_idx={action_idx} tokens={wanted!r}."
                    )
                action_indices[sample_idx, action_idx] = int(legal_idx)
        return action_indices

    def _legal_action_lookup(
        self,
        *,
        legal_token_rows: np.ndarray,
        legal_weight_row: np.ndarray,
    ) -> dict[tuple[int, ...], int]:
        out: dict[tuple[int, ...], int] = {}
        for action_idx in range(int(legal_token_rows.shape[0])):
            if float(legal_weight_row[action_idx]) <= 0.0:
                continue
            key = self._token_tuple(legal_token_rows[action_idx])
            if key:
                out[key] = int(action_idx)
        return out

    def _token_tuple(self, row: np.ndarray) -> tuple[int, ...]:
        pad_token = int(self._game.PAD_TOKEN)
        return tuple(int(token) for token in np.asarray(row, dtype = np.int64).tolist() if int(token) != pad_token)

    @staticmethod
    def _mix_seed(seed: int, *parts: str) -> int:
        mixed = int(seed) & 0xFFFFFFFF
        for part in parts:
            mixed ^= int(crc32(str(part).encode("utf-8")) & 0xFFFFFFFF)
        return int(mixed & 0x7FFFFFFF)


__all__ = ["CriticActionValueEstimator"]
