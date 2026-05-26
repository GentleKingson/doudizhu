"""Internal Yomi checkpoint loading and belief sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from marvel.runtime.checkpoints import CheckpointState, CheckpointStore, load_checkpoint_into_state
from yomi.config import ModelConfig, deserialize_train_config
from yomi.model import BeliefDecoderConcat


@dataclass(frozen = True)
class CurrentBeliefSample:
    history_str: str


class YomiBeliefModel:
    """Load a trained Yomi checkpoint and sample belief-conditioned determinized histories."""

    def __init__(
        self,
        *,
        model: BeliefDecoderConcat,
        game: Any,
        device: torch.device,
        train_config: Any | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        self._model = model
        self.game = game
        self._device = torch.device(device)
        self.train_config = train_config
        self.checkpoint_path = checkpoint_path

    @classmethod
    def load_from_checkpoint(
        cls,
        *,
        checkpoint: str | Path,
        device: str | torch.device = "cuda",
        game: Any | None = None,
    ) -> "YomiBeliefModel":
        device_obj = torch.device(str(device))
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            device_obj = torch.device("cpu")

        resolved_checkpoint = CheckpointStore.resolve_checkpoint_file_path(checkpoint)
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"Yomi checkpoint not found: {resolved_checkpoint}")

        payload = CheckpointStore.load_checkpoint_payload_file(resolved_checkpoint, device = device_obj)
        train_config = deserialize_train_config(payload.get("train_config") or payload.get("yomi_config"))
        resolved_game = game if game is not None else getattr(train_config, "game", None)
        if resolved_game is None:
            raise ValueError("Yomi belief checkpoint is missing train_config game metadata; pass game explicitly.")

        if train_config is None:
            raise ValueError("Yomi belief checkpoint is missing train_config model metadata.")
        resolved_model_cfg = ModelConfig(**vars(train_config.model))

        model = BeliefDecoderConcat(game = resolved_game, model_cfg = resolved_model_cfg).to(device_obj)
        state = CheckpointState({
            "model_state_dict": model,
            "iteration": 0,
        })
        load_checkpoint_into_state(payload, state)
        if state.iteration <= 0:
            raise RuntimeError("Yomi belief checkpoint is missing required iteration metadata.")
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return cls(
            model = model,
            game = resolved_game,
            device = device_obj,
            train_config = train_config,
            checkpoint_path = resolved_checkpoint,
        )

    def sample_determinized_histories(
        self,
        *,
        infoset_strs: Sequence[str],
        num_samples: int,
        temperature: float,
        seed: int,
    ) -> list[list[CurrentBeliefSample]]:
        state_strings = [str(infoset).strip() for infoset in infoset_strs]
        if any(not state for state in state_strings):
            raise ValueError("infoset_strs must not contain empty items.")
        if not state_strings:
            return []
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be positive.")
        if not np.isfinite(float(temperature)) or float(temperature) < 0.0:
            raise ValueError("temperature must be finite and non-negative.")

        batch = self.game.parse_infoset_strs(list(state_strings), [""] * len(state_strings))
        obs_tokens_np = np.asarray(batch.observations.tokens, dtype = np.int64)
        obs_channels_np = np.asarray(batch.observations.channels, dtype = np.int64)
        num_steps_np = np.asarray(batch.num_steps, dtype = np.int64)
        player_ids_hist_np = np.asarray(batch.player_ids, dtype = np.int64)

        batch_size, time_dim, _, obs_width = obs_tokens_np.shape
        num_channels = int(obs_channels_np.shape[-1])
        obs_prefix_tokens_np = np.full(
            (batch_size, time_dim, obs_width),
            fill_value = int(self.game.PAD_TOKEN),
            dtype = np.int64,
        )
        obs_prefix_mask_np = np.zeros((batch_size, time_dim), dtype = np.int64)
        state_channels_np = np.zeros((batch_size, num_channels), dtype = np.int64)

        for batch_idx in range(batch_size):
            step_count = int(num_steps_np[batch_idx])
            if step_count <= 0:
                raise RuntimeError("parse_infoset_strs returned a non-positive num_steps entry for Yomi belief input.")
            player_id = int(player_ids_hist_np[batch_idx, step_count - 1])
            if player_id < 0 or player_id >= int(self.game.NUM_PLAYERS):
                raise RuntimeError(
                    "parse_infoset_strs returned a non-active final player_id for Yomi belief input: "
                    f"{player_id}. Expected [0, {int(self.game.NUM_PLAYERS)})."
                )
            obs_prefix_tokens_np[batch_idx] = obs_tokens_np[batch_idx, :, player_id, :]
            obs_prefix_mask_np[batch_idx, :step_count] = 1
            state_channels_np[batch_idx] = obs_channels_np[batch_idx, step_count - 1, player_id, :]

        obs_prefix_tokens_base = torch.as_tensor(obs_prefix_tokens_np, dtype = torch.long, device = self._device)
        obs_prefix_mask_base = torch.as_tensor(obs_prefix_mask_np, dtype = torch.long, device = self._device)
        state_channels_base = torch.as_tensor(state_channels_np, dtype = torch.float32, device = self._device)

        sample_count = int(num_samples)
        batch_size = len(state_strings)
        total = batch_size * sample_count
        expanded_infoset_strs = [
            str(state_strings[batch_idx])
            for batch_idx in range(batch_size)
            for _ in range(sample_count)
        ]
        obs_prefix_tokens = obs_prefix_tokens_base.repeat_interleave(sample_count, dim = 0)
        obs_prefix_mask = obs_prefix_mask_base.repeat_interleave(sample_count, dim = 0)
        state_channels = state_channels_base.repeat_interleave(sample_count, dim = 0)
        belief_input_tokens = torch.full(
            (total, int(self.game.MAX_BELIEF_ROWS), int(self.game.MAX_BELIEF_TOKENS)),
            fill_value = int(self.game.PAD_TOKEN),
            dtype = torch.long,
            device = self._device,
        )
        candidate_tokens = torch.full(
            (total, int(self.game.MAX_BELIEF_ROWS), int(self.game.MAX_BELIEF_CANDIDATES), int(self.game.MAX_BELIEF_TOKENS)),
            fill_value = int(self.game.PAD_TOKEN),
            dtype = torch.long,
            device = self._device,
        )
        candidate_valid = torch.zeros(
            (total, int(self.game.MAX_BELIEF_ROWS), int(self.game.MAX_BELIEF_CANDIDATES)),
            dtype = torch.long,
            device = self._device,
        )
        belief_prefixes = [""] * total
        finished = [False] * total

        generator = torch.Generator(device = self._device if self._device.type != "mps" else "cpu")
        generator.manual_seed(int(seed))

        with torch.inference_mode():
            for row_idx in range(int(self.game.MAX_BELIEF_ROWS)):
                active_indices: list[int] = []
                for sample_idx, is_done in enumerate(finished):
                    if bool(is_done):
                        continue
                    prefix_text = str(belief_prefixes[sample_idx]).strip()
                    if prefix_text:
                        prefix_rows = int(prefix_text.count("|")) + 1
                        if prefix_rows >= int(self.game.MAX_BELIEF_ROWS):
                            finished[sample_idx] = True
                            continue
                    active_indices.append(int(sample_idx))
                if not active_indices:
                    break

                active_infoset_strs = [str(expanded_infoset_strs[sample_idx]) for sample_idx in active_indices]
                active_prefixes = [str(belief_prefixes[sample_idx]) for sample_idx in active_indices]
                decoded = self.game.expand_belief_prefix_strs(active_infoset_strs, active_prefixes)
                row_tokens_active_np = np.asarray(decoded["candidate_tokens"], dtype = np.int64)
                row_valid_active_np = np.asarray(decoded["candidate_valid"], dtype = np.int64)
                belief_str_rows_active = [[str(item) for item in row] for row in decoded["candidate_belief_strs"]]
                expected_row_tokens_shape = (
                    len(active_indices),
                    int(self.game.MAX_BELIEF_CANDIDATES),
                    int(self.game.MAX_BELIEF_TOKENS),
                )
                expected_row_valid_shape = (
                    len(active_indices),
                    int(self.game.MAX_BELIEF_CANDIDATES),
                )
                if row_tokens_active_np.shape != expected_row_tokens_shape:
                    raise RuntimeError(
                        "expand_belief_prefix_strs candidate_tokens shape mismatch during Yomi sampling: "
                        f"{tuple(row_tokens_active_np.shape)} vs {expected_row_tokens_shape}."
                    )
                if row_valid_active_np.shape != expected_row_valid_shape:
                    raise RuntimeError(
                        "expand_belief_prefix_strs candidate_valid shape mismatch during Yomi sampling: "
                        f"{tuple(row_valid_active_np.shape)} vs {expected_row_valid_shape}."
                    )
                if len(belief_str_rows_active) != len(active_indices):
                    raise RuntimeError(
                        "expand_belief_prefix_strs candidate_belief_strs batch mismatch during Yomi sampling: "
                        f"{len(belief_str_rows_active)} vs {len(active_indices)}."
                    )
                for row in belief_str_rows_active:
                    if len(row) != int(self.game.MAX_BELIEF_CANDIDATES):
                        raise RuntimeError(
                            "expand_belief_prefix_strs candidate_belief_strs candidate width mismatch during Yomi sampling."
                        )

                row_tokens_np = np.full(
                    (
                        total,
                        int(self.game.MAX_BELIEF_CANDIDATES),
                        int(self.game.MAX_BELIEF_TOKENS),
                    ),
                    fill_value = int(self.game.PAD_TOKEN),
                    dtype = np.int64,
                )
                row_valid_np = np.zeros(
                    (total, int(self.game.MAX_BELIEF_CANDIDATES)),
                    dtype = np.int64,
                )
                belief_str_rows = [
                    [""] * int(self.game.MAX_BELIEF_CANDIDATES)
                    for _ in range(total)
                ]
                for local_idx, sample_idx in enumerate(active_indices):
                    row_tokens_np[int(sample_idx)] = row_tokens_active_np[int(local_idx)]
                    row_valid_np[int(sample_idx)] = row_valid_active_np[int(local_idx)]
                    belief_str_rows[int(sample_idx)] = list(belief_str_rows_active[int(local_idx)])

                candidate_tokens[:, row_idx] = torch.as_tensor(row_tokens_np, dtype = torch.long, device = self._device)
                candidate_valid[:, row_idx] = torch.as_tensor(row_valid_np, dtype = torch.long, device = self._device)
                active = candidate_valid[:, row_idx].any(dim = -1)
                if not bool(active.any().item()):
                    break

                output = self._model(
                    obs_prefix_tokens = obs_prefix_tokens,
                    obs_prefix_mask = obs_prefix_mask,
                    state_channels = state_channels,
                    belief_input_tokens = belief_input_tokens,
                    candidate_tokens = candidate_tokens,
                    candidate_valid = candidate_valid,
                )
                logits = output.logits[:, row_idx, :].float()
                valid_mask = candidate_valid[:, row_idx].to(dtype = torch.bool)
                if float(temperature) <= 0.0:
                    masked_logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
                    chosen = torch.where(
                        active,
                        masked_logits.argmax(dim = -1),
                        torch.zeros(total, dtype = torch.long, device = self._device),
                    )
                else:
                    masked_logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
                    probs = torch.zeros_like(masked_logits)
                    if bool(active.any().item()):
                        probs_active = torch.softmax(masked_logits[active] / float(temperature), dim = -1)
                        probs[active] = probs_active
                    probs[~active, 0] = 1.0
                    chosen = torch.multinomial(probs, 1, generator = generator).squeeze(-1)

                row_indices = torch.arange(total, device = self._device)
                picked_tokens = candidate_tokens[row_indices, row_idx, chosen]
                picked_tokens = torch.where(
                    active.unsqueeze(-1),
                    picked_tokens,
                    torch.full_like(picked_tokens, fill_value = int(self.game.PAD_TOKEN)),
                )
                if row_idx + 1 < int(self.game.MAX_BELIEF_ROWS):
                    belief_input_tokens[:, row_idx + 1, :] = picked_tokens
                chosen_np = chosen.detach().cpu().numpy()
                active_np = active.detach().cpu().numpy().astype(bool)
                for sample_idx in range(total):
                    if not bool(active_np[sample_idx]):
                        finished[sample_idx] = True
                        continue
                    next_prefix = str(belief_str_rows[sample_idx][int(chosen_np[sample_idx])]).strip()
                    if not next_prefix:
                        raise RuntimeError("expand_belief_prefix_strs returned an empty belief_str for a valid sampled candidate.")
                    belief_prefixes[sample_idx] = next_prefix

        out: list[list[CurrentBeliefSample]] = []
        for batch_idx, infoset_str in enumerate(state_strings):
            row_samples: list[CurrentBeliefSample] = []
            for sample_idx in range(sample_count):
                flat_idx = int(batch_idx * sample_count + sample_idx)
                belief_str = str(belief_prefixes[flat_idx]).strip()
                row_samples.append(
                    CurrentBeliefSample(
                        history_str = str(self.game.determinize_history_str(infoset_str, belief_str))
                    )
                )
            out.append(row_samples)
        return out


__all__ = [
    "CurrentBeliefSample",
    "YomiBeliefModel",
]
