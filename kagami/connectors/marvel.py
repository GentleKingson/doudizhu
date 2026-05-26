"""MARVEL-backed step evaluator and semantic-to-logit adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from kagami.core.common import softmax_policy_probs, validate_infoset_strs, validate_policy_eval_row, validate_temperature

from kagami.core.types import (
    PolicyActionProb,
    PolicyActionProbBatch,
    PolicyEval,
    PolicyModel,
    policy_action_prob_batch_from_rows,
    policy_action_prob_row_from_batch,
)
from marvel.model import Agent
from marvel.model.tensordict import TensorDict
from marvel.runtime.checkpoints import CheckpointManager, CheckpointStore
from marvel.runtime.rollout_conversion import history_batch_to_rollout


@dataclass(frozen = True)
class ActionLogits:
    """One-step MARVEL policy evaluation for an infoset string."""

    action_strs: list[str]
    is_commit: list[bool]
    logits: "torch.Tensor"  # [A_valid] float32


class MarvelModel(PolicyModel):
    backend_kind: str = "marvel"

    """Load a MARVEL checkpoint as a bound Kagami step evaluator."""

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        device: str | "torch.device" = "cuda",
        _agent: Any | None = None,
        _game: Any | None = None,
        _device: "torch.device" | None = None,
    ) -> None:
        if checkpoint is None:
            if _agent is None or _game is None or _device is None:
                raise ValueError("MarvelModel requires checkpoint=... or internal bound components.")
            self._agent = _agent
            self._game = _game
            self._device = _device
            return
        if _agent is not None or _game is not None or _device is not None:
            raise ValueError("checkpoint cannot be combined with internal bound-component overrides.")

        resolved_checkpoint = CheckpointStore.resolve_checkpoint_file_path(checkpoint)
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

        device_obj = torch.device(str(device))
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            device_obj = torch.device("cpu")

        agent = CheckpointManager.load_model_from_checkpoint(
            resolved_checkpoint,
            model_cls = Agent,
            device = device_obj,
        )
        agent.eval()
        train_cfg = getattr(agent, "train_config", None)
        if train_cfg is None:
            raise RuntimeError("Checkpoint does not include TrainConfig; cannot resolve game module.")

        self._agent = agent
        self._game = train_cfg.game
        self._device = device_obj

    def evaluate(self, *, infoset_str: str, pending_action: str) -> PolicyEval:
        results = self.evaluate_batch(
            infoset_strs = [infoset_str],
            pending_actions = [pending_action],
        )
        if len(results) != 1:
            raise RuntimeError(f"Expected one evaluation result, got {len(results)}.")
        return results[0]

    def policy(
        self,
        *,
        infoset_str: str,
        pending_action: str,
        action_temp: float,
    ) -> PolicyActionProb:
        batch = self.policy_batch(
            infoset_strs = [infoset_str],
            pending_actions = [pending_action],
            action_temps = [float(action_temp)],
        )
        if len(batch.action_strs_batch) != 1:
            raise RuntimeError(f"Expected one policy-action row, got {len(batch.action_strs_batch)}.")
        return policy_action_prob_row_from_batch(batch, row_idx = 0)

    def policy_batch(
        self,
        *,
        infoset_strs: Sequence[str],
        pending_actions: Sequence[str],
        action_temps: Sequence[float],
    ) -> PolicyActionProbBatch:
        state_strings = validate_infoset_strs(infoset_strs)
        pending_list = [str(item).strip() for item in pending_actions]
        if len(pending_list) != len(state_strings):
            raise ValueError("pending_actions must match the number of infoset_strs.")
        temperatures = [float(value) for value in action_temps]
        if len(temperatures) != len(state_strings):
            raise ValueError("action_temps must match the number of infoset_strs.")
        for temp in temperatures:
            validate_temperature(float(temp), name = "action_temps")

        eval_rows = self.evaluate_batch(
            infoset_strs = state_strings,
            pending_actions = pending_list,
        )
        rows: list[PolicyActionProb] = []
        for row_idx, eval_row in enumerate(eval_rows):
            action_strs, is_commit, logits = validate_policy_eval_row(eval_row)
            probs = softmax_policy_probs(logits, temperature = float(temperatures[row_idx]))
            if tuple(probs.shape) != (len(action_strs),):
                raise RuntimeError(
                    "Policy probability row shape mismatch after softmax: "
                    f"expected {(len(action_strs),)}, got {tuple(probs.shape)}."
                )
            tensors = TensorDict(
                {
                    "probs": probs.detach().to(dtype = torch.float64),
                    "is_commit": torch.as_tensor(is_commit, dtype = torch.bool, device = probs.device),
                },
                batch_size = torch.Size([len(action_strs)]),
                device = probs.device,
            )
            rows.append(PolicyActionProb(action_strs = action_strs, tensors = tensors))
        return policy_action_prob_batch_from_rows(rows)

    def evaluate_batch(self, *, infoset_strs: Sequence[str], pending_actions: Sequence[str]) -> list[PolicyEval]:
        if len(pending_actions) != len(infoset_strs):
            raise ValueError("pending_actions must match the number of infoset_strs.")
        if not infoset_strs:
            return []

        state_strings = [str(infoset_str) for infoset_str in infoset_strs]
        pending_list = [str(pending_action) for pending_action in pending_actions]
        batch = self._game.parse_infoset_strs(state_strings, pending_list)
        infosets = list(batch)
        rollout, num_steps_np, player_ids_np, is_commit_np, _avail_weights_np = history_batch_to_rollout(
            history_batch = batch,
            device = self._device,
            pad_token_id = int(self._game.PAD_TOKEN),
            prefix_mode = "ignore",
        )
        batch_size = len(infosets)
        num_steps_t = torch.as_tensor(num_steps_np, dtype = torch.long, device = self._device)
        if batch_size != int(num_steps_t.shape[0]):
            raise RuntimeError("parse_infoset_strs returned inconsistent history rows.")

        with torch.inference_mode():
            probs = self._agent(rollout = rollout, use_cache = False).avail_probs

        device = probs.device
        num_steps_t = num_steps_t.to(device = device)
        player_ids_t = torch.as_tensor(player_ids_np, dtype = torch.long, device = device)
        is_commit_t = torch.as_tensor(is_commit_np, dtype = torch.bool, device = device)

        last_steps_t = num_steps_t - 1
        invalid_steps = (last_steps_t < 0) | (last_steps_t >= int(probs.shape[1]))
        if bool(invalid_steps.any().item()):
            batch_idx = int(torch.nonzero(invalid_steps, as_tuple = False)[0].item())
            raise RuntimeError(
                f"HistoryBatch.num_steps contains an out-of-range timestep for batch_index={batch_idx}."
            )

        batch_idx_t = torch.arange(batch_size, device = device)
        probs_last = probs[batch_idx_t, last_steps_t]
        valid_mask = rollout["avail_weights"][batch_idx_t, last_steps_t] > 0

        action_strings_rows = [self._last_available_action_strings(infoset = infoset) for infoset in infosets]
        action_lengths_t = torch.as_tensor(
            [len(row) for row in action_strings_rows],
            dtype = torch.long,
            device = device,
        )
        action_indices_t = torch.arange(valid_mask.shape[1], dtype = torch.long, device = device)
        action_overflow = valid_mask & (action_indices_t.unsqueeze(0) >= action_lengths_t.unsqueeze(1))
        if bool(action_overflow.any().item()):
            batch_idx = int(torch.nonzero(action_overflow.any(dim = 1), as_tuple = False)[0].item())
            raise RuntimeError(
                "infoset.last_available_action_strs do not align with policy action indices "
                f"(batch_index={batch_idx}, pending_action={pending_list[batch_idx]!r}, "
                f"infoset_str={state_strings[batch_idx]!r})."
            )

        acting_players_t = torch.as_tensor(
            [int(infoset.last_available_actions.player_id) for infoset in infosets],
            dtype = torch.long,
            device = device,
        )
        player_mismatch = player_ids_t[batch_idx_t, last_steps_t] != acting_players_t
        if bool(player_mismatch.any().item()):
            batch_idx = int(torch.nonzero(player_mismatch, as_tuple = False)[0].item())
            raise RuntimeError(
                "Final-step player_id mismatch between HistoryBatch.player_ids final row and last_available_actions.player_id "
                f"(batch_index={batch_idx}, pending_action={pending_list[batch_idx]!r}, "
                f"infoset_str={state_strings[batch_idx]!r})."
            )

        valid_counts_t = valid_mask.sum(dim = 1, dtype = torch.long)
        no_valid = valid_counts_t == 0
        if bool(no_valid.any().item()):
            batch_idx = int(torch.nonzero(no_valid, as_tuple = False)[0].item())
            raise RuntimeError(
                f"No available actions for acting player (player_id={int(acting_players_t[batch_idx].item())}, batch_index={batch_idx}, "
                f"pending_action={pending_list[batch_idx]!r}, infoset_str={state_strings[batch_idx]!r})."
            )

        counts = [int(value) for value in valid_counts_t.detach().cpu().tolist()]
        flat_valid_logits = probs_last.float().clamp(min = 1e-9).log()[valid_mask]
        logits_rows = flat_valid_logits.split(counts)
        flat_valid_indices_t = action_indices_t.unsqueeze(0).expand_as(valid_mask)[valid_mask]
        index_rows_t = flat_valid_indices_t.split(counts)

        return [
            ActionLogits(
                action_strs = [
                    action_strings_rows[batch_idx][int(idx)]
                    for idx in valid_indices.detach().cpu().tolist()
                ],
                is_commit = [
                    bool(is_commit_t[batch_idx, int(idx)].item())
                    for idx in valid_indices.detach().cpu().tolist()
                ],
                logits = logits_rows[batch_idx],
            )
            for batch_idx, valid_indices in enumerate(index_rows_t)
        ]

    @staticmethod
    def _last_available_action_strings(*, infoset: Any) -> list[str]:
        raw_strings = getattr(infoset, "last_available_action_strs", None)
        if raw_strings is None:
            raise RuntimeError("parse_infoset_str output does not expose infoset.last_available_action_strs.")
        action_tokens = infoset.last_available_actions.tokens
        if action_tokens.ndim != 2:
            raise RuntimeError("infoset.last_available_actions.tokens must have shape [A, MAX_ACTION_TOKENS].")
        out = [str(action_str) for action_str in raw_strings]
        if len(out) != int(action_tokens.shape[0]):
            raise RuntimeError(
                "infoset.last_available_action_strs must align with infoset.last_available_actions.tokens along axis 0."
            )
        return out


MarvelConnector = MarvelModel


__all__ = [
    "ActionLogits",
    "MarvelConnector",
    "MarvelModel",
]
