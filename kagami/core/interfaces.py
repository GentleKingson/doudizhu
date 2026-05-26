"""Game-agnostic inference interfaces for one-step action distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import math
from pathlib import Path
from typing import Sequence

import torch

from kagami.core.common import validate_infoset_strs, validate_temperature
from kagami.core.types import (
    PolicyActionProb,
    PolicyActionProbBatch,
    PolicyModel,
    policy_action_prob_row_from_batch,
)


class InferenceEngine(ABC):
    """Common engine interface for infoset-to-policy-action inference."""

    def __init__(self, *, game: object, action_temp: float = 0.0, sample_seed: int = 0) -> None:
        if game is None:
            raise ValueError("game must be non-null.")
        self._game = game
        self._action_temp = float(action_temp)
        validate_temperature(self._action_temp, name = "action_temp")
        self._sample_seed = int(sample_seed)
        self._backend_kind: str = "marvel"
        self._torch_rng = torch.Generator(device = "cpu")
        self._torch_rng.manual_seed(int(self._sample_seed))

    def bind_marvel_model(
        self,
        *,
        checkpoint: str | Path,
        device: str | torch.device = "cuda",
    ) -> None:
        from kagami.connectors.marvel import MarvelModel

        policy_model: PolicyModel = MarvelModel(
            checkpoint = checkpoint,
            device = device,
        )
        model_game = getattr(policy_model, "_game", None)
        if model_game is not None and model_game is not self._game:
            if self._can_rebind_implicit_game(model_game):
                self._game = model_game
            else:
                raise ValueError("MARVEL checkpoint game does not match this engine's bound game.")
        self._policy_model = policy_model
        self._backend_kind = "marvel"

    def bind_marvel_critic(
        self,
        *,
        checkpoint: str | Path,
        device: str | torch.device = "cuda",
    ) -> None:
        from marvel.model import CentralCritic
        from marvel.runtime.checkpoints import CheckpointManager, CheckpointStore

        resolved_checkpoint = CheckpointStore.resolve_checkpoint_file_path(checkpoint)
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

        device_obj = torch.device(str(device))
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            device_obj = torch.device("cpu")

        critic_model = CheckpointManager.load_model_from_checkpoint(
            resolved_checkpoint,
            model_cls = CentralCritic,
            device = device_obj,
            model_kwargs = {"output_mode": "qvalue"},
        )
        train_cfg = getattr(critic_model, "train_config", None)
        model_game = getattr(train_cfg, "game", None)
        if model_game is not None and model_game is not self._game:
            if self._can_rebind_implicit_game(model_game):
                self._game = model_game
            else:
                raise ValueError("MARVEL critic checkpoint game does not match this engine's bound game.")

        self._critic_model = critic_model
        self._critic_device = device_obj

    def bind_yomi_model(
        self,
        *,
        checkpoint: str | Path,
        device: str | torch.device = "cuda",
    ) -> None:
        from kagami.connectors.yomi import YomiBeliefModel

        belief_model = YomiBeliefModel.load_from_checkpoint(
            checkpoint = checkpoint,
            device = device,
            game = self._game,
        )
        belief_game = getattr(belief_model, "game", None)
        if belief_game is not None and belief_game is not self._game:
            if self._can_rebind_implicit_game(belief_game):
                self._game = belief_game
            else:
                raise ValueError("Yomi checkpoint game does not match this engine's bound game.")
        self._belief_model = belief_model

    def _can_rebind_implicit_game(self, model_game: object) -> bool:
        if model_game is None:
            return False
        if not bool(getattr(self, "_allow_implicit_game_rebind", False)):
            return False
        if getattr(self, "_policy_model", None) is not None:
            return False
        if getattr(self, "_belief_model", None) is not None:
            return False
        return True

    def infer(
        self,
        infoset_str: str,
        pending_action: str = "",
    ) -> PolicyActionProb:
        rows = self.infer_batch(
            [infoset_str],
            [pending_action],
        )
        if len(rows.action_strs_batch) != 1:
            raise RuntimeError(f"Expected one policy-action row, got {len(rows.action_strs_batch)}.")
        return policy_action_prob_row_from_batch(rows, row_idx = 0)

    @abstractmethod
    def infer_batch(
        self,
        infoset_strs: Sequence[str],
        pending_actions: Sequence[str] | None = None,
    ) -> PolicyActionProbBatch:
        ...

    def act(
        self,
        infoset_str: str,
    ) -> str:
        out = self.act_batch([str(infoset_str)])
        if len(out) != 1:
            raise RuntimeError(f"Expected one committed action, got {len(out)}.")
        return str(out[0])

    def act_batch(
        self,
        infoset_strs: Sequence[str],
    ) -> list[str]:
        state_strings = validate_infoset_strs(infoset_strs)
        if not state_strings:
            return []

        depth_limit = self._max_game_horizon()
        pending_actions = ["" for _ in state_strings]
        unresolved = list(range(len(state_strings)))
        committed_actions = ["" for _ in state_strings]

        for depth in range(depth_limit + 1):
            if not unresolved:
                return committed_actions
            batch = self.infer_batch(
                infoset_strs = [state_strings[idx] for idx in unresolved],
                pending_actions = [pending_actions[idx] for idx in unresolved],
            )
            if len(batch.action_strs_batch) != len(unresolved):
                raise RuntimeError(
                    "infer_batch returned unexpected row count while sampling committed actions: "
                    f"expected={len(unresolved)}, got={len(batch.action_strs_batch)}."
                )

            next_unresolved: list[int] = []
            for local_idx, global_idx in enumerate(unresolved):
                row = policy_action_prob_row_from_batch(batch, row_idx = local_idx)
                action_str, is_commit = self._sample_policy_action(row)
                if bool(is_commit):
                    committed_actions[global_idx] = str(action_str)
                else:
                    pending_actions[global_idx] = str(action_str)
                    next_unresolved.append(global_idx)

            unresolved = next_unresolved
            if unresolved and depth == depth_limit:
                raise RuntimeError(
                    "Committed-action sampling exceeded MAX_GAME_HORIZON before reaching a commit; "
                    f"depth_limit={depth_limit} unresolved_rows={len(unresolved)}"
                )

        raise RuntimeError("Committed-action sampling loop terminated unexpectedly.")

    def prob(
        self,
        infoset_str: str,
        committed_action: str,
    ) -> float:
        probs = self.prob_batch(
            [str(infoset_str)],
            [str(committed_action)],
        )
        if len(probs) != 1:
            raise RuntimeError(f"Expected one committed-action probability, got {len(probs)}.")
        return float(probs[0])

    def prob_batch(
        self,
        infoset_strs: Sequence[str],
        committed_actions: Sequence[str],
    ) -> list[float]:
        state_strings = validate_infoset_strs(infoset_strs)
        target_actions = [str(action).strip() for action in committed_actions]
        if any(not action for action in target_actions):
            raise ValueError("committed_actions must not contain empty items.")
        if len(target_actions) != len(state_strings):
            raise ValueError("committed_actions must match the number of infoset_strs.")
        if not state_strings:
            return []

        depth_limit = self._max_game_horizon()
        target_keys = [action.lower() for action in target_actions]

        target_mass = [0.0 for _ in state_strings]
        frontiers: list[dict[str, float]] = [{"": 1.0} for _ in state_strings]

        for depth in range(depth_limit + 1):
            next_frontiers: list[dict[str, float]] = [defaultdict(float) for _ in state_strings]
            batch_infosets: list[str] = []
            batch_pending: list[str] = []
            batch_parent_mass: list[float] = []
            batch_owner: list[int] = []
            had_any_eval = False

            def _flush_eval_batch() -> None:
                if not batch_infosets:
                    return
                rows = self.infer_batch(
                    infoset_strs = batch_infosets,
                    pending_actions = batch_pending,
                )
                if len(rows.action_strs_batch) != len(batch_infosets):
                    raise RuntimeError(
                        "infer_batch returned unexpected row count while evaluating committed-action probability: "
                        f"expected={len(batch_infosets)}, got={len(rows.action_strs_batch)}."
                    )
                for row_idx in range(len(rows.action_strs_batch)):
                    owner = int(batch_owner[row_idx])
                    parent_mass = float(batch_parent_mass[row_idx])
                    row = policy_action_prob_row_from_batch(rows, row_idx = row_idx)
                    for action_idx, action in enumerate(row.action_strs):
                        action_str = str(action).strip()
                        if not action_str:
                            raise RuntimeError(
                                "infer_batch returned an empty action_str while evaluating committed-action probability."
                            )
                        prob = float(row.probs[action_idx].item())
                        if not math.isfinite(prob) or prob < 0.0:
                            raise RuntimeError(
                                "infer_batch returned invalid probability while evaluating committed-action probability: "
                                f"action={action_str!r} prob={prob!r}"
                            )
                        mass = parent_mass * prob
                        if mass <= 0.0:
                            continue
                        is_commit = bool(row.is_commit[action_idx].item())
                        if is_commit:
                            if action_str.lower() == target_keys[owner]:
                                target_mass[owner] += float(mass)
                        else:
                            next_frontiers[owner][action_str] += float(mass)
                batch_infosets.clear()
                batch_pending.clear()
                batch_parent_mass.clear()
                batch_owner.clear()

            for owner_idx, frontier in enumerate(frontiers):
                for pending_action, parent_mass in frontier.items():
                    if float(parent_mass) <= 0.0:
                        continue
                    had_any_eval = True
                    batch_infosets.append(state_strings[owner_idx])
                    batch_pending.append(str(pending_action))
                    batch_parent_mass.append(float(parent_mass))
                    batch_owner.append(owner_idx)

            if not had_any_eval:
                return target_mass

            _flush_eval_batch()
            frontiers = [dict(frontier) for frontier in next_frontiers]
            if any(frontiers) and depth == depth_limit:
                remaining = sum(len(frontier) for frontier in frontiers)
                raise RuntimeError(
                    "Committed-action probability expansion exceeded MAX_GAME_HORIZON; "
                    f"depth_limit={depth_limit} frontier_size={remaining}"
                )

        return target_mass

    def _max_game_horizon(self) -> int:
        raw = getattr(self._game, "MAX_GAME_HORIZON", None)
        if raw is None:
            raise RuntimeError("Bound game is missing required MAX_GAME_HORIZON.")
        horizon = int(raw)
        if horizon <= 0:
            raise RuntimeError(f"MAX_GAME_HORIZON must be positive, got {horizon}.")
        return horizon

    def _sample_policy_action(self, row: PolicyActionProb) -> tuple[str, bool]:
        action_count = len(row.action_strs)
        if action_count <= 0:
            raise RuntimeError("Engine produced an empty action distribution.")
        probs = row.probs.detach().to(dtype = torch.float64, device = torch.device("cpu"))
        is_commit = row.is_commit.detach().to(dtype = torch.bool, device = torch.device("cpu"))
        if int(probs.ndim) != 1 or int(probs.shape[0]) != action_count:
            raise RuntimeError(
                "Policy action row shape mismatch for probs: "
                f"len(action_strs)={action_count}, shape={tuple(probs.shape)}."
            )
        if int(is_commit.ndim) != 1 or int(is_commit.shape[0]) != action_count:
            raise RuntimeError(
                "Policy action row shape mismatch for is_commit: "
                f"len(action_strs)={action_count}, shape={tuple(is_commit.shape)}."
            )
        if not bool(torch.isfinite(probs).all().item()) or bool((probs < 0.0).any().item()):
            raise RuntimeError("Policy action row contains invalid probabilities.")
        total = float(torch.sum(probs).item())
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("Policy action row has non-positive or non-finite total probability.")
        probs = probs / total
        chosen_idx = int(torch.multinomial(probs, num_samples = 1, replacement = True, generator = self._torch_rng).item())
        return str(row.action_strs[chosen_idx]), bool(is_commit[chosen_idx].item())

__all__ = [
    "InferenceEngine",
]
