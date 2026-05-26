"""Game-agnostic pending-prefix policy query services for search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from kagami.core.common import softmax_policy_probs, validate_infoset_strs, validate_policy_eval_row
from kagami.core.types import PolicyActionProb, PolicyModel, policy_action_prob_row_from_batch
from kagami.search.horizon import resolve_max_game_horizon


@dataclass(frozen = True)
class PendingPolicyEval:
    action_strs: list[str]
    is_commit: list[bool]
    action_token_rows: np.ndarray
    logits: torch.Tensor


class PendingPolicyQueryService:
    """Pending-prefix policy query and committed-action sampling behavior."""

    def __init__(self, *, game: Any, policy_model: PolicyModel) -> None:
        self._game = game
        self._policy_model = policy_model

    def evaluate_pending_policy(self, *, infoset_str: str, pending_action: str) -> PendingPolicyEval:
        eval_rows = self._policy_model.evaluate_batch(
            infoset_strs = [str(infoset_str)],
            pending_actions = [str(pending_action)],
        )
        if len(eval_rows) != 1:
            raise RuntimeError(f"Expected one pending-policy evaluation, got {len(eval_rows)}.")
        action_strs, is_commit, logits = validate_policy_eval_row(eval_rows[0])

        decoded = self._game.parse_infoset_str(str(infoset_str), str(pending_action))
        decoded_strs_full = [str(action).strip() for action in decoded.last_available_action_strs]
        decoded_commit_full = [bool(flag) for flag in decoded.last_available_actions.is_commit]
        decoded_tokens_full = np.asarray(decoded.last_available_actions.tokens, dtype = np.int64)
        decoded_weights_full = np.asarray(decoded.last_available_actions.weights, dtype = np.float64)
        if int(decoded_tokens_full.ndim) != 2:
            raise RuntimeError(
                "parse_infoset_str returned invalid last_available_actions.tokens shape while materializing action tokens: "
                f"{tuple(decoded_tokens_full.shape)}."
            )
        if (
            len(decoded_strs_full) != int(decoded_tokens_full.shape[0])
            or len(decoded_commit_full) != int(decoded_tokens_full.shape[0])
            or decoded_weights_full.shape[0] != int(decoded_tokens_full.shape[0])
        ):
            raise RuntimeError(
                "parse_infoset_str returned misaligned action fields while materializing action tokens: "
                f"len(strs)={len(decoded_strs_full)} len(commit)={len(decoded_commit_full)} "
                f"rows={int(decoded_tokens_full.shape[0])} len(weights)={int(decoded_weights_full.shape[0])}."
            )

        valid_indices = [idx for idx, weight in enumerate(decoded_weights_full.tolist()) if float(weight) > 0.0]
        decoded_strs = [decoded_strs_full[idx] for idx in valid_indices]
        decoded_commit = [decoded_commit_full[idx] for idx in valid_indices]
        decoded_tokens = decoded_tokens_full[valid_indices]
        if len(decoded_strs) != len(action_strs):
            raise RuntimeError(
                "parse_infoset_str valid action count mismatch with policy evaluator output: "
                f"decoded={len(decoded_strs)} eval={len(action_strs)}."
            )

        for row_idx, (eval_str, eval_commit, dec_str, dec_commit) in enumerate(
            zip(action_strs, is_commit, decoded_strs, decoded_commit, strict = False)
        ):
            if str(eval_str) != str(dec_str):
                raise RuntimeError(
                    "parse_infoset_str and policy evaluator action_str ordering mismatch: "
                    f"row={row_idx} eval={eval_str!r} decoded={dec_str!r}."
                )
            if bool(eval_commit) != bool(dec_commit):
                raise RuntimeError(
                    "parse_infoset_str and policy evaluator is_commit mismatch: "
                    f"row={row_idx} eval={bool(eval_commit)} decoded={bool(dec_commit)}."
                )

        return PendingPolicyEval(
            action_strs = action_strs,
            is_commit = is_commit,
            action_token_rows = decoded_tokens,
            logits = logits,
        )

    def sample_committed_actions_batch(
        self,
        *,
        infoset_strs: Sequence[str],
        action_temp: float,
        seed: int,
        initial_pending_actions: Sequence[str] | None = None,
    ) -> list[str]:
        state_strings = validate_infoset_strs(infoset_strs)
        if not state_strings:
            return []

        if initial_pending_actions is None:
            pending_actions = [""] * len(state_strings)
        else:
            pending_actions = [str(item).strip() for item in initial_pending_actions]
            if len(pending_actions) != len(state_strings):
                raise ValueError("initial_pending_actions must match infoset_strs.")

        chooser = np.random.default_rng(int(seed))
        unresolved = list(range(len(state_strings)))
        committed_actions = ["" for _ in state_strings]
        depth_limit = resolve_max_game_horizon(self._game)

        evaluate_batch_fn = getattr(self._policy_model, "evaluate_batch", None)
        has_evaluate_batch = callable(evaluate_batch_fn)

        for depth in range(depth_limit + 1):
            if not unresolved:
                return committed_actions

            unresolved_infosets = [state_strings[idx] for idx in unresolved]
            unresolved_pending = [pending_actions[idx] for idx in unresolved]
            unique_query_by_key: dict[tuple[str, str], int] = {}
            unique_infosets: list[str] = []
            unique_pending: list[str] = []
            for infoset_str, pending_action in zip(unresolved_infosets, unresolved_pending, strict = False):
                key = (str(infoset_str), str(pending_action))
                if key in unique_query_by_key:
                    continue
                unique_query_by_key[key] = len(unique_infosets)
                unique_infosets.append(str(infoset_str))
                unique_pending.append(str(pending_action))

            next_unresolved: list[int] = []
            if has_evaluate_batch:
                eval_rows = evaluate_batch_fn(infoset_strs = unique_infosets, pending_actions = unique_pending)
                if len(eval_rows) != len(unique_infosets):
                    raise RuntimeError(
                        "Pending-action evaluate_batch returned unexpected row count: "
                        f"expected={len(unique_infosets)}, got={len(eval_rows)}."
                    )
                sampled_rows: list[tuple[list[str], np.ndarray, list[bool]]] = []
                for eval_row in eval_rows:
                    action_strs = [str(action).strip() for action in eval_row.action_strs]
                    if not action_strs:
                        raise RuntimeError("Pending-action row contains no actions.")
                    if any(not action for action in action_strs):
                        raise RuntimeError("Pending-action row contains an empty semantic action string.")
                    is_commit_row = [bool(flag) for flag in eval_row.is_commit]
                    logits = eval_row.logits.detach().to(dtype = torch.float64)
                    if int(logits.ndim) != 1 or int(logits.shape[0]) != len(action_strs):
                        raise RuntimeError(
                            "Pending-action row logits shape mismatch: "
                            f"len(action_strs)={len(action_strs)}, shape={tuple(logits.shape)}."
                        )
                    probs = softmax_policy_probs(logits, temperature = float(action_temp))
                    if int(probs.ndim) != 1 or int(probs.shape[0]) != len(action_strs):
                        raise RuntimeError(
                            "Pending-action row probability shape mismatch after softmax: "
                            f"len(action_strs)={len(action_strs)}, shape={tuple(probs.shape)}."
                        )
                    probs_np = probs.detach().to(dtype = torch.float64, device = torch.device("cpu")).numpy()
                    total = float(np.sum(probs_np))
                    if not math.isfinite(total) or total <= 0.0:
                        raise RuntimeError("Pending-action row has non-positive or non-finite probability total.")
                    sampled_rows.append((action_strs, probs_np / total, is_commit_row))

                for global_idx in unresolved:
                    key = (str(state_strings[global_idx]), str(pending_actions[global_idx]))
                    row_idx = unique_query_by_key.get(key, -1)
                    if row_idx < 0:
                        raise RuntimeError("Pending-action dedup lookup failed while sampling committed actions.")
                    action_strs, probs_np, is_commit_row = sampled_rows[row_idx]
                    chosen_idx = int(chooser.choice(len(action_strs), p = probs_np))
                    chosen_action = str(action_strs[chosen_idx])
                    chosen_commit = bool(is_commit_row[chosen_idx])
                    if bool(chosen_commit):
                        committed_actions[global_idx] = str(chosen_action)
                    else:
                        pending_actions[global_idx] = str(chosen_action)
                        next_unresolved.append(global_idx)
            else:
                batch = self._policy_model.policy_batch(
                    infoset_strs = unique_infosets,
                    pending_actions = unique_pending,
                    action_temps = [float(action_temp)] * len(unique_infosets),
                )
                if len(batch.action_strs_batch) != len(unique_infosets):
                    raise RuntimeError(
                        "Pending-action policy batch returned unexpected row count: "
                        f"expected={len(unique_infosets)}, got={len(batch.action_strs_batch)}."
                    )
                dedup_rows = {
                    (str(unique_infosets[row_idx]), str(unique_pending[row_idx])): policy_action_prob_row_from_batch(
                        batch,
                        row_idx = row_idx,
                    )
                    for row_idx in range(len(unique_infosets))
                }
                for global_idx in unresolved:
                    key = (str(state_strings[global_idx]), str(pending_actions[global_idx]))
                    row = dedup_rows.get(key)
                    if row is None:
                        raise RuntimeError("Pending-action dedup lookup failed while sampling committed actions.")
                    chosen_action, chosen_commit = self.sample_policy_action_from_row(row = row, chooser = chooser)
                    if bool(chosen_commit):
                        committed_actions[global_idx] = str(chosen_action)
                    else:
                        pending_actions[global_idx] = str(chosen_action)
                        next_unresolved.append(global_idx)

            unresolved = next_unresolved
            if unresolved and depth == depth_limit:
                raise RuntimeError(
                    "Pending-action expansion exceeded MAX_GAME_HORIZON during committed-action sampling; "
                    f"depth_limit={depth_limit} unresolved_rows={len(unresolved)}"
                )

        raise RuntimeError("Pending-action completion loop terminated unexpectedly.")

    @staticmethod
    def sample_policy_action_from_row(*, row: PolicyActionProb, chooser: np.random.Generator) -> tuple[str, bool]:
        if not row.action_strs:
            raise RuntimeError("Pending-action row contains no actions.")
        probs = row.probs.detach().to(dtype = torch.float64, device = torch.device("cpu"))
        is_commit = row.is_commit.detach().to(dtype = torch.bool, device = torch.device("cpu"))
        if int(probs.ndim) != 1 or int(probs.shape[0]) != len(row.action_strs):
            raise RuntimeError(
                "Pending-action row probability shape mismatch: "
                f"len(action_strs)={len(row.action_strs)}, shape={tuple(probs.shape)}."
            )
        total = float(torch.sum(probs).item())
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("Pending-action row has non-positive or non-finite probability total.")
        normalized = (probs / total).detach().cpu().numpy()
        chosen_idx = int(chooser.choice(len(row.action_strs), p = normalized))
        return str(row.action_strs[chosen_idx]), bool(is_commit[chosen_idx].item())


__all__ = [
    "PendingPolicyEval",
    "PendingPolicyQueryService",
]
