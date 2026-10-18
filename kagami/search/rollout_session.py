"""Game-agnostic search rollout session adapter over MARVEL Sampler."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from kagami.search._sampler_bootstrap import (
    build_sampler,
    prefill_sampler_rollout_buffer_from_history_strs,
    validate_prefilled_rollout_timeline,
)
from marvel.runtime.sampler import Sampler


class SearchRolloutSession:
    """Prepare/validate one search rollout context and delegate stepping to MARVEL Sampler."""

    def __init__(
        self,
        *,
        game: Any,
        policy_model: Any,
        root_player_id: int,
        rollout_temp: float,
        max_horizon: int,
    ) -> None:
        self._game = game
        self._policy_model = policy_model
        self._root_player_id = int(root_player_id)
        self._rollout_temp = float(rollout_temp)
        self._max_horizon = int(max_horizon)
        if self._max_horizon <= 0:
            raise RuntimeError(f"max_horizon must be positive, got {self._max_horizon}.")

    def rollout_selected_action_rows(
        self,
        *,
        root_history_strs: Sequence[str],
        pending_action: str,
        action_token_rows: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """Rollout a batch of determinized-root trajectories taking a specified first action.

        Handoff contract: ``envs`` is kept at the decision state (``root_step``), and the
        root action is driven through the first ``rollout_step`` via
        ``forced_action_indices`` rather than via a pre-call to ``envs.step(root_actions)``.
        This matches the contract enforced by
        :func:`kagami.search._sampler_bootstrap.prefill_sampler_rollout_buffer_from_infoset_strs`
        — the env state must not be advanced past the state described by the
        ``history_strs`` argument before the Sampler is handed off.

        Known residual issue (v1, 2026-04): even with the correct handoff, rollouts
        produced through this path score ~3 reward points below the eval-path baseline
        on Hanabi self-play (observed: mean 21.5 vs eval 24.57 over 600 random-chance
        rollouts on the same initial states). A fresh Sampler built via
        ``EnvBatch.reset(seed)`` matches the eval exactly (24.57); only the
        ``reconstruct_envs(history_strs, pending, seed)`` path exhibits the regression.
        The root cause likely lies in how ``reconstruct_envs`` initializes env-internal
        state (RNG, residual deck, etc.) vs what a freshly-reset-and-replayed env would
        produce. See ``/home/fanzy/hanabi_diagnostic/`` on the Schmidt cluster for the
        repro scripts and ``docs/kagami-oop-architecture.md`` for the tracked follow-up.
        Until diagnosed, consumers that need honest V_π estimates (e.g. search benchmarks)
        should prefer an eval-path rollout built on
        ``evaluations.<game>.host.*Env + policy_model.policy_batch``.
        """
        state_histories = [str(item) for item in root_history_strs]
        if not state_histories:
            raise RuntimeError("Update-equivalent rollout requires at least one determinized state.")

        selected_action_rows = np.asarray(action_token_rows, dtype = np.int64)
        if int(selected_action_rows.ndim) != 2:
            raise RuntimeError(
                "action_token_rows must have shape [B, MAX_ACTION_TOKENS] when rolling out pending histories: "
                f"{tuple(selected_action_rows.shape)}."
            )
        if int(selected_action_rows.shape[0]) != len(state_histories):
            raise RuntimeError(
                "action_token_rows batch size must match root_history_strs during pending rollout: "
                f"len(histories)={len(state_histories)} rows={int(selected_action_rows.shape[0])}."
            )
        if not callable(getattr(self._game, "reconstruct_envs", None)):
            raise RuntimeError("Update-equivalent search requires game.reconstruct_envs(...).")

        agent_model = getattr(self._policy_model, "_agent")
        if agent_model is None:
            raise RuntimeError("Env-native rollout requires a bound MARVEL agent model.")
        if getattr(self._policy_model, "_device", None) is None:
            raise RuntimeError("Env-native rollout requires policy_model._device.")

        pending = str(pending_action)
        envs, root_step = self._game.reconstruct_envs(
            state_histories,
            pending_actions = [pending] * len(state_histories),
            seed = int(seed),
        )
        # DO NOT step envs with root_actions here — Sampler must receive envs at the
        # decision state (root_step) so that buffer[num_steps] and parsed_envs_step are
        # aligned. The root action is applied inside the first rollout_step via
        # forced_action_indices, which lets the normal buffer-update path record the
        # root transition at the correct trajectory position.
        root_action_indices = self.select_legal_action_indices(
            envs_step = root_step,
            action_token_rows = selected_action_rows,
        )

        sampler = build_sampler(
            game = self._game,
            policy_model = self._policy_model,
            rollout_temp = float(self._rollout_temp),
            envs = envs,
            envs_step = root_step,
            collect_batch_size = 0,
        )
        prefill_sampler_rollout_buffer_from_history_strs(
            game = self._game,
            sampler = sampler,
            history_strs = state_histories,
            pending_action = pending,
        )
        validate_prefilled_rollout_timeline(
            sampler = sampler,
            next_step = root_step,
            max_horizon = self._max_horizon,
        )
        return self.run_sampler_rollout(
            sampler = sampler,
            agent_model = agent_model,
            # Keep search sampling independent of process startup and request order.
            generator = torch.Generator(
                device = sampler.device if sampler.device.type != "mps" else "cpu",
            ).manual_seed(int(seed)),
            forced_first_action_indices = torch.as_tensor(
                root_action_indices, dtype = torch.long,
                device = torch.device(getattr(self._policy_model, "_device")),
            ),
        )

    def select_legal_action_indices(
        self,
        *,
        envs_step: Any,
        action_token_rows: np.ndarray,
    ) -> np.ndarray:
        """Return per-env indices into ``envs_step.available_actions`` that match the
        requested ``action_token_rows``. Raises if a requested token row is not among
        the legal actions for that env.
        """
        legal_tokens = np.asarray(envs_step.available_actions.tokens, dtype = np.int64)
        legal_weights = np.asarray(envs_step.available_actions.weights, dtype = np.float64)
        if int(legal_tokens.ndim) != 3:
            raise RuntimeError(
                "reconstruct_envs returned invalid available_actions.tokens shape for root-action selection: "
                f"{tuple(legal_tokens.shape)}."
            )
        if legal_weights.shape[:2] != legal_tokens.shape[:2]:
            raise RuntimeError(
                "reconstruct_envs returned misaligned legal-action tensors for root-action selection: "
                f"tokens={tuple(legal_tokens.shape)} weights={tuple(legal_weights.shape)}."
            )

        batch_size = int(legal_tokens.shape[0])
        requested = np.asarray(action_token_rows, dtype = np.int64)
        if int(requested.ndim) != 2:
            raise RuntimeError(
                "action_token_rows must have shape [B, MAX_ACTION_TOKENS] for policy-action selection: "
                f"{tuple(requested.shape)}."
            )
        if int(requested.shape[0]) != batch_size or int(requested.shape[1]) != int(legal_tokens.shape[2]):
            raise RuntimeError(
                "action_token_rows shape mismatch for policy-action selection: "
                f"expected={(batch_size, int(legal_tokens.shape[2]))}, got={tuple(requested.shape)}."
            )

        pad_token = int(self._game.PAD_TOKEN)
        indices = np.full((batch_size,), fill_value = -1, dtype = np.int64)
        for row_idx in range(batch_size):
            wanted = tuple(int(tok) for tok in requested[row_idx].tolist() if int(tok) != pad_token)
            if not wanted:
                raise RuntimeError(
                    "Root-action selection requires non-empty token rows: "
                    f"row={row_idx}."
                )
            row_lookup: dict[tuple[int, ...], int] = {}
            for action_idx in range(int(legal_tokens.shape[1])):
                if float(legal_weights[row_idx, action_idx]) <= 0.0:
                    continue
                token_tuple = tuple(int(tok) for tok in legal_tokens[row_idx, action_idx].tolist() if int(tok) != pad_token)
                row_lookup[token_tuple] = int(action_idx)
            action_idx = row_lookup.get(wanted, -1)
            if action_idx < 0:
                raise RuntimeError(
                    "Selected action token row is not legal at reconstructed decision state: "
                    f"row={row_idx} tokens={wanted!r}."
                )
            indices[row_idx] = int(action_idx)
        return indices

    def select_legal_action_token_rows(
        self,
        *,
        envs_step: Any,
        action_token_rows: np.ndarray,
    ) -> np.ndarray:
        legal_tokens = np.asarray(envs_step.available_actions.tokens, dtype = np.int64)
        legal_weights = np.asarray(envs_step.available_actions.weights, dtype = np.float64)
        if int(legal_tokens.ndim) != 3:
            raise RuntimeError(
                "reconstruct_envs returned invalid available_actions.tokens shape for root-action selection: "
                f"{tuple(legal_tokens.shape)}."
            )
        if legal_weights.shape[:2] != legal_tokens.shape[:2]:
            raise RuntimeError(
                "reconstruct_envs returned misaligned legal-action tensors for root-action selection: "
                f"tokens={tuple(legal_tokens.shape)} weights={tuple(legal_weights.shape)}."
            )

        batch_size = int(legal_tokens.shape[0])
        requested = np.asarray(action_token_rows, dtype = np.int64)
        if int(requested.ndim) != 2:
            raise RuntimeError(
                "action_token_rows must have shape [B, MAX_ACTION_TOKENS] for policy-action selection: "
                f"{tuple(requested.shape)}."
            )
        if int(requested.shape[0]) != batch_size or int(requested.shape[1]) != int(legal_tokens.shape[2]):
            raise RuntimeError(
                "action_token_rows shape mismatch for policy-action selection: "
                f"expected={(batch_size, int(legal_tokens.shape[2]))}, got={tuple(requested.shape)}."
            )

        pad_token = int(self._game.PAD_TOKEN)
        selected = np.full(
            (batch_size, int(legal_tokens.shape[2])),
            fill_value = pad_token,
            dtype = np.int64,
        )
        for row_idx in range(batch_size):
            wanted = tuple(int(tok) for tok in requested[row_idx].tolist() if int(tok) != pad_token)
            if not wanted:
                raise RuntimeError(
                    "Root-action selection requires non-empty token rows: "
                    f"row={row_idx}."
                )
            row_lookup: dict[tuple[int, ...], int] = {}
            for action_idx in range(int(legal_tokens.shape[1])):
                if float(legal_weights[row_idx, action_idx]) <= 0.0:
                    continue
                token_tuple = tuple(int(tok) for tok in legal_tokens[row_idx, action_idx].tolist() if int(tok) != pad_token)
                row_lookup[token_tuple] = int(action_idx)
            action_idx = row_lookup.get(wanted, -1)
            if action_idx < 0:
                raise RuntimeError(
                    "Selected action token row is not legal at reconstructed decision state: "
                    f"row={row_idx} tokens={wanted!r}."
                )
            selected[row_idx] = legal_tokens[row_idx, action_idx]
        return selected

    def prefill_sampler_rollout_buffer(
        self,
        *,
        sampler: Sampler,
        history_strs: Sequence[str],
        pending_action: str,
    ) -> None:
        prefill_sampler_rollout_buffer_from_history_strs(
            game = self._game,
            sampler = sampler,
            history_strs = history_strs,
            pending_action = pending_action,
        )

    def build_sampler(self, *, envs: Any, envs_step: Any) -> Sampler:
        return build_sampler(
            game = self._game,
            policy_model = self._policy_model,
            rollout_temp = float(self._rollout_temp),
            envs = envs,
            envs_step = envs_step,
            collect_batch_size = 0,
        )

    def validate_prefilled_rollout_timeline(self, *, sampler: Sampler, next_step: Any) -> None:
        validate_prefilled_rollout_timeline(
            sampler = sampler,
            next_step = next_step,
            max_horizon = self._max_horizon,
        )

    def run_sampler_rollout(
        self,
        *,
        sampler: Sampler,
        agent_model: Any,
        forced_first_action_indices: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> np.ndarray:
        """Run the Sampler to terminal and return per-env root-player rewards.

        When ``forced_first_action_indices`` is provided, the first ``rollout_step`` uses
        those indices in place of its argmax/sampling selection. This is how
        ``rollout_selected_action_rows`` injects a predetermined root action without
        breaking the prefill↔env-state invariant (see the docstring on that method).
        """
        rewards = np.zeros(int(sampler.parsed_envs_step.batch_size[0]), dtype = np.float64)
        terminal = (
            sampler.parsed_envs_step.outcome["is_terminal"]
            .detach()
            .to(dtype = torch.bool, device = torch.device("cpu"))
            .numpy()
        )
        step_rewards = (
            sampler.parsed_envs_step.outcome["rewards"]
            .detach()
            .to(dtype = torch.float64, device = torch.device("cpu"))
            .numpy()
        )
        rewards[terminal] = step_rewards[terminal, self._root_player_id]
        is_terminal = terminal.copy()
        rollout_loops = 0

        with torch.inference_mode():
            sampler.prepare_agent_cache(agent_model = agent_model)
            while not bool(np.all(is_terminal)):
                rollout_loops += 1
                if rollout_loops > self._max_horizon:
                    raise RuntimeError("Update-equivalent rollout exceeded MAX_GAME_HORIZON before reaching terminal.")
                first_iteration = rollout_loops == 1
                if first_iteration and forced_first_action_indices is not None:
                    step_done = sampler.rollout_step(
                        agent_model = agent_model,
                        forced_action_indices = forced_first_action_indices,
                        generator = generator,
                    )
                else:
                    step_done = sampler.rollout_step(agent_model = agent_model, generator = generator)
                step_done_np = np.asarray(step_done.detach().to(device = torch.device("cpu")).numpy(), dtype = bool)
                outcome_rewards = (
                    sampler.parsed_envs_step.outcome["rewards"]
                    .detach()
                    .to(dtype = torch.float64, device = torch.device("cpu"))
                    .numpy()
                )
                newly_done = np.logical_and(np.logical_not(is_terminal), step_done_np)
                rewards[newly_done] = outcome_rewards[newly_done, self._root_player_id]
                is_terminal = np.logical_or(is_terminal, step_done_np)
        return rewards


__all__ = ["SearchRolloutSession"]
