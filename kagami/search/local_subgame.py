"""Local critic-backed Kagami subgame search."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from marvel.model.tensordict import TensorDict
import torch

from kagami.core.common import softmax_policy_probs, validate_temperature
from kagami.core.types import ActionValueEstimator, PolicyActionProb, PolicyWeight
from kagami.search.critic_value_estimator import CriticActionValueEstimator
from kagami.search.pending_policy import PendingPolicyEval, PendingPolicyQueryService
from kagami.search.routines import (
    DEFAULT_LOCAL_SEARCH_STABILITY_MARGIN,
    build_local_budget_ladder,
    default_local_search_floor,
    validate_search_budget,
    validate_search_floor,
)
from kagami.search.update_equivalent import PendingPolicyAction, UpdateEquivalentSearchEngine


ActionValueEstimatorFactory = Callable[..., ActionValueEstimator]
SubgameValueEstimatorFactory = Callable[..., "SubgameActionValueEstimator"]


@dataclass(frozen = True)
class LocalSearchSnapshot:
    budget: int
    actions: tuple[PendingPolicyAction, ...]
    top_action: str
    top_probability: float
    probability_margin: float


@dataclass(frozen = True)
class _SubgameKey:
    infoset_str: str
    pending_action: str


@dataclass
class _SubgameActionEdge:
    action_str: str
    is_commit: bool
    prior: float
    token_row: tuple[int, ...]
    sampled_count: int = 0
    leaf_value: np.ndarray | None = None
    terminal_values: list[np.ndarray] = field(default_factory = list)
    child_counts: dict[_SubgameKey, int] = field(default_factory = dict)


@dataclass
class _SubgameNode:
    key: _SubgameKey
    player_id: int
    pending_eval: PendingPolicyEval
    actions: list[_SubgameActionEdge]
    sample_histories: list[str] = field(default_factory = list)

    @property
    def sample_count(self) -> int:
        return len(self.sample_histories)


@dataclass
class _TabularSubgame:
    root_key: _SubgameKey
    root_player_id: int
    nodes: dict[_SubgameKey, _SubgameNode] = field(default_factory = dict)
    player_count: int = 0

    def record_value_width(self, value: np.ndarray) -> None:
        self.player_count = max(int(self.player_count), int(np.asarray(value).shape[0]))


class _FixedParticleBeliefModel:
    def __init__(self, belief_samples: Sequence[Any]) -> None:
        self._belief_samples = tuple(belief_samples)
        if not self._belief_samples:
            raise ValueError("local-subgame critic evaluation requires at least one belief sample.")

    def sample_determinized_histories(
        self,
        *,
        infoset_strs: Sequence[str],
        num_samples: int,
        temperature: float,
        seed: int,
    ) -> list[list[Any]]:
        del temperature, seed
        if len(infoset_strs) != 1:
            raise RuntimeError("Fixed local-subgame belief samples support one root infoset at a time.")
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        if count > len(self._belief_samples):
            raise RuntimeError(
                "Local-subgame fixed particle prefix is shorter than the requested critic sample count: "
                f"requested={count} available={len(self._belief_samples)}."
            )
        return [list(self._belief_samples[:count])]


class SubgameActionValueEstimator:
    """Critic action values for concrete sampled subgame nodes."""

    def __init__(
        self,
        *,
        game: Any,
        critic_model: Any,
        device: str | torch.device,
    ) -> None:
        self._game = game
        self._helper = CriticActionValueEstimator(
            game = game,
            belief_model = _FixedParticleBeliefModel([object()]),
            critic_model = critic_model,
            device = device,
            belief_temp = 1.0,
            belief_sample_count = 1,
        )

    def estimate_action_value_vectors(
        self,
        *,
        history_rows: Sequence[str],
        infoset_str: str,
        root_player_id: int,
        pending_action: str,
        pending_eval: PendingPolicyEval,
        seed: int,
    ) -> dict[str, np.ndarray]:
        del infoset_str, root_player_id
        action_strs = [str(action).strip() for action in pending_eval.action_strs]
        if not action_strs:
            raise RuntimeError("Subgame critic evaluation received no actions.")
        rows = [str(row) for row in history_rows]
        if not rows:
            raise RuntimeError("Subgame critic evaluation requires at least one concrete history.")

        rollout, num_steps_np = self._helper._build_critic_rollout(
            history_rows = rows,
            pending_action = str(pending_action).strip(),
            seed = int(seed),
        )
        qvalues_np = self._helper._evaluate_critic_qvalues(rollout = rollout, batch_size = len(rows))
        action_indices = self._helper._select_legal_action_indices(
            rollout = rollout,
            num_steps_np = num_steps_np,
            action_token_rows = np.asarray(pending_eval.action_token_rows, dtype = np.int64),
        )

        row_idx = np.arange(len(rows), dtype = np.int64)
        last_steps = np.asarray(num_steps_np, dtype = np.int64) - 1
        out: dict[str, np.ndarray] = {}
        for action_idx, action_str in enumerate(action_strs):
            legal_indices = action_indices[:, int(action_idx)]
            values = qvalues_np[row_idx, last_steps, legal_indices, :]
            out[str(action_str)] = np.mean(values, axis = 0, dtype = np.float64)
        return out


class _ActionValueEstimatorVectorAdapter:
    def __init__(
        self,
        *,
        game: Any,
        factory: ActionValueEstimatorFactory,
        belief_temp: float,
        critic_model: Any | None,
        device: torch.device,
    ) -> None:
        self._game = game
        self._factory = factory
        self._belief_temp = float(belief_temp)
        self._critic_model = critic_model
        self._device = device

    def estimate_action_value_vectors(
        self,
        *,
        history_rows: Sequence[str],
        infoset_str: str,
        root_player_id: int,
        pending_action: str,
        pending_eval: PendingPolicyEval,
        seed: int,
    ) -> dict[str, np.ndarray]:
        belief_samples = tuple(_HistorySample(str(row)) for row in history_rows)
        estimator = self._factory(
            game = self._game,
            belief_samples = belief_samples,
            belief_temp = float(self._belief_temp),
            belief_sample_count = len(belief_samples),
            critic_model = self._critic_model,
            device = self._device,
        )
        qvalues = estimator.estimate_action_qvalues(
            infoset_str = str(infoset_str),
            root_player_id = int(root_player_id),
            pending_action = str(pending_action),
            pending_eval = pending_eval,
            seed = int(seed),
        )
        width = max(int(root_player_id) + 1, 1)
        out: dict[str, np.ndarray] = {}
        for action in pending_eval.action_strs:
            vec = np.zeros((width,), dtype = np.float64)
            vec[int(root_player_id)] = float(qvalues.get(str(action), 0.0))
            out[str(action)] = vec
        return out


@dataclass(frozen = True)
class _HistorySample:
    history_str: str


@dataclass
class _PcfrNodeState:
    prior: np.ndarray
    cumulative_advantages: np.ndarray
    last_advantages: np.ndarray
    strategy_sum: np.ndarray
    strategy_weight: float = 0.0


def _normalize_probabilities(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype = np.float64).reshape(-1)
    if int(arr.shape[0]) == 0:
        raise RuntimeError("Cannot normalize an empty probability vector.")
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full((int(arr.shape[0]),), 1.0 / float(arr.shape[0]), dtype = np.float64)
    return arr / total


def _compute_kl_anchored_strategy(
    *,
    prior: np.ndarray,
    advantages: np.ndarray,
    search_beta: float,
    search_temp: float,
) -> np.ndarray:
    normalized_prior = _normalize_probabilities(prior)
    if float(search_beta) == 0.0:
        return normalized_prior.copy()

    advantage_arr = np.asarray(advantages, dtype = np.float64).reshape(-1)
    if int(advantage_arr.shape[0]) != int(normalized_prior.shape[0]):
        raise RuntimeError(
            "Anchored PCFR advantage width mismatch: "
            f"prior={int(normalized_prior.shape[0])} advantages={int(advantage_arr.shape[0])}."
        )
    centered_advantages = advantage_arr - float(np.max(advantage_arr))
    logits = torch.log(torch.clamp(torch.as_tensor(normalized_prior, dtype = torch.float64), min = 1e-300, max = 1.0))
    logits = logits + float(search_beta) * torch.as_tensor(centered_advantages, dtype = torch.float64)
    return softmax_policy_probs(logits, temperature = float(search_temp)).detach().cpu().numpy()


class PolicyAnchoredPcfrSolver:
    """Predictive CFR table with KL-anchored local strategies around Marvel priors."""

    def __init__(
        self,
        *,
        search_beta: float,
        search_temp: float,
    ) -> None:
        self._search_beta = float(search_beta)
        self._search_temp = float(search_temp)
        if self._search_beta < 0.0:
            raise ValueError("search_beta must be non-negative.")
        validate_temperature(self._search_temp, name = "search_temp")

    def solve(self, *, tree: _TabularSubgame, iterations: int) -> tuple[PendingPolicyAction, ...]:
        iteration_count = int(iterations)
        if iteration_count <= 0:
            raise ValueError("PCFR iterations must be positive.")
        if tree.root_key not in tree.nodes:
            raise RuntimeError("PCFR subgame tree does not contain its root node.")

        states = {key: self._build_node_state(node) for key, node in tree.nodes.items()}
        for iteration_idx in range(iteration_count):
            memo: dict[_SubgameKey, np.ndarray] = {}
            weight = float(iteration_idx + 1)
            self._compute_node_value(
                tree = tree,
                states = states,
                key = tree.root_key,
                memo = memo,
                strategy_weight = weight,
            )

        root = tree.nodes[tree.root_key]
        root_state = states[tree.root_key]
        root_strategy = self._average_strategy(root_state)
        return tuple(
            PendingPolicyAction(
                action_str = str(edge.action_str),
                prob = float(root_strategy[action_idx]),
                is_commit = bool(edge.is_commit),
            )
            for action_idx, edge in enumerate(root.actions)
        )

    def _build_node_state(self, node: _SubgameNode) -> _PcfrNodeState:
        priors = np.asarray([max(float(edge.prior), 0.0) for edge in node.actions], dtype = np.float64)
        total = float(np.sum(priors))
        if total <= 0.0:
            priors = np.full((len(node.actions),), 1.0 / float(len(node.actions)), dtype = np.float64)
        else:
            priors = priors / total
        return _PcfrNodeState(
            prior = priors,
            cumulative_advantages = np.zeros_like(priors),
            last_advantages = np.zeros_like(priors),
            strategy_sum = np.zeros_like(priors),
        )

    def _compute_node_value(
        self,
        *,
        tree: _TabularSubgame,
        states: dict[_SubgameKey, _PcfrNodeState],
        key: _SubgameKey,
        memo: dict[_SubgameKey, np.ndarray],
        strategy_weight: float,
    ) -> np.ndarray:
        cached = memo.get(key)
        if cached is not None:
            return cached

        node = tree.nodes[key]
        state = states[key]
        strategy = self._current_strategy(state)
        action_values = np.asarray(
            [
                self._compute_edge_value(
                    tree = tree,
                    states = states,
                    parent = node,
                    edge = edge,
                    memo = memo,
                    strategy_weight = strategy_weight,
                )
                for edge in node.actions
            ],
            dtype = np.float64,
        )
        node_value = np.sum(strategy[:, None] * action_values, axis = 0, dtype = np.float64)
        player_id = int(node.player_id)
        if 0 <= player_id < int(tree.player_count):
            instant = action_values[:, player_id] - float(node_value[player_id])
            state.cumulative_advantages = state.cumulative_advantages + instant
            state.last_advantages = instant
            state.strategy_sum += float(strategy_weight) * strategy
            state.strategy_weight += float(strategy_weight)
        else:
            state.strategy_sum += float(strategy_weight) * strategy
            state.strategy_weight += float(strategy_weight)
        memo[key] = node_value
        return node_value

    def _compute_edge_value(
        self,
        *,
        tree: _TabularSubgame,
        states: dict[_SubgameKey, _PcfrNodeState],
        parent: _SubgameNode,
        edge: _SubgameActionEdge,
        memo: dict[_SubgameKey, np.ndarray],
        strategy_weight: float,
    ) -> np.ndarray:
        player_count = int(tree.player_count)
        leaf_value = self._value_vector(edge.leaf_value, player_count = player_count)
        total_samples = max(int(parent.sample_count), 1)
        expanded_samples = len(edge.terminal_values) + sum(int(count) for count in edge.child_counts.values())
        unresolved_samples = max(total_samples - expanded_samples, 0)
        value_sum = float(unresolved_samples) * leaf_value

        for terminal_value in edge.terminal_values:
            value_sum = value_sum + self._value_vector(terminal_value, player_count = player_count)
        for child_key, count in edge.child_counts.items():
            child_value = self._compute_node_value(
                tree = tree,
                states = states,
                key = child_key,
                memo = memo,
                strategy_weight = strategy_weight,
            )
            value_sum = value_sum + float(count) * child_value
        return value_sum / float(total_samples)

    def _current_strategy(self, state: _PcfrNodeState) -> np.ndarray:
        predicted_advantages = state.cumulative_advantages + state.last_advantages
        return _compute_kl_anchored_strategy(
            prior = state.prior,
            advantages = predicted_advantages,
            search_beta = float(self._search_beta),
            search_temp = float(self._search_temp),
        )

    def _average_strategy(self, state: _PcfrNodeState) -> np.ndarray:
        if float(state.strategy_weight) <= 0.0 or float(np.sum(state.strategy_sum)) <= 0.0:
            return _normalize_probabilities(state.prior)
        return _normalize_probabilities(np.asarray(state.strategy_sum, dtype = np.float64) / float(state.strategy_weight))

    def _value_vector(self, value: np.ndarray | None, *, player_count: int) -> np.ndarray:
        if value is None:
            return np.zeros((player_count,), dtype = np.float64)
        arr = np.asarray(value, dtype = np.float64).reshape(-1)
        if int(arr.shape[0]) == int(player_count):
            return arr
        out = np.zeros((player_count,), dtype = np.float64)
        width = min(int(player_count), int(arr.shape[0]))
        out[:width] = arr[:width]
        return out


class _SampledSubgameBuilder:
    def __init__(
        self,
        *,
        game: Any,
        policy_query_service: PendingPolicyQueryService,
        value_estimator: SubgameActionValueEstimator | _ActionValueEstimatorVectorAdapter,
        root_player_id: int,
        max_query_depth: int,
        seed: int,
    ) -> None:
        self._game = game
        self._policy_query_service = policy_query_service
        self._value_estimator = value_estimator
        self._root_player_id = int(root_player_id)
        self._max_query_depth = int(max_query_depth)
        self._seed = int(seed)
        if self._max_query_depth <= 0:
            raise ValueError("max_query_depth must be positive.")

    def build(
        self,
        *,
        infoset_str: str,
        pending_action: str,
        belief_samples: Sequence[Any],
    ) -> _TabularSubgame:
        root_key = _SubgameKey(infoset_str = str(infoset_str), pending_action = str(pending_action).strip())
        tree = _TabularSubgame(root_key = root_key, root_player_id = int(self._root_player_id))
        for sample_idx, sample in enumerate(belief_samples):
            history_str = str(getattr(sample, "history_str"))
            self._expand_history(
                tree = tree,
                history_str = history_str,
                infoset_str = str(infoset_str),
                pending_action = str(pending_action).strip(),
                remaining_depth = int(self._max_query_depth),
                seed = UpdateEquivalentSearchEngine.mix_seed(self._seed, str(sample_idx), history_str),
            )
        self._populate_leaf_values(tree = tree)
        if tree.player_count <= 0:
            tree.player_count = max(int(self._root_player_id) + 1, 1)
        return tree

    def _expand_history(
        self,
        *,
        tree: _TabularSubgame,
        history_str: str,
        infoset_str: str,
        pending_action: str,
        remaining_depth: int,
        seed: int,
    ) -> None:
        node = self._ensure_node(tree = tree, infoset_str = infoset_str, pending_action = pending_action)
        node.sample_histories.append(str(history_str))
        if int(remaining_depth) <= 0 or int(node.player_id) < 0:
            return
        if not node.actions:
            return

        action_idx = self._sample_action_index(node = node, seed = int(seed))
        edge = node.actions[action_idx]
        edge.sampled_count += 1

        if not bool(edge.is_commit):
            child_key = _SubgameKey(infoset_str = str(infoset_str), pending_action = str(edge.action_str))
            edge.child_counts[child_key] = edge.child_counts.get(child_key, 0) + 1
            self._expand_history(
                tree = tree,
                history_str = str(history_str),
                infoset_str = str(infoset_str),
                pending_action = str(edge.action_str),
                remaining_depth = int(remaining_depth) - 1,
                seed = UpdateEquivalentSearchEngine.mix_seed(int(seed), str(edge.action_str), "pending"),
            )
            return

        step_result = self._step_history(
            history_str = str(history_str),
            pending_action = str(pending_action),
            edge = edge,
            seed = UpdateEquivalentSearchEngine.mix_seed(int(seed), str(edge.action_str), "step"),
        )
        if step_result is None:
            return
        next_history, terminal, rewards, next_player_id = step_result
        if bool(terminal):
            edge.terminal_values.append(np.asarray(rewards, dtype = np.float64))
            tree.record_value_width(np.asarray(rewards, dtype = np.float64))
            return
        if int(next_player_id) < 0:
            return
        infoset_from_history = getattr(self._game, "infoset_str_from_history_str", None)
        if not callable(infoset_from_history):
            return
        child_infoset = str(infoset_from_history(str(next_history)))
        child_key = _SubgameKey(infoset_str = child_infoset, pending_action = "")
        edge.child_counts[child_key] = edge.child_counts.get(child_key, 0) + 1
        self._expand_history(
            tree = tree,
            history_str = str(next_history),
            infoset_str = child_infoset,
            pending_action = "",
            remaining_depth = int(remaining_depth) - 1,
            seed = UpdateEquivalentSearchEngine.mix_seed(int(seed), str(edge.action_str), "child"),
        )

    def _ensure_node(self, *, tree: _TabularSubgame, infoset_str: str, pending_action: str) -> _SubgameNode:
        key = _SubgameKey(infoset_str = str(infoset_str), pending_action = str(pending_action).strip())
        cached = tree.nodes.get(key)
        if cached is not None:
            return cached

        decoded = self._game.parse_infoset_str(str(infoset_str), str(pending_action).strip())
        player_id = int(np.asarray(decoded.player_ids, dtype = np.int64)[-1])
        pending_eval = self._policy_query_service.evaluate_pending_policy(
            infoset_str = str(infoset_str),
            pending_action = str(pending_action).strip(),
        )
        prior_probs = softmax_policy_probs(pending_eval.logits, temperature = 1.0).detach().cpu().numpy()
        prior_probs = np.asarray(prior_probs, dtype = np.float64)
        total = float(np.sum(prior_probs))
        if total <= 0.0:
            prior_probs = np.full((len(pending_eval.action_strs),), 1.0 / float(len(pending_eval.action_strs)))
        else:
            prior_probs = prior_probs / total
        token_rows = np.asarray(pending_eval.action_token_rows, dtype = np.int64)
        actions = [
            _SubgameActionEdge(
                action_str = str(action_str),
                is_commit = bool(is_commit),
                prior = float(prior),
                token_row = tuple(int(token) for token in token_row.tolist()),
            )
            for action_str, is_commit, prior, token_row in zip(
                pending_eval.action_strs,
                pending_eval.is_commit,
                prior_probs.tolist(),
                token_rows,
                strict = False,
            )
        ]
        if not actions:
            raise RuntimeError("Local-subgame node has no policy actions.")
        node = _SubgameNode(
            key = key,
            player_id = int(player_id),
            pending_eval = pending_eval,
            actions = actions,
        )
        tree.nodes[key] = node
        return node

    def _sample_action_index(self, *, node: _SubgameNode, seed: int) -> int:
        probs = np.asarray([max(float(edge.prior), 0.0) for edge in node.actions], dtype = np.float64)
        total = float(np.sum(probs))
        if total <= 0.0:
            probs = np.full((len(node.actions),), 1.0 / float(len(node.actions)), dtype = np.float64)
        else:
            probs = probs / total
        return int(np.random.default_rng(int(seed)).choice(len(node.actions), p = probs))

    def _step_history(
        self,
        *,
        history_str: str,
        pending_action: str,
        edge: _SubgameActionEdge,
        seed: int,
    ) -> tuple[str, bool, np.ndarray, int] | None:
        reconstruct_envs = getattr(self._game, "reconstruct_envs", None)
        actions_from_np = getattr(self._game, "actions_from_np", None)
        if not callable(reconstruct_envs) or not callable(actions_from_np):
            return None

        envs, _root_step = reconstruct_envs(
            [str(history_str)],
            pending_actions = [str(pending_action).strip()],
            seed = int(seed),
        )
        action_tokens = np.asarray([list(edge.token_row)], dtype = np.int64)
        action_batch = actions_from_np(action_tokens)
        next_step = envs.step(action_batch)
        next_history_rows = envs.advance_history_strs([str(history_str)])
        next_history = str(next_history_rows[0])
        terminal = bool(np.asarray(next_step.is_terminal, dtype = np.bool_)[0])
        rewards = np.asarray(next_step.rewards, dtype = np.float64)[0]
        next_player_id = int(np.asarray(next_step.player_ids, dtype = np.int64)[0])
        return next_history, terminal, rewards, next_player_id

    def _populate_leaf_values(self, *, tree: _TabularSubgame) -> None:
        for node in tree.nodes.values():
            if not node.sample_histories:
                continue
            values = self._value_estimator.estimate_action_value_vectors(
                history_rows = node.sample_histories,
                infoset_str = str(node.key.infoset_str),
                root_player_id = int(self._root_player_id),
                pending_action = str(node.key.pending_action),
                pending_eval = node.pending_eval,
                seed = UpdateEquivalentSearchEngine.mix_seed(
                    self._seed,
                    str(node.key.infoset_str),
                    str(node.key.pending_action),
                    "leaf",
                ),
            )
            for edge in node.actions:
                value = np.asarray(values[str(edge.action_str)], dtype = np.float64)
                edge.leaf_value = value
                tree.record_value_width(value)


class PolicyRegularizedNoRegretSolver:
    """Compatibility wrapper for one-infoset KL-anchored policy solves."""

    @staticmethod
    def solve_action_distribution(
        *,
        qvalues: Mapping[str, float],
        pi_policy: Sequence[PolicyWeight],
        search_beta: float,
        search_temp: float,
    ) -> list[PolicyWeight]:
        if float(search_beta) < 0.0:
            raise ValueError("search_beta must be non-negative.")
        validate_temperature(float(search_temp), name = "search_temp")
        actions = [str(item.action_str) for item in pi_policy]
        if not actions:
            raise RuntimeError("Policy-anchored solve received no actions.")
        prior = np.asarray([max(float(item.prob), 0.0) for item in pi_policy], dtype = np.float64)
        q = np.asarray([float(qvalues.get(action, 0.0)) for action in actions], dtype = np.float64)
        probs = _compute_kl_anchored_strategy(
            prior = prior,
            advantages = q,
            search_beta = float(search_beta),
            search_temp = float(search_temp),
        )
        out = [
            PolicyWeight(action_str = action, prob = float(prob))
            for action, prob in zip(actions, probs.tolist(), strict = False)
        ]
        out.sort(key = lambda item: (-float(item.prob), str(item.action_str)))
        return out


class LocalSubgameSearchEngine:
    """Budget-ladder sampled tabular subgame search over Yomi particle prefixes."""

    def __init__(
        self,
        *,
        game: Any,
        policy_model: Any,
        belief_model: Any,
        critic_model: Any | None = None,
        critic_device: str | torch.device | None = None,
        policy_query_service: PendingPolicyQueryService | None = None,
        action_value_estimator_factory: ActionValueEstimatorFactory | None = None,
        subgame_value_estimator_factory: SubgameValueEstimatorFactory | None = None,
        belief_temp: float,
        search_beta: float,
        search_budget: int,
        search_floor: int | None = None,
        stability_margin: float = DEFAULT_LOCAL_SEARCH_STABILITY_MARGIN,
        search_temp: float = 1.0,
        action_temp: float = 0.0,
        seed: int = 0,
        max_query_depth: int = 2,
    ) -> None:
        self._game = game
        self._policy_model = policy_model
        self._belief_model = belief_model
        self._critic_model = critic_model
        self._critic_device = torch.device("cpu" if critic_device is None else critic_device)
        self._action_value_estimator_factory = action_value_estimator_factory
        self._subgame_value_estimator_factory = subgame_value_estimator_factory
        self._belief_temp = float(belief_temp)
        self._search_beta = float(search_beta)
        self._search_budget = int(search_budget)
        self._search_floor = (
            default_local_search_floor(self._search_budget)
            if search_floor is None
            else int(search_floor)
        )
        self._stability_margin = float(stability_margin)
        self._search_temp = float(search_temp)
        self._action_temp = float(action_temp)
        self._seed = int(seed)
        self._max_query_depth = int(max_query_depth)
        self._last_budget_snapshots: tuple[LocalSearchSnapshot, ...] = ()

        validate_search_budget(self._search_budget)
        validate_search_floor(self._search_floor, search_budget = self._search_budget)
        if self._search_beta < 0.0:
            raise ValueError("search_beta must be non-negative.")
        if self._stability_margin < 0.0:
            raise ValueError("stability_margin must be non-negative.")
        validate_temperature(self._search_temp, name = "search_temp")
        validate_temperature(self._action_temp, name = "action_temp")
        if self._max_query_depth <= 0:
            raise ValueError("max_query_depth must be positive.")
        if (
            self._subgame_value_estimator_factory is None
            and self._action_value_estimator_factory is None
            and self._critic_model is None
        ):
            raise ValueError("local-subgame search requires a qvalue CentralCritic.")

        self._policy_query_service = (
            PendingPolicyQueryService(game = game, policy_model = policy_model)
            if policy_query_service is None
            else policy_query_service
        )
        self._budget_ladder = build_local_budget_ladder(
            search_floor = self._search_floor,
            search_budget = self._search_budget,
        )

    @property
    def budget_ladder(self) -> tuple[int, ...]:
        return self._budget_ladder

    @property
    def last_budget_snapshots(self) -> tuple[LocalSearchSnapshot, ...]:
        return self._last_budget_snapshots

    def compute_policy_action_prob(self, *, infoset_str: str, pending_action: str) -> PolicyActionProb:
        decoded = self._game.parse_infoset_str(str(infoset_str), "")
        root_player_id = int(np.asarray(decoded.player_ids, dtype = np.int64)[-1])
        if root_player_id < 0:
            raise RuntimeError("local-subgame search requires an active-player infoset_str.")

        normalized_pending = str(pending_action).strip()
        root_seed = UpdateEquivalentSearchEngine.mix_seed(self._seed, str(infoset_str), normalized_pending)
        belief_samples = self._sample_root_belief_particles(
            infoset_str = str(infoset_str),
            seed = UpdateEquivalentSearchEngine.mix_seed(root_seed, "belief"),
        )

        snapshots: list[LocalSearchSnapshot] = []
        stable_top_action: str | None = None
        stable_count = 0
        for budget in self._budget_ladder:
            value_estimator = self._build_subgame_value_estimator()
            builder = _SampledSubgameBuilder(
                game = self._game,
                policy_query_service = self._policy_query_service,
                value_estimator = value_estimator,
                root_player_id = int(root_player_id),
                max_query_depth = int(self._max_query_depth),
                seed = UpdateEquivalentSearchEngine.mix_seed(root_seed, str(budget), "tree"),
            )
            tree = builder.build(
                infoset_str = str(infoset_str),
                pending_action = normalized_pending,
                belief_samples = belief_samples[: int(budget)],
            )
            actions = PolicyAnchoredPcfrSolver(
                search_beta = float(self._search_beta),
                search_temp = float(self._search_temp),
            ).solve(tree = tree, iterations = int(budget))
            snapshot = self._build_snapshot(budget = int(budget), actions = actions)
            snapshots.append(snapshot)

            if snapshot.top_action == stable_top_action and snapshot.probability_margin >= self._stability_margin:
                stable_count += 1
            else:
                stable_top_action = snapshot.top_action
                stable_count = 1
            if stable_count >= 2:
                break

        self._last_budget_snapshots = tuple(snapshots)
        if not snapshots:
            raise RuntimeError("local-subgame search produced no budget snapshots.")
        return self._policy_action_prob_from_actions(snapshots[-1].actions)

    def _sample_root_belief_particles(self, *, infoset_str: str, seed: int) -> tuple[Any, ...]:
        samples_batch = self._belief_model.sample_determinized_histories(
            infoset_strs = [str(infoset_str)],
            num_samples = int(self._search_budget),
            temperature = float(self._belief_temp),
            seed = int(seed),
        )
        if len(samples_batch) != 1:
            raise RuntimeError(f"Expected one local-subgame belief sample row, got {len(samples_batch)}.")
        samples = tuple(samples_batch[0])
        if len(samples) < self._search_budget:
            raise RuntimeError(
                "Yomi belief model returned fewer local-subgame particles than requested: "
                f"requested={self._search_budget} got={len(samples)}."
            )
        return samples[: self._search_budget]

    def _build_subgame_value_estimator(self) -> SubgameActionValueEstimator | _ActionValueEstimatorVectorAdapter:
        if self._subgame_value_estimator_factory is not None:
            return self._subgame_value_estimator_factory(
                game = self._game,
                critic_model = self._critic_model,
                device = self._critic_device,
            )
        if self._action_value_estimator_factory is not None:
            return _ActionValueEstimatorVectorAdapter(
                game = self._game,
                factory = self._action_value_estimator_factory,
                belief_temp = float(self._belief_temp),
                critic_model = self._critic_model,
                device = self._critic_device,
            )
        return SubgameActionValueEstimator(
            game = self._game,
            critic_model = self._critic_model,
            device = self._critic_device,
        )

    def _build_snapshot(self, *, budget: int, actions: Sequence[PendingPolicyAction]) -> LocalSearchSnapshot:
        ranked = list(actions)
        if not ranked:
            raise RuntimeError("local-subgame snapshot has no actions.")
        committed = [item for item in ranked if bool(item.is_commit)]
        candidates = committed if committed else ranked
        candidates.sort(key = lambda item: (-float(item.prob), str(item.action_str), int(bool(item.is_commit))))
        top = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        margin = float(top.prob) - (0.0 if runner_up is None else float(runner_up.prob))
        return LocalSearchSnapshot(
            budget = int(budget),
            actions = tuple(ranked),
            top_action = str(top.action_str),
            top_probability = float(top.prob),
            probability_margin = float(margin),
        )

    def _policy_action_prob_from_actions(self, actions: Sequence[PendingPolicyAction]) -> PolicyActionProb:
        raw_tuples = [
            (str(item.action_str), bool(item.is_commit), float(item.prob))
            for item in actions
            if float(item.prob) > 0.0
        ]
        raw_tuples.sort(key = lambda item: (-float(item[2]), str(item[0]), int(bool(item[1]))))
        if not raw_tuples:
            raise RuntimeError("local-subgame search produced an empty policy-action row.")

        raw_probs = torch.as_tensor([float(item[2]) for item in raw_tuples], dtype = torch.float64)
        positive_mask = raw_probs > 0.0
        if not bool(torch.any(positive_mask).item()):
            raise RuntimeError("local-subgame search produced no positive action probabilities.")
        final_probs = torch.zeros_like(raw_probs, dtype = torch.float64)
        positive_logits = torch.log(raw_probs[positive_mask])
        final_probs[positive_mask] = softmax_policy_probs(
            positive_logits,
            temperature = float(self._action_temp),
        )
        tuples = [
            (action_str, is_commit, float(prob))
            for (action_str, is_commit, _raw_prob), prob in zip(
                raw_tuples,
                final_probs.detach().cpu().tolist(),
                strict = False,
            )
            if float(prob) > 0.0
        ]
        tuples.sort(key = lambda item: (-float(item[2]), str(item[0]), int(bool(item[1]))))
        if not tuples:
            raise RuntimeError("local-subgame search produced an empty policy-action row after final temperature.")

        probs = torch.as_tensor([float(item[2]) for item in tuples], dtype = torch.float64)
        is_commit = torch.as_tensor([bool(item[1]) for item in tuples], dtype = torch.bool)
        tensors = TensorDict(
            {
                "probs": probs,
                "is_commit": is_commit,
            },
            batch_size = torch.Size([len(tuples)]),
            device = probs.device,
        )
        return PolicyActionProb(
            action_strs = [str(item[0]) for item in tuples],
            tensors = tensors,
        )


__all__ = [
    "LocalSearchSnapshot",
    "LocalSubgameSearchEngine",
    "PolicyAnchoredPcfrSolver",
    "PolicyRegularizedNoRegretSolver",
    "SubgameActionValueEstimator",
]
