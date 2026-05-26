from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import kagami.search as kagami_search
from kagami.core.types import PolicyWeight
from kagami.search.local_subgame import LocalSubgameSearchEngine
from kagami.search.pending_policy import PendingPolicyEval
from kagami.search.routines import build_local_budget_ladder, default_local_search_floor


class _FakeGame:
    PAD_TOKEN = 0

    def parse_infoset_str(self, infoset_str: str, pending_action: str = "") -> SimpleNamespace:
        del infoset_str, pending_action
        return SimpleNamespace(player_ids = np.asarray([0], dtype = np.int64))


class _FakeBeliefModel:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], int, float, int]] = []

    def sample_determinized_histories(
        self,
        *,
        infoset_strs: list[str],
        num_samples: int,
        temperature: float,
        seed: int,
    ) -> list[list[SimpleNamespace]]:
        self.calls.append((tuple(infoset_strs), int(num_samples), float(temperature), int(seed)))
        return [[SimpleNamespace(history_str = f"H{idx}") for idx in range(int(num_samples))]]


class _FakePolicyQueryService:
    def evaluate_pending_policy(self, *, infoset_str: str, pending_action: str) -> PendingPolicyEval:
        del infoset_str, pending_action
        return PendingPolicyEval(
            action_strs = ["A", "B"],
            is_commit = [True, True],
            action_token_rows = np.asarray([[1, 0], [2, 0]], dtype = np.int64),
            logits = torch.log(torch.as_tensor([0.5, 0.5], dtype = torch.float64)),
        )


class _ScriptedEstimator:
    def __init__(self, *, qvalues: dict[str, float]) -> None:
        self._qvalues = dict(qvalues)

    def estimate_action_qvalues(self, **kwargs) -> dict[str, float]:  # noqa: ANN003
        pending_eval = kwargs["pending_eval"]
        return {str(action): float(self._qvalues[str(action)]) for action in pending_eval.action_strs}


class _BranchingStep:
    def __init__(self, *, terminal: bool, rewards: tuple[float, float], player_id: int) -> None:
        self.is_terminal = np.asarray([bool(terminal)], dtype = np.bool_)
        self.rewards = np.asarray([list(rewards)], dtype = np.float64)
        self.player_ids = np.asarray([int(player_id)], dtype = np.int64)


class _BranchingEnvBatch:
    def __init__(self, *, history: str) -> None:
        self._history = str(history)
        self._next_history = str(history)

    def step(self, action_batch):  # noqa: ANN001, ANN201
        action = tuple(int(token) for token in action_batch[0] if int(token) != 0)
        if self._history == "root":
            if action == (1,):
                self._next_history = "child"
                return _BranchingStep(terminal = False, rewards = (0.0, 0.0), player_id = 1)
            if action == (2,):
                self._next_history = "terminal_b"
                return _BranchingStep(terminal = True, rewards = (0.0, 0.0), player_id = -2)
        if self._history == "child":
            self._next_history = f"terminal_child_{action[0]}"
            if action == (3,):
                return _BranchingStep(terminal = True, rewards = (3.0, -3.0), player_id = -2)
            if action == (4,):
                return _BranchingStep(terminal = True, rewards = (-3.0, 3.0), player_id = -2)
        raise RuntimeError(f"Unexpected fake transition: history={self._history!r} action={action!r}.")

    def advance_history_strs(self, history_strs):  # noqa: ANN001, ANN201
        assert [str(item) for item in history_strs] == [self._history]
        return [self._next_history]


class _BranchingGame:
    PAD_TOKEN = 0

    def parse_infoset_str(self, infoset_str: str, pending_action: str = "") -> SimpleNamespace:
        del pending_action
        player_id = 1 if str(infoset_str) == "CHILD" else 0
        return SimpleNamespace(player_ids = np.asarray([player_id], dtype = np.int64))

    def reconstruct_envs(self, history_strs, pending_actions = None, seed = 0):  # noqa: ANN001, ANN201
        del pending_actions, seed
        return _BranchingEnvBatch(history = str(history_strs[0])), object()

    def actions_from_np(self, tokens):  # noqa: ANN001, ANN201
        return [tuple(int(token) for token in row.tolist()) for row in np.asarray(tokens, dtype = np.int64)]

    def infoset_str_from_history_str(self, history_str: str) -> str:
        if str(history_str) == "child":
            return "CHILD"
        raise RuntimeError(f"Unexpected fake child history: {history_str!r}.")


class _BranchingPolicyQueryService:
    def evaluate_pending_policy(self, *, infoset_str: str, pending_action: str) -> PendingPolicyEval:
        del pending_action
        if str(infoset_str) == "ROOT":
            return PendingPolicyEval(
                action_strs = ["A", "B"],
                is_commit = [True, True],
                action_token_rows = np.asarray([[1, 0], [2, 0]], dtype = np.int64),
                logits = torch.log(torch.as_tensor([0.9, 0.1], dtype = torch.float64)),
            )
        if str(infoset_str) == "CHILD":
            return PendingPolicyEval(
                action_strs = ["X", "Y"],
                is_commit = [True, True],
                action_token_rows = np.asarray([[3, 0], [4, 0]], dtype = np.int64),
                logits = torch.log(torch.as_tensor([0.5, 0.5], dtype = torch.float64)),
            )
        raise RuntimeError(f"Unexpected fake infoset: {infoset_str!r}.")


class _BranchingBeliefModel:
    def sample_determinized_histories(
        self,
        *,
        infoset_strs: list[str],
        num_samples: int,
        temperature: float,
        seed: int,
    ) -> list[list[SimpleNamespace]]:
        del infoset_strs, temperature, seed
        return [[SimpleNamespace(history_str = "root") for _ in range(int(num_samples))]]


class _BranchingSubgameValueEstimator:
    def __init__(self) -> None:
        self.seen_infosets: list[str] = []

    def estimate_action_value_vectors(self, **kwargs) -> dict[str, np.ndarray]:  # noqa: ANN003
        infoset = str(kwargs["infoset_str"])
        pending_eval = kwargs["pending_eval"]
        self.seen_infosets.append(infoset)
        if infoset == "ROOT":
            values = {
                "A": np.asarray([1.0, -1.0], dtype = np.float64),
                "B": np.asarray([0.0, 0.0], dtype = np.float64),
            }
        elif infoset == "CHILD":
            values = {
                "X": np.asarray([3.0, -3.0], dtype = np.float64),
                "Y": np.asarray([-3.0, 3.0], dtype = np.float64),
            }
        else:
            raise RuntimeError(f"Unexpected fake value infoset: {infoset!r}.")
        return {str(action): values[str(action)] for action in pending_eval.action_strs}


def test_local_budget_ladder_doubles_from_floor_and_includes_cap() -> None:
    assert build_local_budget_ladder(search_floor = 3, search_budget = 20) == (3, 6, 12, 20)
    assert build_local_budget_ladder(search_floor = 8, search_budget = 8) == (8,)

    with pytest.raises(ValueError, match = "search_floor"):
        _ = build_local_budget_ladder(search_floor = 0, search_budget = 8)
    with pytest.raises(ValueError, match = "<= search_budget"):
        _ = build_local_budget_ladder(search_floor = 9, search_budget = 8)


def test_default_local_search_floor_caps_to_budget() -> None:
    assert default_local_search_floor(512) == 16
    assert default_local_search_floor(16) == 16
    assert default_local_search_floor(8) == 8


def test_policy_regularized_solver_solves_closed_form_kl_response() -> None:
    solved = kagami_search.PolicyRegularizedNoRegretSolver.solve_action_distribution(
        qvalues = {"A": 1.0, "B": 0.0},
        pi_policy = [
            PolicyWeight(action_str = "A", prob = 0.25),
            PolicyWeight(action_str = "B", prob = 0.75),
        ],
        search_beta = 4.0,
        search_temp = 1.0,
    )

    got = {str(item.action_str): float(item.prob) for item in solved}
    expected_a = 0.25 * np.exp(4.0) / (0.25 * np.exp(4.0) + 0.75)
    assert pytest.approx(sum(got.values()), abs = 1e-12) == 1.0
    assert got["A"] == pytest.approx(expected_a)
    assert got["B"] == pytest.approx(1.0 - expected_a)


def test_policy_regularized_solver_beta_zero_stays_on_policy_prior() -> None:
    solved = kagami_search.PolicyRegularizedNoRegretSolver.solve_action_distribution(
        qvalues = {"A": 10.0, "B": -10.0},
        pi_policy = [
            PolicyWeight(action_str = "A", prob = 0.25),
            PolicyWeight(action_str = "B", prob = 0.75),
        ],
        search_beta = 0.0,
        search_temp = 0.0,
    )

    got = {str(item.action_str): float(item.prob) for item in solved}
    assert got["A"] == pytest.approx(0.25)
    assert got["B"] == pytest.approx(0.75)


def test_local_subgame_samples_max_particles_once_and_reuses_prefixes() -> None:
    belief_model = _FakeBeliefModel()
    seen_prefixes: list[tuple[str, ...]] = []

    def _factory(**kwargs):  # noqa: ANN003, ANN202
        samples = tuple(str(sample.history_str) for sample in kwargs["belief_samples"])
        seen_prefixes.append(samples)
        return _ScriptedEstimator(qvalues = {"A": 1.0, "B": 0.0})

    engine = LocalSubgameSearchEngine(
        game = _FakeGame(),
        policy_model = object(),
        belief_model = belief_model,
        action_value_estimator_factory = _factory,
        policy_query_service = _FakePolicyQueryService(),
        belief_temp = 0.5,
        search_beta = 4.0,
        search_budget = 16,
        search_floor = 4,
        stability_margin = 0.05,
        search_temp = 1.0,
        action_temp = 1.0,
        seed = 123,
    )
    row = engine.compute_policy_action_prob(infoset_str = "I", pending_action = "")

    assert [(items[0], items[1], items[2]) for items in belief_model.calls] == [(("I",), 16, 0.5)]
    assert seen_prefixes == [
        ("H0", "H1", "H2", "H3"),
        ("H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"),
    ]
    assert tuple(snapshot.budget for snapshot in engine.last_budget_snapshots) == (4, 8)
    assert row.action_strs[0] == "A"
    assert float(row.probs[0].item()) > 0.5


def test_local_subgame_falls_back_to_max_budget_when_top_action_changes() -> None:
    belief_model = _FakeBeliefModel()
    scripted = {
        2: {"A": 1.0, "B": 0.0},
        4: {"A": 0.0, "B": 1.0},
        8: {"A": 1.0, "B": 0.0},
    }

    def _factory(**kwargs):  # noqa: ANN003, ANN202
        budget = int(kwargs["belief_sample_count"])
        return _ScriptedEstimator(qvalues = scripted[budget])

    engine = LocalSubgameSearchEngine(
        game = _FakeGame(),
        policy_model = object(),
        belief_model = belief_model,
        action_value_estimator_factory = _factory,
        policy_query_service = _FakePolicyQueryService(),
        belief_temp = 1.0,
        search_beta = 4.0,
        search_budget = 8,
        search_floor = 2,
        stability_margin = 0.05,
        search_temp = 1.0,
        action_temp = 1.0,
        seed = 7,
    )
    row = engine.compute_policy_action_prob(infoset_str = "I", pending_action = "")

    assert tuple(snapshot.budget for snapshot in engine.last_budget_snapshots) == (2, 4, 8)
    assert row.action_strs[0] == "A"


def test_local_subgame_uses_search_temp_for_snapshots_and_action_temp_for_output() -> None:
    belief_model = _FakeBeliefModel()

    def _factory(**kwargs):  # noqa: ANN003, ANN202
        del kwargs
        return _ScriptedEstimator(qvalues = {"A": 1.0, "B": 0.0})

    engine = LocalSubgameSearchEngine(
        game = _FakeGame(),
        policy_model = object(),
        belief_model = belief_model,
        action_value_estimator_factory = _factory,
        policy_query_service = _FakePolicyQueryService(),
        belief_temp = 1.0,
        search_beta = 1.0,
        search_budget = 4,
        search_floor = 2,
        stability_margin = 0.05,
        search_temp = 1.0,
        action_temp = 0.0,
        seed = 7,
    )
    row = engine.compute_policy_action_prob(infoset_str = "I", pending_action = "")

    assert tuple(snapshot.budget for snapshot in engine.last_budget_snapshots) == (2, 4)
    assert engine.last_budget_snapshots[0].top_action == "A"
    assert 0.5 < engine.last_budget_snapshots[0].top_probability < 1.0
    assert row.action_strs == ["A"]
    assert torch.equal(row.probs, torch.ones_like(row.probs))


def test_local_subgame_runs_tabular_pcfr_on_expanded_sampled_tree() -> None:
    value_estimator = _BranchingSubgameValueEstimator()

    engine = LocalSubgameSearchEngine(
        game = _BranchingGame(),
        policy_model = object(),
        belief_model = _BranchingBeliefModel(),
        subgame_value_estimator_factory = lambda **kwargs: value_estimator,  # noqa: ARG005
        policy_query_service = _BranchingPolicyQueryService(),
        belief_temp = 1.0,
        search_beta = 8.0,
        search_budget = 8,
        search_floor = 8,
        stability_margin = 0.0,
        search_temp = 1.0,
        action_temp = 1.0,
        seed = 7,
    )
    row = engine.compute_policy_action_prob(infoset_str = "ROOT", pending_action = "")

    got = {str(action): float(prob) for action, prob in zip(row.action_strs, row.probs.tolist(), strict = False)}
    assert "CHILD" in value_estimator.seen_infosets
    assert got["B"] > got["A"]


def test_local_subgame_requires_qvalue_critic_without_test_factory() -> None:
    with pytest.raises(ValueError, match = "qvalue CentralCritic"):
        _ = LocalSubgameSearchEngine(
            game = _FakeGame(),
            policy_model = object(),
            belief_model = _FakeBeliefModel(),
            belief_temp = 1.0,
            search_beta = 1.0,
            search_budget = 8,
            search_floor = 4,
        )
