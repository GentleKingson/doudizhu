from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

import numpy as np
import nanopokerlib as npl
import pytest
import torch

import kagami.search as kagami_search
from kagami.core.common import softmax_policy_probs
from kagami.core import InferenceEngine
from kagami.core.types import (
    PolicyActionProb,
    PolicyActionProbBatch,
    policy_action_prob_batch_from_rows,
    policy_action_prob_row_from_batch,
)
from kagami.games import DoudizhuInferenceEngine, DoudizhuInferenceEngineConfig
from marvel.model.tensordict import TensorDict


@dataclass(frozen = True)
class _FakeStep:
    action_strs: list[str]
    is_commit: list[bool]
    logits: torch.Tensor


class _FakeEvaluator:
    def __init__(self, steps: dict[str, _FakeStep], *, game: object | None = None) -> None:
        self.steps = dict(steps)
        self.evaluate_calls = 0
        self.evaluate_batch_calls = 0
        if game is not None:
            self._game = game

    def _step_for_pending(self, pending_action: str) -> _FakeStep:
        if pending_action not in self.steps:
            raise RuntimeError(f"Missing fake step for pending_action={pending_action!r}")
        return self.steps[pending_action]

    def evaluate(self, *, infoset_str: str, pending_action: str) -> _FakeStep:
        del infoset_str
        self.evaluate_calls += 1
        return self._step_for_pending(pending_action)

    def evaluate_batch(self, *, infoset_strs: list[str], pending_actions: list[str]) -> list[_FakeStep]:
        if len(infoset_strs) != len(pending_actions):
            raise RuntimeError("batch evaluator inputs are misaligned")
        self.evaluate_batch_calls += 1
        return [self._step_for_pending(pending_action) for pending_action in pending_actions]

    def policy(self, *, infoset_str: str, pending_action: str, action_temp: float) -> PolicyActionProb:
        batch = self.policy_batch(
            infoset_strs = [infoset_str],
            pending_actions = [pending_action],
            action_temps = [float(action_temp)],
        )
        return policy_action_prob_row_from_batch(batch, row_idx = 0)

    def policy_batch(
        self,
        *,
        infoset_strs: list[str],
        pending_actions: list[str],
        action_temps: list[float],
    ) -> PolicyActionProbBatch:
        eval_rows = self.evaluate_batch(infoset_strs = infoset_strs, pending_actions = pending_actions)
        if len(eval_rows) != len(action_temps):
            raise RuntimeError("action_temps must match infoset_strs.")
        rows: list[PolicyActionProb] = []
        for row_idx, step in enumerate(eval_rows):
            probs = softmax_policy_probs(step.logits, temperature = float(action_temps[row_idx]))
            if tuple(probs.shape) != (len(step.action_strs),):
                raise RuntimeError("Policy probability shape mismatch in fake evaluator.")
            tensors = TensorDict(
                {
                    "probs": probs.detach().to(dtype = torch.float64),
                    "is_commit": torch.as_tensor(step.is_commit, dtype = torch.bool, device = probs.device),
                },
                batch_size = torch.Size([len(step.action_strs)]),
                device = probs.device,
            )
            rows.append(PolicyActionProb(action_strs = list(step.action_strs), tensors = tensors))
        return policy_action_prob_batch_from_rows(rows)


def _make_step(actions: list[str], commit: list[bool], probs: list[float]) -> _FakeStep:
    if len(actions) != len(commit) or len(actions) != len(probs):
        raise RuntimeError("fake step input length mismatch")
    p = np.asarray(probs, dtype = np.float64)
    p = p / float(np.sum(p))
    logits = torch.tensor(np.log(np.clip(p, 1e-30, 1.0)), dtype = torch.float64)
    return _FakeStep(action_strs = actions, is_commit = commit, logits = logits)


def _row_to_items(row: PolicyActionProb) -> list[dict[str, float | str | bool]]:
    return [
        {
            "action_str": str(action_str),
            "is_commit": bool(row.is_commit[idx].item()),
            "prob": float(row.probs[idx].item()),
        }
        for idx, action_str in enumerate(row.action_strs)
    ]


def _batch_to_items(batch: PolicyActionProbBatch) -> list[list[dict[str, float | str | bool]]]:
    return [
        _row_to_items(policy_action_prob_row_from_batch(batch, row_idx = row_idx))
        for row_idx in range(len(batch.action_strs_batch))
    ]


def _sample_initial_counts(game, seed: int = 0):  # noqa: ANN001
    env = game.Environment()
    step = env.reset(seed = seed)

    initial_hands = [[0] * int(game.NUM_RANKS) for _ in range(int(game.NUM_PLAYERS))]
    landlord_public = [0] * int(game.NUM_RANKS)
    tok = game.token
    chance_steps = 0

    while int(step.player_id) < 0:
        step = env.step(game.Action())
        toks_p1 = step.observations.tokens[1]
        if len(toks_p1) < 2:
            chance_steps += 1
            if chance_steps > int(game.max_game_horizon):
                raise RuntimeError("Exceeded chance-step cap while sampling Dou Dizhu root state.")
            continue

        receiver = int(tok.decode_player(int(toks_p1[0])))
        if receiver == 1:
            for player_id in range(int(game.NUM_PLAYERS)):
                tokens = step.observations.tokens[player_id]
                if len(tokens) < 2:
                    continue
                rank = int(tok.decode_rank(int(tokens[1])))
                initial_hands[player_id][rank] += 1
        else:
            rank = int(tok.decode_rank(int(step.observations.tokens[0][1])))
            initial_hands[0][rank] += 1
            landlord_public[rank] += 1

        chance_steps += 1
        if chance_steps > int(game.max_game_horizon):
            raise RuntimeError("Exceeded chance-step cap while sampling Dou Dizhu root state.")

    return env, step, tuple(tuple(int(v) for v in row) for row in initial_hands), tuple(int(v) for v in landlord_public)


def _counts_to_cards(game, counts):  # noqa: ANN001
    return "".join(str(game.token.rank_to_string(rank)) * int(qty) for rank, qty in enumerate(counts))


def _belief_target_str(game, payload) -> str:  # noqa: ANN001
    candidate_tokens = np.asarray(payload["candidate_tokens"], dtype = np.int64)
    target_index = np.asarray(payload["target_index"], dtype = np.int64)
    belief_length = int(payload["belief_length"])
    tags = ("L", "P1", "P2")
    pieces: list[str] = []
    for row_idx in range(belief_length):
        row = candidate_tokens[row_idx, int(target_index[row_idx])]
        owner = int(game.token.decode_player(int(row[0])))
        rank = int(game.token.decode_rank(int(row[1])))
        pieces.append(f"{tags[owner]}+{str(game.token.rank_to_string(rank))}")
    return "|".join(pieces)


def _root_infoset_and_history(game, seed: int = 0):  # noqa: ANN001
    env, step, initial_hands, landlord_public = _sample_initial_counts(game, seed = seed)
    if int(step.player_id) != 0:
        raise RuntimeError(f"Expected landlord root player, got {step.player_id}.")

    landlord_private = [
        int(initial_hands[0][rank_idx]) - int(landlord_public[rank_idx])
        for rank_idx in range(int(game.NUM_RANKS))
    ]
    if any(qty < 0 for qty in landlord_private):
        raise RuntimeError("landlord_public exceeds landlord initial hand while building root infoset.")

    infoset_str = "|".join(
        [
            f"L+{_counts_to_cards(game, landlord_private)}",
            f"L+{_counts_to_cards(game, landlord_public)}",
        ]
    )
    decoded = game.parse_infoset_str(infoset_str)
    action_strings = [str(item) for item in decoded.last_available_action_strs]
    is_commit = np.asarray(decoded.last_available_actions.is_commit, dtype = bool)
    committed_actions = [
        action_str
        for action_str, commit in zip(action_strings, is_commit, strict = False)
        if bool(commit) and str(action_str).strip()
    ]
    belief_str = _belief_target_str(game, env.belief_supervision())
    history_str = str(game.determinize_history_str(infoset_str, belief_str))
    return infoset_str, history_str, committed_actions


class _FixedBeliefModel:
    def __init__(self, *, history_str: str) -> None:
        self.history_str = str(history_str)

    def sample_determinized_histories(
        self,
        *,
        infoset_strs: list[str],
        num_samples: int,
        temperature: float,
        seed: int,
    ) -> list[list[SimpleNamespace]]:
        del temperature, seed
        return [
            [
                SimpleNamespace(history_str = self.history_str)
                for _ in range(int(num_samples))
            ]
            for _ in infoset_strs
        ]


def test_doudizhu_engine_implements_game_agnostic_interface() -> None:
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["PASS", "333"], [True, True], [0.4, 0.6]),
        }
    )
    engine = DoudizhuInferenceEngine(policy_model = evaluator)
    assert isinstance(engine, InferenceEngine)


def test_doudizhu_engine_exposes_model_binding_helpers() -> None:
    engine = DoudizhuInferenceEngine(policy_model = _FakeEvaluator({"": _make_step(["PASS"], [True], [1.0])}))
    assert callable(getattr(engine, "bind_marvel_model"))
    assert callable(getattr(engine, "bind_yomi_model"))


def test_doudizhu_engine_default_config_uses_local_subgame_search_defaults() -> None:
    cfg = DoudizhuInferenceEngineConfig()
    assert cfg.action_temp == pytest.approx(0.0)
    assert cfg.search_temp == pytest.approx(1.0)
    assert cfg.update_equivalent.routine == "local-subgame"
    assert cfg.update_equivalent.rollout_budget == 512
    assert cfg.update_equivalent.search_floor is None


def test_bind_marvel_model_rebinds_implicit_default_game(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = DoudizhuInferenceEngine()
    original_game = engine._game
    target_game = object()

    class _FakeMarvelModel:
        def __init__(self, **_kwargs) -> None:  # noqa: ANN003
            self._game = target_game

    monkeypatch.setattr("kagami.connectors.marvel.MarvelModel", _FakeMarvelModel)
    engine.bind_marvel_model(checkpoint = "/tmp/ignored.pt", device = "cpu")

    assert engine._game is target_game
    assert engine._policy_model._game is target_game
    assert engine._game is not original_game


def test_bind_marvel_model_rejects_game_mismatch_for_explicit_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit_game = object()
    engine = DoudizhuInferenceEngine(game = explicit_game)

    class _FakeMarvelModel:
        def __init__(self, **_kwargs) -> None:  # noqa: ANN003
            self._game = object()

    monkeypatch.setattr("kagami.connectors.marvel.MarvelModel", _FakeMarvelModel)
    with pytest.raises(ValueError, match = "bound game"):
        engine.bind_marvel_model(checkpoint = "/tmp/ignored.pt", device = "cpu")
    assert engine._game is explicit_game


def test_doudizhu_engine_rejects_mode_kwarg() -> None:
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["PASS", "333"], [True, True], [0.4, 0.6]),
        }
    )
    engine = DoudizhuInferenceEngine(policy_model = evaluator)
    with pytest.raises(TypeError, match = "mode"):
        _ = engine.infer(infoset_str = "L+3344|L+BR2", mode = "distribution")  # type: ignore[call-arg]


def test_doudizhu_engine_rejects_per_call_temperature_kwargs() -> None:
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["PASS", "333"], [True, True], [0.4, 0.6]),
        }
    )
    engine = DoudizhuInferenceEngine(policy_model = evaluator)
    with pytest.raises(TypeError, match = "action_temp"):
        _ = engine.infer(infoset_str = "L+3344|L+BR2", action_temp = 1.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match = "action_temp"):
        _ = engine.infer_batch(  # type: ignore[call-arg]
            infoset_strs = ["L+3344|L+BR2"],
            action_temp = 1.0,
        )
    with pytest.raises(TypeError, match = "max_pending_depth"):
        _ = engine.prob_batch(  # type: ignore[call-arg]
            infoset_strs = ["L+3344|L+BR2"],
            committed_actions = ["PASS"],
            max_pending_depth = 8,
        )


def test_step_distribution_is_normalized_and_has_commit_schema() -> None:
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["PASS", "333"], [True, True], [0.3, 0.7]),
        }
    )
    engine = DoudizhuInferenceEngine(policy_model = evaluator)

    dist = _row_to_items(engine.infer(infoset_str = "L+3344|L+BR2"))
    probs = [float(item["prob"]) for item in dist]
    actions = [str(item["action_str"]) for item in dist]
    commits = [bool(item["is_commit"]) for item in dist]
    assert actions == ["PASS", "333"]
    assert commits == [True, True]
    assert math.isclose(sum(probs), 1.0, rel_tol = 0.0, abs_tol = 1e-12)
    assert probs[1] > probs[0]


def test_step_distribution_from_pending_action_is_valid() -> None:
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["3+", "PASS"], [False, True], [0.6, 0.4]),
            "3+": _make_step(["333", "334"], [True, True], [0.2, 0.8]),
        }
    )
    engine = DoudizhuInferenceEngine(policy_model = evaluator)

    dist = _row_to_items(engine.infer(infoset_str = "P1+QKA2B|L+BR2|L-3", pending_action = "3+"))
    assert [str(item["action_str"]) for item in dist] == ["333", "334"]
    assert [bool(item["is_commit"]) for item in dist] == [True, True]


def test_prob_batch_generic_pending_expansion() -> None:
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["3+", "PASS"], [False, True], [0.6, 0.4]),
            "3+": _make_step(["333", "334"], [True, True], [0.25, 0.75]),
        }
    )
    engine = DoudizhuInferenceEngine(
        policy_model = evaluator,
        cfg = DoudizhuInferenceEngineConfig(action_temp = 1.0),
    )

    probs = engine.prob_batch(
        infoset_strs = ["S1", "S2", "S3", "S4"],
        committed_actions = ["334", "PASS", "333", "444"],
    )

    assert probs == pytest.approx([0.45, 0.4, 0.15, 0.0], rel = 1e-9, abs = 1e-12)
    assert evaluator.evaluate_batch_calls == 2
    assert evaluator.evaluate_calls == 0


def test_prob_scalar_wrapper() -> None:
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["3+", "PASS"], [False, True], [0.6, 0.4]),
            "3+": _make_step(["333", "334"], [True, True], [0.25, 0.75]),
        }
    )
    engine = DoudizhuInferenceEngine(
        policy_model = evaluator,
        cfg = DoudizhuInferenceEngineConfig(action_temp = 1.0),
    )

    prob = engine.prob(
        infoset_str = "S1",
        committed_action = "334",
    )
    assert prob == pytest.approx(0.45, rel = 1e-9, abs = 1e-12)


def test_temperature_zero_yields_greedy_step_distribution() -> None:
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["PASS", "333"], [True, True], [0.1, 0.9]),
        }
    )
    engine = DoudizhuInferenceEngine(
        policy_model = evaluator,
        cfg = DoudizhuInferenceEngineConfig(action_temp = 0.0),
    )

    dist = _row_to_items(engine.infer(infoset_str = "L+3344|L+BR2"))
    assert dist == [
        {"action_str": "PASS", "is_commit": True, "prob": 0.0},
        {"action_str": "333", "is_commit": True, "prob": 1.0},
    ]


def test_tool_side_action_loop_works_in_direct_mode() -> None:
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["3+", "PASS"], [False, True], [0.8, 0.2]),
            "3+": _make_step(["333", "334"], [True, True], [0.1, 0.9]),
        }
    )
    engine = DoudizhuInferenceEngine(
        policy_model = evaluator,
        cfg = DoudizhuInferenceEngineConfig(action_temp = 0.0),
    )

    actions = [str(action) for action in engine.act_batch(infoset_strs = ["S1", "S2", "S3"])]
    assert actions == ["334", "334", "334"]
    assert evaluator.evaluate_batch_calls == 2
    assert evaluator.evaluate_calls == 0


def test_doudizhu_engine_rejects_non_tensor_logits() -> None:
    step = _FakeStep(action_strs = ["PASS"], is_commit = [True], logits = np.asarray([0.0], dtype = np.float64))  # type: ignore[arg-type]
    evaluator = _FakeEvaluator({"": step})
    engine = DoudizhuInferenceEngine(policy_model = evaluator)

    with pytest.raises(TypeError, match = "torch.Tensor"):
        _ = engine.infer(infoset_str = "L+3344|L+BR2")


def test_update_equivalent_requires_belief_connector_at_inference() -> None:
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["PASS", "333"], [True, True], [0.5, 0.5]),
        },
        game = npl.doudizhu_tiny,
    )
    cfg = DoudizhuInferenceEngineConfig(
        update_equivalent = DoudizhuInferenceEngineConfig.UpdateEquivalent(
            enabled = True,
            routine = "update-equiv",
            search_floor = None,
        )
    )
    engine = DoudizhuInferenceEngine(policy_model = evaluator, cfg = cfg)
    with pytest.raises(ValueError, match = "Yomi belief connector"):
        _ = engine.infer(infoset_str = "L+3344|L+BR2")


def test_update_equivalent_step_distribution_is_normalized_and_uses_all_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = npl.doudizhu_tiny
    infoset_str, history_str, committed_actions = _root_infoset_and_history(game, seed = 17)
    committed_actions = committed_actions[:3]
    if len(committed_actions) < 3:
        pytest.skip("Need at least three committed root actions for rollout-budget coverage.")

    evaluator = _FakeEvaluator(
        {
            "": _make_step(committed_actions, [True, True, True], [0.7, 0.2, 0.1]),
        },
        game = game,
    )
    cfg = DoudizhuInferenceEngineConfig(
        belief_temp = 1.0,
        search_temp = 0.0,
        action_temp = 1.0,
        update_equivalent = DoudizhuInferenceEngineConfig.UpdateEquivalent(
            enabled = True,
            routine = "update-equiv",
            rollout_budget = 5,
            search_floor = None,
            search_beta = 1.0,
            seed = 9,
        ),
    )
    belief_model = _FixedBeliefModel(history_str = history_str)

    seen_rollout_counts: dict[str, int] = {}
    action_by_token = {
        1: committed_actions[0],
        2: committed_actions[1],
        3: committed_actions[2],
    }

    def _fake_evaluate_pending_policy(self, *, infoset_str: str, pending_action: str):  # noqa: ANN001
        del self, infoset_str
        if str(pending_action) != "":
            raise RuntimeError(f"Unexpected pending action in test evaluator: {pending_action!r}")
        return kagami_search.PendingPolicyEval(
            action_strs = [committed_actions[0], committed_actions[1], committed_actions[2]],
            is_commit = [True, True, True],
            action_token_rows = np.asarray([[1, 0], [2, 0], [3, 0]], dtype = np.int64),
            logits = torch.log(torch.as_tensor([0.7, 0.2, 0.1], dtype = torch.float64)),
        )

    def _fake_estimate_policy_action_values_batch(self, **kwargs) -> dict[str, float]:  # noqa: ANN003
        del self
        action_rows = np.asarray(kwargs["action_token_rows"], dtype = np.int64)
        rollout_counts = [int(v) for v in kwargs["rollout_counts"]]
        action_names = [str(v) for v in kwargs["action_strs"]]
        for action_idx, action_name in enumerate(action_names):
            token_key = int(action_rows[action_idx, 0])
            expected_action = action_by_token[token_key]
            if str(expected_action) != str(action_name):
                raise RuntimeError(
                    f"Token/action mismatch in fake batch estimator: token={token_key} expected={expected_action!r} got={action_name!r}."
                )
            seen_rollout_counts[str(action_name)] = int(rollout_counts[action_idx])
        return {
            committed_actions[0]: 1.5,
            committed_actions[1]: -0.5,
            committed_actions[2]: -1.5,
        }

    monkeypatch.setattr(
        kagami_search.PendingPolicyQueryService,
        "evaluate_pending_policy",
        _fake_evaluate_pending_policy,
    )
    monkeypatch.setattr(
        kagami_search.PolicyActionValueEstimator,
        "estimate_policy_action_values_batch",
        _fake_estimate_policy_action_values_batch,
    )

    engine = DoudizhuInferenceEngine(
        policy_model = evaluator,
        cfg = cfg,
        belief_model = belief_model,
        game = game,
    )
    dist = _row_to_items(engine.infer(infoset_str = infoset_str))
    got = {str(item["action_str"]): float(item["prob"]) for item in dist}

    assert seen_rollout_counts == {
        committed_actions[0]: 2,
        committed_actions[1]: 2,
        committed_actions[2]: 2,
    }
    assert pytest.approx(sum(got.values()), abs = 1e-12) == 1.0
    assert got[committed_actions[0]] > got[committed_actions[1]]
    assert got[committed_actions[1]] > 0.0
    assert got[committed_actions[2]] > 0.0


def test_update_equivalent_tool_side_action_loop_reaches_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = npl.doudizhu_tiny
    infoset_str, history_str, _committed_actions = _root_infoset_and_history(game, seed = 21)
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["3+", "PASS"], [False, True], [0.6, 0.4]),
            "3+": _make_step(["333", "334"], [True, True], [0.5, 0.5]),
        },
        game = game,
    )
    cfg = DoudizhuInferenceEngineConfig(
        belief_temp = 1.0,
        search_temp = 1.0,
        action_temp = 0.0,
        update_equivalent = DoudizhuInferenceEngineConfig.UpdateEquivalent(
            enabled = True,
            routine = "update-equiv",
            rollout_budget = 4,
            search_floor = None,
            search_beta = 1.0,
            seed = 5,
        ),
    )
    belief_model = _FixedBeliefModel(history_str = history_str)
    action_by_token = {
        1: "3+",
        2: "PASS",
        3: "333",
        4: "334",
    }

    def _fake_evaluate_pending_policy(self, *, infoset_str: str, pending_action: str):  # noqa: ANN001
        del self, infoset_str
        if str(pending_action) == "":
            return kagami_search.PendingPolicyEval(
                action_strs = ["3+", "PASS"],
                is_commit = [False, True],
                action_token_rows = np.asarray([[1, 0], [2, 0]], dtype = np.int64),
                logits = torch.log(torch.as_tensor([0.6, 0.4], dtype = torch.float64)),
            )
        if str(pending_action) == "3+":
            return kagami_search.PendingPolicyEval(
                action_strs = ["333", "334"],
                is_commit = [True, True],
                action_token_rows = np.asarray([[3, 0], [4, 0]], dtype = np.int64),
                logits = torch.log(torch.as_tensor([0.5, 0.5], dtype = torch.float64)),
            )
        raise RuntimeError(f"Unexpected pending action in test evaluator: {pending_action!r}")

    def _fake_estimate_policy_action_values_batch(self, **kwargs) -> dict[str, float]:  # noqa: ANN003
        del self
        action_rows = np.asarray(kwargs["action_token_rows"], dtype = np.int64)
        out: dict[str, float] = {}
        for action_idx, action_name in enumerate([str(v) for v in kwargs["action_strs"]]):
            token_key = int(action_rows[action_idx, 0])
            action_str = action_by_token[token_key]
            if str(action_name) != str(action_str):
                raise RuntimeError(
                    f"Token/action mismatch in fake batch estimator: token={token_key} expected={action_str!r} got={action_name!r}."
                )
            out[action_name] = {
                "3+": 2.0,
                "PASS": -1.0,
                "333": 0.1,
                "334": 1.0,
            }[action_name]
        return out

    monkeypatch.setattr(
        kagami_search.PendingPolicyQueryService,
        "evaluate_pending_policy",
        _fake_evaluate_pending_policy,
    )
    monkeypatch.setattr(
        kagami_search.PolicyActionValueEstimator,
        "estimate_policy_action_values_batch",
        _fake_estimate_policy_action_values_batch,
    )

    engine = DoudizhuInferenceEngine(
        policy_model = evaluator,
        cfg = cfg,
        belief_model = belief_model,
        game = game,
    )
    actions = [str(action) for action in engine.act_batch(infoset_strs = [infoset_str])]
    assert actions == ["334"]


def test_update_equivalent_step_distribution_applies_action_temp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = npl.doudizhu_tiny
    infoset_str, history_str, _committed_actions = _root_infoset_and_history(game, seed = 21)
    evaluator = _FakeEvaluator(
        {
            "": _make_step(["3+", "PASS"], [False, True], [0.6, 0.4]),
            "3+": _make_step(["333", "334"], [True, True], [0.5, 0.5]),
        },
        game = game,
    )
    cfg = DoudizhuInferenceEngineConfig(
        belief_temp = 1.0,
        search_temp = 1.0,
        action_temp = 0.0,
        update_equivalent = DoudizhuInferenceEngineConfig.UpdateEquivalent(
            enabled = True,
            routine = "update-equiv",
            rollout_budget = 4,
            search_floor = None,
            search_beta = 1.0,
            seed = 5,
        ),
    )
    belief_model = _FixedBeliefModel(history_str = history_str)
    action_by_token = {
        1: "3+",
        2: "PASS",
        3: "333",
        4: "334",
    }

    def _fake_evaluate_pending_policy(self, *, infoset_str: str, pending_action: str):  # noqa: ANN001
        del self, infoset_str
        if str(pending_action) == "":
            return kagami_search.PendingPolicyEval(
                action_strs = ["3+", "PASS"],
                is_commit = [False, True],
                action_token_rows = np.asarray([[1, 0], [2, 0]], dtype = np.int64),
                logits = torch.log(torch.as_tensor([0.6, 0.4], dtype = torch.float64)),
            )
        if str(pending_action) == "3+":
            return kagami_search.PendingPolicyEval(
                action_strs = ["333", "334"],
                is_commit = [True, True],
                action_token_rows = np.asarray([[3, 0], [4, 0]], dtype = np.int64),
                logits = torch.log(torch.as_tensor([0.5, 0.5], dtype = torch.float64)),
            )
        raise RuntimeError(f"Unexpected pending action in test evaluator: {pending_action!r}")

    def _fake_estimate_policy_action_values_batch(self, **kwargs) -> dict[str, float]:  # noqa: ANN003
        del self
        action_rows = np.asarray(kwargs["action_token_rows"], dtype = np.int64)
        out: dict[str, float] = {}
        for action_idx, action_name in enumerate([str(v) for v in kwargs["action_strs"]]):
            token_key = int(action_rows[action_idx, 0])
            action_str = action_by_token[token_key]
            if str(action_name) != str(action_str):
                raise RuntimeError(
                    f"Token/action mismatch in fake batch estimator: token={token_key} expected={action_str!r} got={action_name!r}."
                )
            out[action_name] = {
                "3+": 2.0,
                "PASS": -1.0,
                "333": 0.1,
                "334": 1.0,
            }[action_name]
        return out

    monkeypatch.setattr(
        kagami_search.PendingPolicyQueryService,
        "evaluate_pending_policy",
        _fake_evaluate_pending_policy,
    )
    monkeypatch.setattr(
        kagami_search.PolicyActionValueEstimator,
        "estimate_policy_action_values_batch",
        _fake_estimate_policy_action_values_batch,
    )

    engine = DoudizhuInferenceEngine(
        policy_model = evaluator,
        cfg = cfg,
        belief_model = belief_model,
        game = game,
    )
    dist = _row_to_items(engine.infer(infoset_str = infoset_str))
    prob_by_action = {str(item["action_str"]): float(item["prob"]) for item in dist}
    assert prob_by_action.get("3+", 0.0) == pytest.approx(1.0, rel = 0.0, abs = 1e-12)
    assert prob_by_action.get("PASS", 0.0) == pytest.approx(0.0, rel = 0.0, abs = 1e-12)


def test_update_equivalent_config_no_longer_accepts_max_depth() -> None:
    with pytest.raises(TypeError, match = "max_depth"):
        _ = DoudizhuInferenceEngineConfig.UpdateEquivalent(max_depth = 8)  # type: ignore[call-arg]


def test_update_equivalent_config_no_longer_accepts_num_determinization_samples() -> None:
    with pytest.raises(TypeError, match = "num_determinization_samples"):
        _ = DoudizhuInferenceEngineConfig.UpdateEquivalent(
            num_determinization_samples = 8,  # type: ignore[call-arg]
        )


def test_update_equivalent_config_no_longer_accepts_beam_width() -> None:
    with pytest.raises(TypeError, match = "beam_width"):
        _ = DoudizhuInferenceEngineConfig.UpdateEquivalent(beam_width = 8)  # type: ignore[call-arg]


def test_update_equivalent_config_no_longer_accepts_rollouts_per_action() -> None:
    with pytest.raises(TypeError, match = "rollouts_per_action"):
        _ = DoudizhuInferenceEngineConfig.UpdateEquivalent(
            rollouts_per_action = 8,  # type: ignore[call-arg]
        )


def test_config_rejects_negative_belief_temp() -> None:
    evaluator = _FakeEvaluator({"": _make_step(["PASS"], [True], [1.0])})
    with pytest.raises(ValueError, match = "belief_temp"):
        _ = DoudizhuInferenceEngine(
            policy_model = evaluator,
            cfg = DoudizhuInferenceEngineConfig(belief_temp = -0.1),
        )


def test_config_rejects_negative_search_temp() -> None:
    evaluator = _FakeEvaluator({"": _make_step(["PASS"], [True], [1.0])})
    with pytest.raises(ValueError, match = "search_temp"):
        _ = DoudizhuInferenceEngine(
            policy_model = evaluator,
            cfg = DoudizhuInferenceEngineConfig(search_temp = -0.1),
        )


def test_update_equivalent_config_rejects_invalid_routine_and_search_floor() -> None:
    evaluator = _FakeEvaluator({"": _make_step(["PASS"], [True], [1.0])})
    with pytest.raises(ValueError, match = "routine"):
        _ = DoudizhuInferenceEngine(
            policy_model = evaluator,
            cfg = DoudizhuInferenceEngineConfig(
                update_equivalent = DoudizhuInferenceEngineConfig.UpdateEquivalent(routine = "bogus")
            ),
        )
    with pytest.raises(ValueError, match = "search_floor"):
        _ = DoudizhuInferenceEngine(
            policy_model = evaluator,
            cfg = DoudizhuInferenceEngineConfig(
                update_equivalent = DoudizhuInferenceEngineConfig.UpdateEquivalent(
                    routine = "update-equiv",
                    search_floor = 4,
                )
            ),
        )
    _ = DoudizhuInferenceEngine(
        policy_model = evaluator,
        cfg = DoudizhuInferenceEngineConfig(
            update_equivalent = DoudizhuInferenceEngineConfig.UpdateEquivalent(
                routine = "local-subgame",
                rollout_budget = 8,
            )
        ),
    )


def test_doudizhu_local_subgame_requires_bound_critic() -> None:
    evaluator = _FakeEvaluator({"": _make_step(["PASS"], [True], [1.0])})
    engine = DoudizhuInferenceEngine(
        policy_model = evaluator,
        belief_model = object(),
        cfg = DoudizhuInferenceEngineConfig(
            update_equivalent = DoudizhuInferenceEngineConfig.UpdateEquivalent(
                enabled = True,
                routine = "local-subgame",
                rollout_budget = 4,
                search_floor = 2,
            )
        ),
    )

    with pytest.raises(ValueError, match = "CentralCritic"):
        _ = engine.infer("S")
