from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import nanopokerlib as npl
import pytest
import torch

import kagami.search as kagami_search
import kagami.search._sampler_bootstrap as sampler_bootstrap
import kagami.search.rollout_session as rollout_session
from kagami.core.types import (
    PolicyActionProb,
    PolicyWeight,
    policy_action_prob_batch_from_rows,
)
from marvel.model.tensordict import TensorDict


_PLAYER_TAGS = ("L", "P1", "P2")


def _counts_to_cards(game, counts) -> str:  # noqa: ANN001
    return "".join(str(game.token.rank_to_string(rank_idx)) * int(qty) for rank_idx, qty in enumerate(counts))


def _sample_initial_state(game, *, seed: int):  # noqa: ANN001
    env = game.Environment()
    step = env.reset(seed = seed)

    initial_hands = [[0] * int(game.NUM_RANKS) for _ in range(int(game.NUM_PLAYERS))]
    landlord_public = [0] * int(game.NUM_RANKS)
    token = game.token
    chance_steps = 0

    while int(step.player_id) < 0:
        step = env.step(game.Action())
        p1_tokens = step.observations.tokens[1]
        if len(p1_tokens) < 2:
            chance_steps += 1
            if chance_steps > int(game.max_game_horizon):
                raise RuntimeError("Exceeded chance-step cap while sampling Dou Dizhu root state.")
            continue
        receiver = int(token.decode_player(int(p1_tokens[0])))
        if receiver == 1:
            for player_id in range(int(game.NUM_PLAYERS)):
                tokens = step.observations.tokens[player_id]
                if len(tokens) < 2:
                    continue
                rank = int(token.decode_rank(int(tokens[1])))
                initial_hands[player_id][rank] += 1
        else:
            rank = int(token.decode_rank(int(step.observations.tokens[0][1])))
            initial_hands[0][rank] += 1
            landlord_public[rank] += 1

        chance_steps += 1
        if chance_steps > int(game.max_game_horizon):
            raise RuntimeError("Exceeded chance-step cap while sampling Dou Dizhu root state.")

    return env, step, initial_hands, landlord_public


def _build_infoset_str(
    game,
    initial_hands,
    landlord_public,
    *,
    view_player_id: int,
    committed_segments: list[str] | None = None,
) -> str:  # noqa: ANN001
    segments = [] if committed_segments is None else list(committed_segments)
    if int(view_player_id) == 0:
        private = [
            int(initial_hands[0][rank_idx]) - int(landlord_public[rank_idx])
            for rank_idx in range(int(game.NUM_RANKS))
        ]
    else:
        private = [int(v) for v in initial_hands[int(view_player_id)]]
    return "|".join(
        [
            f"{_PLAYER_TAGS[int(view_player_id)]}+{_counts_to_cards(game, private)}",
            f"L+{_counts_to_cards(game, landlord_public)}",
            *segments,
        ]
    )


def _sorted_actions(step) -> list[tuple[int, ...]]:  # noqa: ANN001
    return sorted(tuple(int(tok) for tok in proto.tokens) for proto in step.available_actions if float(proto.weight) > 0.0)


def _assert_step_equivalent(expected_step, actual_step) -> None:  # noqa: ANN001
    assert int(actual_step.player_id) == int(expected_step.player_id)
    assert _sorted_actions(actual_step) == _sorted_actions(expected_step)
    acting_player = int(expected_step.player_id)
    np.testing.assert_array_equal(
        np.asarray(actual_step.observations.tokens[acting_player], dtype = np.int32),
        np.asarray(expected_step.observations.tokens[acting_player], dtype = np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(actual_step.observations.channels[acting_player], dtype = np.uint8),
        np.asarray(expected_step.observations.channels[acting_player], dtype = np.uint8),
    )


def _belief_target_tokens(game, payload) -> np.ndarray:  # noqa: ANN001
    target = np.full(
        (int(game.MAX_BELIEF_ROWS), int(game.MAX_BELIEF_TOKENS)),
        fill_value = int(game.PAD_TOKEN),
        dtype = np.int64,
    )
    candidate_tokens = np.asarray(payload["candidate_tokens"], dtype = np.int64)
    target_index = np.asarray(payload["target_index"], dtype = np.int64)
    belief_length = int(payload["belief_length"])
    for row_idx in range(belief_length):
        target[row_idx] = candidate_tokens[row_idx, int(target_index[row_idx])]
    return target


def _belief_target_str(game, payload) -> str:  # noqa: ANN001
    target = _belief_target_tokens(game, payload)
    pieces: list[str] = []
    for row in target:
        owner_token = int(row[0])
        if owner_token == int(game.PAD_TOKEN):
            continue
        owner = int(game.token.decode_player(owner_token))
        rank = int(game.token.decode_rank(int(row[1])))
        pieces.append(f"{_PLAYER_TAGS[owner]}+{str(game.token.rank_to_string(rank))}")
    return "|".join(pieces)


class _FakeDecodeGame:
    MAX_GAME_HORIZON = 8

    def parse_infoset_str(self, infoset_str: str, pending_action: str = "") -> SimpleNamespace:
        del infoset_str, pending_action
        return SimpleNamespace(player_ids = np.asarray([0], dtype = np.int64))


def _policy_row(*, action_strs: list[str], is_commit: list[bool], probs: list[float]) -> PolicyActionProb:
    probs_t = torch.as_tensor(probs, dtype = torch.float64)
    commit_t = torch.as_tensor(is_commit, dtype = torch.bool)
    tensors = TensorDict(
        {
            "probs": probs_t,
            "is_commit": commit_t,
        },
        batch_size = torch.Size([len(action_strs)]),
        device = probs_t.device,
    )
    return PolicyActionProb(action_strs = list(action_strs), tensors = tensors)


def test_belief_supervision_shape_contract() -> None:
    game = npl.doudizhu_tiny
    env, step, initial_hands, landlord_public = _sample_initial_state(game, seed = 3)
    del step, initial_hands, landlord_public
    payload = env.belief_supervision()
    assert sorted(payload.keys()) == ["belief_length", "candidate_tokens", "candidate_valid", "target_index"]
    assert np.asarray(payload["candidate_tokens"], dtype = np.int32).shape == (
        int(game.MAX_BELIEF_ROWS),
        int(game.MAX_BELIEF_CANDIDATES),
        int(game.MAX_BELIEF_TOKENS),
    )
    assert np.asarray(payload["candidate_valid"], dtype = np.uint8).shape == (
        int(game.MAX_BELIEF_ROWS),
        int(game.MAX_BELIEF_CANDIDATES),
    )
    assert np.asarray(payload["target_index"], dtype = np.int32).shape == (int(game.MAX_BELIEF_ROWS),)
    assert 0 <= int(payload["belief_length"]) <= int(game.MAX_BELIEF_ROWS)


def test_sampled_determinization_reconstructs_root_state() -> None:
    game = npl.doudizhu_tiny
    env, step, initial_hands, landlord_public = _sample_initial_state(game, seed = 5)
    infoset_str = _build_infoset_str(
        game,
        initial_hands,
        landlord_public,
        view_player_id = int(step.player_id),
    )
    payload = env.belief_supervision()
    belief_str = _belief_target_str(game, payload)
    history_str = game.determinize_history_str(infoset_str, belief_str)

    assert [str(item) for item in game.infoset_str_from_history_strs([history_str])] == [infoset_str]
    _recon_env, recon_step = game.reconstruct_env(history_str, seed = 123)
    _assert_step_equivalent(step, recon_step)


def test_sampled_tokens_reject_impossible_opponent_hand_sizes() -> None:
    game = npl.doudizhu_tiny
    env, step, initial_hands, landlord_public = _sample_initial_state(game, seed = 19)
    root_infoset = _build_infoset_str(
        game,
        initial_hands,
        landlord_public,
        view_player_id = int(step.player_id),
    )
    decoded = game.parse_infoset_str(root_infoset)
    committed_action_idx = None
    for idx, (action_str, is_commit) in enumerate(
        zip(decoded.last_available_action_strs, decoded.last_available_actions.is_commit, strict = False)
    ):
        if bool(is_commit) and str(action_str).strip():
            committed_action_idx = idx
            break
    if committed_action_idx is None:
        pytest.skip("No committed root action available for peasant-turn belief validation.")

    action_str = str(decoded.last_available_action_strs[int(committed_action_idx)])
    action_tokens = [
        int(tok)
        for tok in decoded.last_available_actions.tokens[int(committed_action_idx)].tolist()
        if int(tok) != int(game.PAD_TOKEN)
    ]
    step = env.step(game.Action(tokens = action_tokens))
    assert int(step.player_id) == 1

    infoset_str = _build_infoset_str(
        game,
        initial_hands,
        landlord_public,
        view_player_id = int(step.player_id),
        committed_segments = [f"L-{action_str}"],
    )
    belief_tokens = _belief_target_tokens(game, env.belief_supervision())

    landlord_token = int(game.token.encode_player(0))
    other_peasant_token = int(game.token.encode_player(2))
    active_rows = np.where(belief_tokens[:, 0] != int(game.PAD_TOKEN))[0]
    landlord_rows = [row_idx for row_idx in active_rows if int(belief_tokens[row_idx, 0]) == landlord_token]
    if not landlord_rows:
        pytest.skip("No landlord-owned belief rows available for hand-size violation test.")

    tampered = belief_tokens.copy()
    tampered[int(landlord_rows[0]), 0] = other_peasant_token

    pieces: list[str] = []
    for row in tampered:
        owner_token = int(row[0])
        if owner_token == int(game.PAD_TOKEN):
            continue
        owner = int(game.token.decode_player(owner_token))
        rank = int(game.token.decode_rank(int(row[1])))
        pieces.append(f"{_PLAYER_TAGS[owner]}+{str(game.token.rank_to_string(rank))}")
    tampered_belief_str = "|".join(pieces)

    with pytest.raises(RuntimeError, match = "multiset|constraints"):
        game.determinize_history_str(infoset_str, tampered_belief_str)


def test_solve_update_equivalent_policy_matches_closed_form() -> None:
    qvalues = {"A": 1.25, "B": -0.5, "C": 0.0}
    pi_policy = [
        PolicyWeight(action_str = "A", prob = 0.5),
        PolicyWeight(action_str = "B", prob = 0.3),
        PolicyWeight(action_str = "C", prob = 0.2),
    ]
    search_beta = 1.75
    action_temp = 0.5

    dist = kagami_search.UpdateEquivalentSearchEngine.solve_update_equivalent_policy(
        qvalues = qvalues,
        pi_policy = pi_policy,
        search_beta = search_beta,
        action_temp = action_temp,
    )
    got = {str(item.action_str): float(item.prob) for item in dist}

    logits = {
        action.action_str: (
            search_beta * float(qvalues[action.action_str])
            + np.log(float(action.prob))
        )
        for action in pi_policy
    }
    unnormalized = {
        action: float(np.exp(float(logit) / float(action_temp)))
        for action, logit in logits.items()
    }
    total = float(sum(unnormalized.values()))
    expected = {action: float(value / total) for action, value in unnormalized.items()}

    for action_str, prob in expected.items():
        assert pytest.approx(got[action_str], rel = 1e-12, abs = 1e-12) == prob


def test_compute_update_equivalent_distribution_searches_all_actions_and_keeps_policy_prior_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_evaluate_pending_policy(self, *, infoset_str: str, pending_action: str):  # noqa: ANN001
        del self, infoset_str, pending_action
        return kagami_search.PendingPolicyEval(
            action_strs = ["A", "B", "C", "D"],
            is_commit = [True, True, True, True],
            action_token_rows = np.asarray(
                [
                    [1, 0],
                    [2, 0],
                    [3, 0],
                    [4, 0],
                ],
                dtype = np.int64,
            ),
            logits = torch.log(torch.as_tensor([0.5, 0.3, 0.15, 0.05], dtype = torch.float64)),
        )

    monkeypatch.setattr(
        kagami_search.PendingPolicyQueryService,
        "evaluate_pending_policy",
        _fake_evaluate_pending_policy,
    )

    seen_rollout_counts: dict[str, int] = {}

    def _fake_estimate_policy_action_values_batch(self, **kwargs) -> dict[str, float]:  # noqa: ANN003
        del self
        action_rows = np.asarray(kwargs["action_token_rows"], dtype = np.int64)
        rollout_counts = [int(v) for v in kwargs["rollout_counts"]]
        for action_idx, row in enumerate(action_rows.tolist()):
            seen_rollout_counts[str(int(row[0]))] = int(rollout_counts[action_idx])
        return {
            "A": 2.0,
            "B": 1.0,
            "C": 0.0,
            "D": -1.0,
        }

    monkeypatch.setattr(
        kagami_search.PolicyActionValueEstimator,
        "estimate_policy_action_values_batch",
        _fake_estimate_policy_action_values_batch,
    )

    class _DummyBeliefModel:
        def sample_determinized_histories(self, *, infoset_strs, num_samples, temperature, seed):  # noqa: ANN001
            del temperature, seed
            return [[object() for _ in range(int(num_samples))] for _ in infoset_strs]

    dist = kagami_search.UpdateEquivalentSearchEngine(
        game = _FakeDecodeGame(),
        policy_model = object(),
        action_value_estimator = kagami_search.RolloutActionValueEstimator(
            game = _FakeDecodeGame(),
            policy_model = object(),
            belief_model = _DummyBeliefModel(),
            belief_temp = 1.0,
            rollout_temp = 0.0,
            rollout_budget = 10,
        ),
        search_beta = 1.0,
        seed = 11,
        action_temp = 1.0,
    ).compute_policy_action_prob(infoset_str = "S", pending_action = "")
    got = {str(action): float(prob) for action, prob in zip(dist.action_strs, dist.probs.detach().cpu().tolist(), strict = False)}

    assert seen_rollout_counts == {"1": 3, "2": 3, "3": 3, "4": 3}
    assert pytest.approx(sum(got.values()), abs = 1e-12) == 1.0
    assert got["A"] > got["B"] > got["C"] > got["D"]
    assert pytest.approx(got["C"] / got["D"], rel = 1e-12, abs = 1e-12) == (
        np.exp(1.0 * (0.0 - (-1.0))) * (0.15 / 0.05)
    )


def test_sample_committed_action_repeats_pending_step_until_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    seen_pending_actions: list[str] = []

    class _FakeGame:
        MAX_GAME_HORIZON = 8

    class _FakePolicyModel:
        def policy_batch(self, *, infoset_strs, pending_actions, action_temps):  # noqa: ANN001
            del infoset_strs, action_temps
            pending_list = [str(item) for item in pending_actions]
            seen_pending_actions.extend(pending_list)
            if pending_list == [""]:
                return policy_action_prob_batch_from_rows(
                    [_policy_row(action_strs = ["3+"], is_commit = [False], probs = [1.0])]
                )
            if pending_list == ["3+"]:
                return policy_action_prob_batch_from_rows(
                    [_policy_row(action_strs = ["334"], is_commit = [True], probs = [1.0])]
                )
            raise RuntimeError(f"Unexpected pending actions: {pending_list!r}")

    actions = kagami_search.PendingPolicyQueryService(
        game = _FakeGame(),
        policy_model = _FakePolicyModel(),
    ).sample_committed_actions_batch(infoset_strs = ["S"], action_temp = 1.0, seed = 7)

    assert actions == ["334"]
    assert seen_pending_actions == ["", "3+"]


def test_compute_update_equivalent_distribution_applies_action_temp_across_pending_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_compute_pending_policy_action_prob(self, **kwargs):  # noqa: ANN003
        del self
        pending_action = str(kwargs["pending_action"])
        if pending_action == "":
            return [
                kagami_search.PendingPolicyAction(action_str = "3+", prob = 1.0, is_commit = False),
                kagami_search.PendingPolicyAction(action_str = "PASS", prob = 0.0, is_commit = True),
            ]
        if pending_action == "3+":
            return [
                kagami_search.PendingPolicyAction(action_str = "334", prob = 1.0, is_commit = True),
                kagami_search.PendingPolicyAction(action_str = "333", prob = 0.0, is_commit = True),
            ]
        raise RuntimeError(f"Unexpected pending action: {pending_action}")

    monkeypatch.setattr(
        kagami_search.UpdateEquivalentSearchEngine,
        "_compute_pending_policy_action_prob",
        _fake_compute_pending_policy_action_prob,
    )

    dist = kagami_search.UpdateEquivalentSearchEngine(
        game = _FakeDecodeGame(),
        policy_model = object(),
        action_value_estimator = object(),
        search_beta = 1.0,
        seed = 7,
        action_temp = 0.0,
    ).compute_policy_action_prob(infoset_str = "S", pending_action = "")

    assert dist.action_strs == ["3+"]
    assert [bool(v) for v in dist.is_commit.detach().cpu().tolist()] == [False]
    assert [float(v) for v in dist.probs.detach().cpu().tolist()] == pytest.approx([1.0], rel = 0.0, abs = 1e-12)


def test_rollout_to_terminal_fails_fast_at_game_horizon() -> None:
    class _FakeParsedStep:
        batch_size = torch.Size([1])
        outcome = {
            "is_terminal": torch.zeros(1, dtype = torch.bool),
            "rewards": torch.zeros((1, 3), dtype = torch.float64),
        }

    class _NeverDoneSampler:
        parsed_envs_step = _FakeParsedStep()

        def prepare_agent_cache(self, *, agent_model):  # noqa: ANN001
            del agent_model

        def rollout_step(self, *, agent_model, generator = None):  # noqa: ANN001
            del agent_model, generator
            return torch.zeros(1, dtype = torch.bool)

    with pytest.raises(RuntimeError, match = "MAX_GAME_HORIZON"):
        _ = rollout_session.SearchRolloutSession(
            game = object(),
            policy_model = object(),
            root_player_id = 0,
            rollout_temp = 1.0,
            max_horizon = 3,
        ).run_sampler_rollout(
            sampler = _NeverDoneSampler(),
            agent_model = object(),
        )


def test_select_policy_action_token_rows_for_env_step_uses_legal_commit_rows() -> None:
    class _FakeGame:
        PAD_TOKEN = 0

        @staticmethod
        def parse_infoset_str(infoset_str: str, pending_action: str = "") -> SimpleNamespace:
            del infoset_str, pending_action
            return SimpleNamespace(
                last_available_action_strs = ["PASS", "334", "3+"],
                last_available_actions = SimpleNamespace(
                    is_commit = np.asarray([True, True, False], dtype = bool),
                    tokens = np.asarray(
                        [
                            [11, 0, 0],
                            [3, 3, 4],
                            [3, 0, 0],
                        ],
                        dtype = np.int64,
                    ),
                ),
            )

    envs_step = SimpleNamespace(
        available_actions = SimpleNamespace(
            tokens = np.asarray(
                [
                    [[3, 3, 4], [11, 0, 0], [9, 9, 0]],
                    [[11, 0, 0], [3, 3, 4], [9, 9, 0]],
                ],
                dtype = np.int64,
            ),
            weights = np.asarray(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype = np.float64,
            ),
            is_commit = np.asarray(
                [
                    [True, True, True],
                    [True, True, True],
                ],
                dtype = bool,
            ),
        )
    )

    got = rollout_session.SearchRolloutSession(
        game = _FakeGame(),
        policy_model = object(),
        root_player_id = 0,
        rollout_temp = 1.0,
        max_horizon = 8,
    ).select_legal_action_token_rows(
        envs_step = envs_step,
        action_token_rows = np.asarray([[3, 3, 4], [3, 3, 4]], dtype = np.int64),
    )
    expected = np.asarray(
        [
            [3, 3, 4],
            [3, 3, 4],
        ],
        dtype = np.int64,
    )
    np.testing.assert_array_equal(got, expected)


def test_select_policy_action_token_rows_for_env_step_fails_on_unavailable_commit_action() -> None:
    class _FakeGame:
        PAD_TOKEN = 0

        @staticmethod
        def parse_infoset_str(infoset_str: str, pending_action: str = "") -> SimpleNamespace:
            del infoset_str, pending_action
            return SimpleNamespace(
                last_available_action_strs = ["PASS"],
                last_available_actions = SimpleNamespace(
                    is_commit = np.asarray([True], dtype = bool),
                    tokens = np.asarray([[11, 0, 0]], dtype = np.int64),
                ),
            )

    envs_step = SimpleNamespace(
        available_actions = SimpleNamespace(
            tokens = np.asarray([[[11, 0, 0]]], dtype = np.int64),
            weights = np.asarray([[1.0]], dtype = np.float64),
            is_commit = np.asarray([[True]], dtype = bool),
        )
    )
    with pytest.raises(RuntimeError, match = "not legal at reconstructed decision state"):
        _ = rollout_session.SearchRolloutSession(
            game = _FakeGame(),
            policy_model = object(),
            root_player_id = 0,
            rollout_temp = 1.0,
            max_horizon = 8,
        ).select_legal_action_token_rows(envs_step = envs_step, action_token_rows = np.asarray([[3, 3, 3]], dtype = np.int64))


def test_estimate_policy_action_values_batch_passes_pending_action_to_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def _fake_rollout_selected_action_rows(self, **kwargs) -> np.ndarray:  # noqa: ANN003
        del self
        captured["pending_action"] = str(kwargs["pending_action"])
        return np.asarray([0.5, 1.5], dtype = np.float64)

    monkeypatch.setattr(
        kagami_search.SearchRolloutSession,
        "rollout_selected_action_rows",
        _fake_rollout_selected_action_rows,
    )

    qvalues = kagami_search.PolicyActionValueEstimator(
        game = SimpleNamespace(MAX_GAME_HORIZON = 16),
        policy_model = object(),
        rollout_temp = 1.0,
    ).estimate_policy_action_values_batch(
        root_player_id = 0,
        pending_action = "3+",
        action_strs = ["A"],
        action_token_rows = np.asarray([[3, 0, 0]], dtype = np.int64),
        rollout_counts = [2],
        belief_samples = [
            SimpleNamespace(history_str = "h0"),
            SimpleNamespace(history_str = "h1"),
        ],
        seed = 7,
    )

    assert captured["pending_action"] == "3+"
    assert qvalues["A"] == pytest.approx(1.0, rel = 0.0, abs = 1e-12)


def test_prefill_rollout_buffer_decodes_infosets_with_pending_action(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_pending_actions: list[str] = []

    class _FakeGame:
        PAD_TOKEN = 0

        @staticmethod
        def infoset_str_from_history_strs(history_rows):  # noqa: ANN001
            return [f"I{idx}" for idx, _ in enumerate(history_rows)]

        @staticmethod
        def parse_infoset_strs(infoset_rows, pending_actions):  # noqa: ANN001
            del infoset_rows
            captured_pending_actions[:] = [str(item) for item in pending_actions]
            return object()

    def _fake_history_batch_to_rollout(**kwargs):  # noqa: ANN003
        del kwargs
        rollout_prefix = {
            "num_steps": torch.as_tensor([2, 1], dtype = torch.long),
            "player_ids": torch.zeros((2, 3), dtype = torch.long),
            "obs_tokens": torch.zeros((2, 3, 1, 1), dtype = torch.long),
            "obs_channels": torch.zeros((2, 3, 1, 1), dtype = torch.uint8),
            "avail_tokens": torch.zeros((2, 3, 2, 1), dtype = torch.long),
            "avail_weights": torch.zeros((2, 3, 2), dtype = torch.float32),
        }
        num_steps_np = np.asarray([2, 1], dtype = np.int64)
        player_ids_np = np.asarray([[0, 0, 0], [0, 0, 0]], dtype = np.int64)
        is_commit_np = np.asarray([[True, False], [False, True]], dtype = bool)
        avail_weights_np = np.zeros((2, 3, 2), dtype = np.float32)
        return rollout_prefix, num_steps_np, player_ids_np, is_commit_np, avail_weights_np

    monkeypatch.setattr(sampler_bootstrap, "history_batch_to_rollout", _fake_history_batch_to_rollout)

    sampler = SimpleNamespace(
        device = torch.device("cpu"),
        rollout_buffer = {
            "num_steps": torch.zeros(2, dtype = torch.long),
            "player_ids": torch.zeros((2, 4), dtype = torch.long),
            "obs_tokens": torch.zeros((2, 4, 1, 1), dtype = torch.long),
            "obs_channels": torch.zeros((2, 4, 1, 1), dtype = torch.uint8),
            "avail_tokens": torch.zeros((2, 4, 2, 1), dtype = torch.long),
            "avail_weights": torch.zeros((2, 4, 2), dtype = torch.float32),
            "avail_is_commit": torch.zeros((2, 4, 2), dtype = torch.bool),
        },
    )

    rollout_session.SearchRolloutSession(
        game = _FakeGame(),
        policy_model = object(),
        root_player_id = 0,
        rollout_temp = 1.0,
        max_horizon = 8,
    ).prefill_sampler_rollout_buffer(
        sampler = sampler,
        history_strs = ["h0", "h1"],
        pending_action = "3+",
    )

    assert captured_pending_actions == ["3+", "3+"]


def test_validate_prefilled_rollout_timeline_rejects_non_terminal_at_horizon() -> None:
    sampler = SimpleNamespace(
        parsed_envs_step = SimpleNamespace(batch_size = torch.Size([2])),
        rollout_buffer = {
            "num_steps": torch.as_tensor([4, 2], dtype = torch.long),
        },
    )
    next_step = SimpleNamespace(is_terminal = np.asarray([False, False], dtype = bool))
    with pytest.raises(RuntimeError, match = "non-terminal env-native rollout state at MAX_GAME_HORIZON"):
        rollout_session.SearchRolloutSession(
            game = object(),
            policy_model = object(),
            root_player_id = 0,
            rollout_temp = 1.0,
            max_horizon = 4,
        ).validate_prefilled_rollout_timeline(
            sampler = sampler,
            next_step = next_step,
        )


def test_validate_prefilled_rollout_timeline_allows_terminal_at_horizon() -> None:
    sampler = SimpleNamespace(
        parsed_envs_step = SimpleNamespace(batch_size = torch.Size([2])),
        rollout_buffer = {
            "num_steps": torch.as_tensor([4, 3], dtype = torch.long),
        },
    )
    next_step = SimpleNamespace(is_terminal = np.asarray([True, False], dtype = bool))
    rollout_session.SearchRolloutSession(
        game = object(),
        policy_model = object(),
        root_player_id = 0,
        rollout_temp = 1.0,
        max_horizon = 4,
    ).validate_prefilled_rollout_timeline(
        sampler = sampler,
        next_step = next_step,
    )


def test_mix_seed_is_non_negative_int32() -> None:
    mixed = kagami_search.UpdateEquivalentSearchEngine.mix_seed(0xFFFFFFFF, "rollout", "seed")
    assert isinstance(mixed, int)
    assert 0 <= int(mixed) <= 0x7FFFFFFF
