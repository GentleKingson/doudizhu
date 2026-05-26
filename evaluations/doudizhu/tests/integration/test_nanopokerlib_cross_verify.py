import pickle
import random
from pathlib import Path

import numpy as np
import pytest

import nanopokerlib as npl

from evaluations.doudizhu.runtime.infoset import History
from evaluations.doudizhu.runtime.protocol import RANK_ORDER, cards_to_action_string


pytestmark = [pytest.mark.integration, pytest.mark.external]

PERFECTDOU_ROOT = Path("external") / "PerfectDou"


def _import_perfectdou(monkeypatch: pytest.MonkeyPatch):
    if not PERFECTDOU_ROOT.exists():
        pytest.skip("external/PerfectDou not available; skipping cross-verification.")
    monkeypatch.syspath_prepend(str(PERFECTDOU_ROOT))
    from perfectdou.env.game import GameEnv, bombs  # type: ignore

    return GameEnv, bombs


def _apply_action(env, action, bombs) -> None:
    action = list(action)
    if action:
        env.last_pid = env.acting_player_position
    if action in bombs:
        env.bomb_num += 1
    env.last_move_dict[env.acting_player_position] = action.copy()
    env.card_play_action_seq.append(action)
    env.update_acting_player_hand_cards(action)
    env.played_cards[env.acting_player_position] += action

    if env.acting_player_position == "landlord" and action and env.three_landlord_cards:
        for card in action:
            if env.three_landlord_cards and card in env.three_landlord_cards:
                env.three_landlord_cards.remove(card)
            if not env.three_landlord_cards:
                break

    env.game_done()
    if not env.game_over:
        env.get_acting_player_position()
        env.game_infoset = env.get_infoset()


def _committed_actions_from_nanopokerlib(game, infoset_str: str) -> set[str]:
    committed: set[str] = set()
    stack = [""]
    visited: set[str] = set()

    while stack:
        pending = stack.pop()
        if pending in visited:
            continue
        visited.add(pending)
        infoset = game.parse_infoset_str(infoset_str, pending)
        t = len(infoset) - 1
        if t < 0:
            raise AssertionError("nanopokerlib infoset has no steps.")

        last = infoset.last_available_actions
        tokens = last.tokens
        strings = [str(action_str) for action_str in infoset.last_available_action_strs]
        if len(strings) != int(tokens.shape[0]):
            raise AssertionError(
                "Action-string alignment mismatch while exploring pending actions "
                f"(num_strings={len(strings)}, num_actions={int(tokens.shape[0])})."
            )
        weights = last.weights
        commits = last.is_commit

        for idx, action_str in enumerate(strings):
            if not action_str:
                continue
            if float(weights[idx]) <= 0.0:
                continue
            if bool(commits[idx]):
                committed.add(action_str)
            else:
                stack.append(action_str)

        if len(visited) > 4096:
            raise AssertionError("Exceeded pending-action exploration limit; possible loop or blow-up.")

    return committed


def _canonical_action_string(action: str) -> str:
    text = (action or "").strip().upper()
    if not text or text == "PASS":
        return "PASS"
    rank_index = {rank: idx for idx, rank in enumerate(RANK_ORDER)}
    return "".join(sorted(text, key = lambda ch: rank_index[ch]))


def test_socket_eval_state_string_matches_nanopokerlib(monkeypatch: pytest.MonkeyPatch):
    GameEnv, bombs = _import_perfectdou(monkeypatch)

    eval_data = PERFECTDOU_ROOT / "eval_data_tiny.pkl"
    if not eval_data.is_file():
        pytest.skip("PerfectDou eval_data_tiny.pkl not available.")
    with eval_data.open("rb") as f:
        deals = pickle.load(f)

    rng = random.Random(0)
    max_deals = min(3, len(deals))
    max_moves_per_deal = 25

    for deal_idx in range(max_deals):
        card_play_data = deals[deal_idx]
        public_cards = list(card_play_data.get("three_landlord_cards") or [])
        assert len(public_cards) == 3

        env = GameEnv({})
        env.reset()
        env.card_play_init(card_play_data)

        moves = 0
        while not env.game_over and moves < max_moves_per_deal:
            infoset = env.game_infoset
            infoset_str = History.from_perfectdou(
                {
                    "player_position": infoset.player_position,
                    "all_handcards": infoset.all_handcards,
                    "played_cards": infoset.played_cards,
                    "card_play_action_seq": infoset.card_play_action_seq,
                    "three_landlord_cards": public_cards,
                },
                public_cards = public_cards,
                strict_turn = True,
            ).to_string()

            perfectdou_actions = {
                _canonical_action_string(cards_to_action_string(a)) for a in infoset.legal_actions
            }
            nanopokerlib_actions = {
                _canonical_action_string(a)
                for a in _committed_actions_from_nanopokerlib(npl.doudizhu, infoset_str)
            }
            assert perfectdou_actions == nanopokerlib_actions

            action = rng.choice(infoset.legal_actions)
            _apply_action(env, action, bombs)
            moves += 1
