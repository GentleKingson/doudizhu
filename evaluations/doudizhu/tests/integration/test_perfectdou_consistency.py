import copy
import random
from pathlib import Path

import pytest
import torch  # noqa: F401

from evaluations.doudizhu.runtime.infoset import History


pytestmark = [pytest.mark.integration, pytest.mark.external]

PERFECTDOU_ROOT = Path("external") / "PerfectDou"
FPDOU_ROOT = Path("external") / "fpdou"


FULL_DECK = [card for card in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]


def _make_deal(rng: random.Random) -> dict:
    cards = list(FULL_DECK)
    rng.shuffle(cards)
    return {
        "landlord": sorted(cards[:20]),
        "landlord_up": sorted(cards[20:37]),
        "landlord_down": sorted(cards[37:54]),
        "three_landlord_cards": sorted(cards[17:20]),
    }


def _socket_infoset(env, seat: str, public_cards):
    payload = {
        "player_position": seat,
        "all_handcards": {pos: list(env.info_sets[pos].player_hand_cards) for pos in ["landlord", "landlord_down", "landlord_up"]},
        "played_cards": env.played_cards,
        "card_play_action_seq": env.card_play_action_seq,
        "three_landlord_cards": env.three_landlord_cards,
    }
    return History.from_perfectdou(payload, public_cards = public_cards, strict_turn = False)


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


def test_perfectdou_socket_infoset_matches_direct_action(monkeypatch: pytest.MonkeyPatch):
    if not PERFECTDOU_ROOT.exists():
        pytest.skip("PerfectDou not available")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.syspath_prepend(str(PERFECTDOU_ROOT))

    try:
        from perfectdou.env.game import GameEnv, bombs  # type: ignore
        from perfectdou.evaluation.perfectdou_agent import PerfectDouAgent  # type: ignore
    except Exception as exc:
        pytest.skip(f"PerfectDou imports unavailable: {exc}")

    rng = random.Random(0)
    deal = _make_deal(rng)
    public_cards = list(deal["three_landlord_cards"])

    agents = {
        "landlord": PerfectDouAgent("landlord"),
        "landlord_down": PerfectDouAgent("landlord_down"),
        "landlord_up": PerfectDouAgent("landlord_up"),
    }

    env = GameEnv({})
    env.reset()
    env.card_play_init(deal)

    moves = 0
    while not env.game_over and moves < 30:
        seat = env.acting_player_position
        direct = agents[seat].act(env.game_infoset)

        info = _socket_infoset(env, seat, public_cards)
        other_states = {
            pos: _socket_infoset(env, pos, public_cards).to_string()
            for pos in ["landlord", "landlord_down", "landlord_up"]
            if pos != seat
        }
        known = History.known_initial_hands_from_state_strings(other_states)
        pd_info = info.with_known_hands(known).to_perfectdou_infoset(seed = 0)
        socket_action = agents[seat].act(pd_info)

        assert direct == socket_action
        _apply_action(env, direct, bombs)
        moves += 1


def test_fpdou_swap_peasants_invariant(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("tdigest")
    if not FPDOU_ROOT.exists():
        pytest.skip("FPDou not available")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.syspath_prepend(str(FPDOU_ROOT))

    try:
        from fpdou.env.env import Env as FpdouEnv  # type: ignore
        from fpdou.evaluation.deep_agent import DeepAgent as FpdouAgent  # type: ignore
    except Exception as exc:
        pytest.skip(f"FPDou imports unavailable: {exc}")

    args = type("Args", (), {"out_conv": 64, "out_res": 64, "blocks": 5})
    ckpt_dir = FPDOU_ROOT / "checkpoints" / "final"
    agent = FpdouAgent(
        "landlord",
        str(ckpt_dir / "landlord_weights.ckpt"),
        args,
        "cpu",
        "distributional",
        "winrate",
    )

    env = FpdouEnv("wp")
    env.reset()
    infoset = env.infoset
    assert infoset.player_position == "landlord"

    swapped = copy.deepcopy(infoset)
    down = swapped.all_handcards["landlord_down"]
    up = swapped.all_handcards["landlord_up"]
    swapped.all_handcards["landlord_down"] = up
    swapped.all_handcards["landlord_up"] = down

    action = agent.act(infoset)
    swapped_action = agent.act(swapped)
    assert action == swapped_action
