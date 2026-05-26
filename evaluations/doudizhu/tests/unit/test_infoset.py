from pathlib import Path

import pytest
import torch  # noqa: F401

from evaluations.doudizhu.runtime.infoset import History


PERFECTDOU_ROOT = Path("external") / "PerfectDou"


def _require_perfectdou(monkeypatch: pytest.MonkeyPatch) -> None:
    if not PERFECTDOU_ROOT.exists():
        pytest.skip("PerfectDou not available")
    monkeypatch.syspath_prepend(str(PERFECTDOU_ROOT))
    try:
        from perfectdou.env.game import InfoSet  # type: ignore  # noqa: F401
    except Exception as exc:
        pytest.skip(f"PerfectDou imports unavailable: {exc}")


def test_infoset_from_perfectdou_encodes_expected():
    infoset = {
        "player_position": "landlord",
        "all_handcards": {
            "landlord": [3, 4, 4, 5, 30],
            "landlord_down": [6, 6, 7, 8],
            "landlord_up": [9, 10, 17],
        },
        "played_cards": {
            "landlord": [20],
            "landlord_down": [],
            "landlord_up": [17],
        },
        "three_landlord_cards": [3, 4, 30],
        "card_play_action_seq": [[20], [], [17]],
    }

    infoset_str = History.from_perfectdou(
        infoset,
        public_cards = infoset["three_landlord_cards"],
        strict_turn = True,
    ).to_string()
    assert infoset_str == "L+45B|L+34R|L-B|P1-PASS|P2-2"


def test_infoset_string_roundtrip():
    infoset_str = "P1+345|L+2BR|L-3|P1-PASS|P2-4"
    decoded = History.from_string(infoset_str)
    assert decoded.to_string() == infoset_str
    assert decoded.view_player_id == "landlord_down"
    assert decoded.next_player == "landlord"


@pytest.mark.external
def test_infoset_to_perfectdou_infoset_determinizes_opponents(monkeypatch: pytest.MonkeyPatch):
    _require_perfectdou(monkeypatch)

    # Landlord public cards include a card that gets played in history.
    infoset_str = "P2+88889999TTTTJJJJQ|L+777|L-7|P1-PASS"
    infoset = History.from_string(infoset_str)
    pd = infoset.to_perfectdou_infoset(seed = 123)

    assert pd.player_position == "landlord_up"
    assert pd.player_hand_cards == infoset.current_hand()
    assert len(pd.three_landlord_cards) == 2
    assert set(pd.three_landlord_cards).issubset(set(pd.all_handcards["landlord"]))

    played = pd.played_cards
    assert sum(len(v) for v in played.values()) == 1
    assert len(pd.all_handcards["landlord"]) == pd.num_cards_left_dict["landlord"]
    assert len(pd.all_handcards["landlord_down"]) == pd.num_cards_left_dict["landlord_down"]
    assert len(pd.all_handcards["landlord_up"]) == pd.num_cards_left_dict["landlord_up"]


@pytest.mark.external
def test_infoset_roundtrip_keeps_fpdou_policy(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    pytest.importorskip("tdigest")

    perfectdou_root = PERFECTDOU_ROOT
    fpdou_root = Path("external") / "fpdou"
    if not perfectdou_root.exists():
        pytest.skip("PerfectDou not available")
    if not fpdou_root.exists():
        pytest.skip("FPDou not available")
    monkeypatch.syspath_prepend(str(perfectdou_root))
    monkeypatch.syspath_prepend(str(fpdou_root))

    try:
        from fpdou.env.env import Env as FpdouEnv
        from fpdou.evaluation.deep_agent import DeepAgent as FpdouAgent
    except Exception as exc:
        pytest.skip(f"FPDou imports unavailable: {exc}")

    from evaluations.doudizhu.agents.fpdou import _build_fpdou_infoset

    args = type("Args", (), {"out_conv": 64, "out_res": 64, "blocks": 5})
    ckpt_dir = fpdou_root / "checkpoints" / "final"
    agents = {
        "landlord": FpdouAgent(
            "landlord",
            str(ckpt_dir / "landlord_weights.ckpt"),
            args,
            "cpu",
            "distributional",
            "winrate",
        ),
        "landlord_up": FpdouAgent(
            "landlord_up",
            str(ckpt_dir / "landlord_up_weights.ckpt"),
            args,
            "cpu",
            "distributional",
            "winrate",
        ),
        "landlord_down": FpdouAgent(
            "landlord_down",
            str(ckpt_dir / "landlord_down_weights.ckpt"),
            args,
            "cpu",
            "distributional",
            "winrate",
        ),
    }

    env = FpdouEnv("wp")
    env.reset()
    public_cards = list(env._env.three_landlord_cards)

    steps = 0
    is_terminal = False
    while not is_terminal and steps < 50:
        infoset = env.infoset
        seat = infoset.player_position

        socket_info = History.from_perfectdou(
            infoset, public_cards = public_cards, strict_turn = True
        )
        roundtrip = History.from_string(socket_info.to_string())

        fpdou_info_direct = _build_fpdou_infoset(seat, socket_info)
        fpdou_info_round = _build_fpdou_infoset(seat, roundtrip)

        action_direct = agents[seat].act(fpdou_info_direct)
        action_round = agents[seat].act(fpdou_info_round)
        assert action_direct == action_round

        _, _, is_terminal, _ = env.step(action_direct)
        steps += 1
