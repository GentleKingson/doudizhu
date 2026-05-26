from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evaluations.doudizhu.runtime.infoset import History
from evaluations.doudizhu.runtime.protocol import parse_action_string


REPO_ROOT = Path(".")
PERFECTDOU_ROOT = REPO_ROOT / "external" / "PerfectDou"

POSITIONS = ["landlord", "landlord_down", "landlord_up"]


def _apply_action(env, action: List[int]) -> None:
    from perfectdou.env.game import bombs  # type: ignore

    if len(action) > 0:
        env.last_pid = env.acting_player_position

    if action in bombs:
        env.bomb_num += 1

    env.last_move_dict[env.acting_player_position] = action.copy()
    env.card_play_action_seq.append(action)
    env.update_acting_player_hand_cards(action)
    env.played_cards[env.acting_player_position] += action

    if env.acting_player_position == "landlord" and len(action) > 0 and env.three_landlord_cards:
        for card in action:
            if env.three_landlord_cards and card in env.three_landlord_cards:
                env.three_landlord_cards.remove(card)
            if not env.three_landlord_cards:
                break

    env.game_done()
    if not env.game_over:
        env.get_acting_player_position()
        env.game_infoset = env.get_infoset()


def _landlord_score(env) -> float:
    if env.player_utility_dict is None:
        raise RuntimeError("Missing player_utility_dict; game may not be finished.")
    mult = 2 ** int(env.bomb_num)
    return 2.0 * mult if env.player_utility_dict["landlord"] > 0 else -2.0 * mult


@dataclass
class StepResult:
    next_player: Optional[str]
    infoset: Optional[History]


class DoudizhuGame:
    def __init__(self, card_play_data: Dict[str, object]) -> None:
        import sys

        if str(PERFECTDOU_ROOT) not in sys.path:
            sys.path.append(str(PERFECTDOU_ROOT))
        from perfectdou.env.game import GameEnv  # type: ignore

        self._env = GameEnv({})
        self._env.reset()
        self._env.card_play_init(card_play_data)
        self._public_cards = list(card_play_data.get("three_landlord_cards") or [])
        if len(self._public_cards) != 3:
            raise ValueError("card_play_data must contain exactly 3 three_landlord_cards.")

    @property
    def env(self):
        return self._env

    @property
    def public_cards(self) -> List[int]:
        return list(self._public_cards)

    def is_terminal(self) -> bool:
        return bool(self._env.game_over)

    def current_player(self) -> str:
        return str(self._env.acting_player_position)

    def current_infoset(self, *, strict_turn: bool = True) -> History:
        return History.from_perfectdou(self._env.game_infoset, public_cards = self._public_cards, strict_turn = strict_turn)

    def init(self) -> Tuple[str, History]:
        return self.current_player(), self.current_infoset(strict_turn = True)

    def infoset_for(self, view_player_id: str, *, strict_turn: bool = False) -> History:
        all_handcards = {pos: list(self._env.info_sets[pos].player_hand_cards) for pos in POSITIONS}
        payload = {
            "player_position": view_player_id,
            "all_handcards": all_handcards,
            "played_cards": self._env.played_cards,
            "card_play_action_seq": self._env.card_play_action_seq,
            "three_landlord_cards": self._env.three_landlord_cards,
        }
        return History.from_perfectdou(payload, public_cards = self._public_cards, strict_turn = strict_turn)

    def step(self, action: str) -> StepResult:
        if self.is_terminal():
            return StepResult(next_player = None, infoset = None)

        raw_action = str(action)
        action_cards = parse_action_string(raw_action)
        infoset = self._env.game_infoset
        if action_cards not in infoset.legal_actions:
            raise RuntimeError(
                f"Illegal action from {infoset.player_position}: '{raw_action}' -> {action_cards} "
                f"not in legal_actions ({len(infoset.legal_actions)} candidates)."
            )

        _apply_action(self._env, action_cards)

        if self.is_terminal():
            return StepResult(next_player = None, infoset = None)

        return StepResult(next_player = self.current_player(), infoset = self.current_infoset(strict_turn = True))

    def landlord_score(self) -> float:
        if not self.is_terminal():
            raise RuntimeError("landlord_score() called before game over.")
        return float(_landlord_score(self._env))
