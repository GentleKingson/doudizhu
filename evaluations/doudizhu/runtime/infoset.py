"""Semantic DouDiZhu infoset strings (PerfectDou + nanopokerlib compatible).

Format:
  <View>+<view private hand>|L+<landlord public cards>|L-<action1>|P1-<action2>|P2-<action3>|...

Where:
  - `L`, `P1`, `P2` identify landlord, landlord_down, landlord_up.
  - The first `+` segment holds the viewing player's *initial* private hand (current + played).
    When the viewing player is the landlord, it excludes the 3 public landlord cards.
  - The second `L+` segment holds the 3 public landlord cards.
  - `-` segments list chronological actions; `PASS` marks an empty move.

The string is intentionally partial: it never includes true opponent hands.
"""

from __future__ import annotations

import collections
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from evaluations.doudizhu.runtime.protocol import CARD_ID_TO_RANK, RANK_INDEX, RANK_ORDER

REPO_ROOT = Path(".")
PERFECTDOU_ROOT = REPO_ROOT / "external" / "PerfectDou"

POSITION_ORDER = ["landlord", "landlord_down", "landlord_up"]
POSITION_TO_CODE = {"landlord": "L", "landlord_down": "P1", "landlord_up": "P2"}
CODE_TO_POSITION = {v: k for k, v in POSITION_TO_CODE.items()}
PASS_TOKEN = "PASS"
_SEGMENT_RE = re.compile(r"^(L|P1|P2)([+-])(.*)$")

FULL_DECK = [card for card in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]


def _next_position(position: str) -> str:
    if position == "landlord":
        return "landlord_down"
    if position == "landlord_down":
        return "landlord_up"
    return "landlord"


def _sort_cards(cards: Iterable[int]) -> List[int]:
    return sorted([int(card) for card in cards], key = lambda card_id: RANK_INDEX[CARD_ID_TO_RANK[int(card_id)]])


def _cards_to_str(cards: Sequence[int], *, empty_token: str = PASS_TOKEN) -> str:
    if not cards:
        return empty_token
    counts: Dict[str, int] = {}
    for card in cards:
        rank = CARD_ID_TO_RANK[int(card)]
        counts[rank] = counts.get(rank, 0) + 1
    parts = []
    for rank in RANK_ORDER:
        count = counts.get(rank, 0)
        if count:
            parts.append(rank * count)
    return "".join(parts)


def _parse_cards(raw: str) -> List[int]:
    text = str(raw).strip().upper()
    if not text or text == PASS_TOKEN:
        return []
    cards: List[int] = []
    rank_to_card = {rank: card_id for card_id, rank in CARD_ID_TO_RANK.items()}
    for ch in text:
        if ch.isspace():
            continue
        if ch not in rank_to_card:
            raise ValueError(f"Unknown rank symbol: {ch!r}")
        cards.append(int(rank_to_card[ch]))
    return cards


def _coerce_infoset_fields(
    obj,
) -> Tuple[str, Mapping[str, Sequence[int]], Mapping[str, Sequence[int]], List[List[int]], Sequence[int]]:
    if isinstance(obj, dict):
        player_position = obj.get("player_position")
        all_handcards = obj.get("all_handcards", {})
        played_cards = obj.get("played_cards", {})
        history = obj.get("card_play_action_seq", []) or []
        three_landlord_cards = obj.get("three_landlord_cards", []) or []
    else:
        player_position = getattr(obj, "player_position", None)
        all_handcards = getattr(obj, "all_handcards", {})
        played_cards = getattr(obj, "played_cards", {})
        history = getattr(obj, "card_play_action_seq", []) or []
        three_landlord_cards = getattr(obj, "three_landlord_cards", []) or []
    if player_position is None:
        raise ValueError("player_position is required to encode the infoset.")
    return (
        str(player_position),
        all_handcards,
        played_cards,
        [list(map(int, act)) for act in history],
        list(map(int, three_landlord_cards)),
    )


def _last_move(history: List[List[int]]) -> List[int]:
    if not history:
        return []
    if len(history[-1]) == 0:
        return history[-2] if len(history) > 1 else []
    return history[-1]


def _last_two_moves(history: List[List[int]]) -> List[List[int]]:
    last_two_moves = [[], []]
    for action in history[-2:]:
        last_two_moves.insert(0, action)
        last_two_moves = last_two_moves[:2]
    return last_two_moves


def _last_move_dict(history: List[List[int]], positions: List[str]) -> Dict[str, List[int]]:
    last = {pos: [] for pos in POSITION_ORDER}
    for pos, action in zip(positions, history):
        last[pos] = list(action)
    return last


def _last_pid(history: List[List[int]], positions: List[str]) -> str:
    last_player = "landlord"
    for pos, action in zip(positions, history):
        if action:
            last_player = pos
    return last_player


def _is_bomb(action: List[int]) -> bool:
    if action == [20, 30]:
        return True
    return len(action) == 4 and len(set(action)) == 1


def _bomb_num(history: List[List[int]]) -> int:
    return sum(1 for action in history if _is_bomb(action))


def _cards_left(history: List[List[int]], positions: List[str]) -> Dict[str, int]:
    counts = {"landlord": 20, "landlord_down": 17, "landlord_up": 17}
    for pos, action in zip(positions, history):
        counts[pos] -= len(action)
    return counts


def _other_hand_cards(current_hand: List[int], history: List[List[int]]) -> List[int]:
    remaining = collections.Counter(FULL_DECK)
    for card in current_hand:
        remaining[int(card)] -= 1
    for action in history:
        for card in action:
            remaining[int(card)] -= 1
    cards: List[int] = []
    for card_id, count in remaining.items():
        if count > 0:
            cards.extend([int(card_id)] * int(count))
    return _sort_cards(cards)


def _legal_actions(hand_cards: List[int], history: List[List[int]]) -> List[List[int]]:
    if str(PERFECTDOU_ROOT) not in sys.path:
        sys.path.append(str(PERFECTDOU_ROOT))
    from perfectdou.env.game import MovesGener  # type: ignore
    from perfectdou.env import move_detector as md  # type: ignore
    from perfectdou.env import move_selector as ms  # type: ignore

    mg = MovesGener(hand_cards)
    rival_move: List[int] = []
    if history:
        rival_move = history[-2] if len(history[-1]) == 0 and len(history) > 1 else history[-1]

    rival_type = md.get_move_type(rival_move)
    rival_move_type = rival_type["type"]
    rival_move_len = rival_type.get("len", 1)
    moves: List[List[int]] = []

    if rival_move_type == md.TYPE_0_PASS:
        moves = mg.gen_moves()
    elif rival_move_type == md.TYPE_1_SINGLE:
        moves = ms.filter_type_1_single(mg.gen_type_1_single(), rival_move)
    elif rival_move_type == md.TYPE_2_PAIR:
        moves = ms.filter_type_2_pair(mg.gen_type_2_pair(), rival_move)
    elif rival_move_type == md.TYPE_3_TRIPLE:
        moves = ms.filter_type_3_triple(mg.gen_type_3_triple(), rival_move)
    elif rival_move_type == md.TYPE_4_BOMB:
        moves = ms.filter_type_4_bomb(mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb(), rival_move)
    elif rival_move_type == md.TYPE_5_KING_BOMB:
        moves = []
    elif rival_move_type == md.TYPE_6_3_1:
        moves = ms.filter_type_6_3_1(mg.gen_type_6_3_1(), rival_move)
    elif rival_move_type == md.TYPE_7_3_2:
        moves = ms.filter_type_7_3_2(mg.gen_type_7_3_2(), rival_move)
    elif rival_move_type == md.TYPE_8_SERIAL_SINGLE:
        moves = ms.filter_type_8_serial_single(mg.gen_type_8_serial_single(repeat_num = rival_move_len), rival_move)
    elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
        moves = ms.filter_type_9_serial_pair(mg.gen_type_9_serial_pair(repeat_num = rival_move_len), rival_move)
    elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
        moves = ms.filter_type_10_serial_triple(mg.gen_type_10_serial_triple(repeat_num = rival_move_len), rival_move)
    elif rival_move_type == md.TYPE_11_SERIAL_3_1:
        moves = ms.filter_type_11_serial_3_1(mg.gen_type_11_serial_3_1(repeat_num = rival_move_len), rival_move)
    elif rival_move_type == md.TYPE_12_SERIAL_3_2:
        moves = ms.filter_type_12_serial_3_2(mg.gen_type_12_serial_3_2(repeat_num = rival_move_len), rival_move)
    elif rival_move_type == md.TYPE_13_4_2:
        moves = ms.filter_type_13_4_2(mg.gen_type_13_4_2(), rival_move)
    elif rival_move_type == md.TYPE_14_4_22:
        moves = ms.filter_type_14_4_22(mg.gen_type_14_4_22(), rival_move)

    if rival_move_type not in [md.TYPE_0_PASS, md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB]:
        moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()

    if rival_move:
        moves = moves + [[]]

    for move in moves:
        move.sort()
    return moves


@dataclass(frozen = True)
class History:
    view_player_id: str
    initial_hand: List[int]
    landlord_public_cards: List[int]
    history: List[List[int]]
    action_positions: List[str]
    known_initial_hands: Dict[str, List[int]] | None = None

    def with_known_hands(self, known_initial_hands: Dict[str, List[int]]) -> "History":
        return History(
            view_player_id = self.view_player_id,
            initial_hand = list(self.initial_hand),
            landlord_public_cards = list(self.landlord_public_cards),
            history = [list(map(int, action)) for action in self.history],
            action_positions = list(self.action_positions),
            known_initial_hands = {k: list(v) for k, v in known_initial_hands.items()},
        )

    @staticmethod
    def known_initial_hands_from_state_strings(infoset_strs: Mapping[str, str]) -> Dict[str, List[int]]:
        known: Dict[str, List[int]] = {}
        for state in infoset_strs.values():
            peer = History.from_string(state)
            if peer.view_player_id == "landlord":
                initial = list(peer.initial_hand) + list(peer.landlord_public_cards)
            else:
                initial = list(peer.initial_hand)
            known[peer.view_player_id] = _sort_cards(initial)
        return known

    @staticmethod
    def from_string(infoset_str: str) -> "History":
        view_player_id = None
        initial_hand: List[int] = []
        landlord_public_cards: List[int] = []
        history: List[List[int]] = []
        action_positions: List[str] = []

        for raw in str(infoset_str).split("|"):
            segment = raw.strip()
            if not segment:
                continue
            match = _SEGMENT_RE.match(segment)
            if not match:
                raise ValueError(f"Invalid segment {segment!r} in state string.")
            code, kind, payload = match.groups()
            position = CODE_TO_POSITION.get(code)
            if position is None:
                raise ValueError(f"Unknown position code {code!r} in state string.")
            cards = _parse_cards(payload)
            if kind == "+":
                if view_player_id is None:
                    view_player_id = position
                    initial_hand = cards
                else:
                    if position != "landlord":
                        raise ValueError("Only landlord public cards may be provided in a second + segment.")
                    if landlord_public_cards:
                        raise ValueError("Only one landlord public + segment is allowed in the state string.")
                    landlord_public_cards = cards
            else:
                history.append(cards)
                action_positions.append(position)

        if view_player_id is None:
            raise ValueError("State string missing + segment for viewing player.")
        if len(landlord_public_cards) != 3:
            raise ValueError("State string missing landlord public cards segment (expected '|L+...').")

        expected = "landlord"
        for position in action_positions:
            if position != expected:
                raise ValueError(f"Action positions out of order; expected {expected} but got {position}.")
            expected = _next_position(expected)

        return History(
            view_player_id = view_player_id,
            initial_hand = list(map(int, initial_hand)),
            landlord_public_cards = list(map(int, landlord_public_cards)),
            history = [list(map(int, action)) for action in history],
            action_positions = list(action_positions),
        )

    @staticmethod
    def from_perfectdou(infoset, *, public_cards: Sequence[int], strict_turn: bool = True) -> "History":
        player_position, all_handcards, played_cards, history, _ = _coerce_infoset_fields(infoset)
        if player_position not in POSITION_TO_CODE:
            raise ValueError(f"Unknown player_position: {player_position}")

        public_cards = list(map(int, public_cards))
        if len(public_cards) != 3:
            raise ValueError("public_cards must contain exactly 3 cards.")

        initial = list(all_handcards.get(player_position, [])) + list(played_cards.get(player_position, []))
        if player_position == "landlord":
            private = list(initial)
            for card in public_cards:
                try:
                    private.remove(int(card))
                except ValueError as exc:
                    raise ValueError("Landlord private hand does not contain the provided public cards.") from exc
            initial_hand = _sort_cards(private)
        else:
            initial_hand = _sort_cards(initial)

        actor = "landlord"
        positions: List[str] = []
        for _ in history:
            positions.append(actor)
            actor = _next_position(actor)

        if strict_turn and actor != player_position:
            raise ValueError(
                "infoset_str must end on the acting player's turn; "
                f"expected next={actor}, got player_position={player_position}"
            )

        return History(
            view_player_id = player_position,
            initial_hand = initial_hand,
            landlord_public_cards = _sort_cards(public_cards),
            history = [list(map(int, action)) for action in history],
            action_positions = positions,
        )

    def to_string(self) -> str:
        if self.view_player_id not in POSITION_TO_CODE:
            raise ValueError(f"Unknown view_player_id: {self.view_player_id}")
        if len(self.landlord_public_cards) != 3:
            raise ValueError("landlord_public_cards must contain exactly 3 cards.")

        pieces: List[str] = [
            f"{POSITION_TO_CODE[self.view_player_id]}+{_cards_to_str(self.initial_hand, empty_token='')}",
            f"L+{_cards_to_str(self.landlord_public_cards, empty_token='')}",
        ]
        actor = "landlord"
        for action in self.history:
            pieces.append(f"{POSITION_TO_CODE[actor]}-{_cards_to_str(action)}")
            actor = _next_position(actor)
        return "|".join(pieces)

    @property
    def next_player(self) -> str:
        actor = "landlord"
        for _ in self.history:
            actor = _next_position(actor)
        return actor

    def played_by_position(self) -> Dict[str, List[int]]:
        played: Dict[str, List[int]] = {pos: [] for pos in POSITION_ORDER}
        for pos, action in zip(self.action_positions, self.history):
            if action:
                played[pos].extend(list(map(int, action)))
        return played

    def current_hand(self) -> List[int]:
        seat = self.view_player_id
        played = self.played_by_position()
        cards = list(self.initial_hand)
        if seat == "landlord":
            cards.extend(list(self.landlord_public_cards))
        for card in played[seat]:
            cards.remove(int(card))
        return _sort_cards(cards)

    def _current_from_initial(self, initial: List[int], played: List[int]) -> List[int]:
        cards = list(initial)
        for card in played:
            cards.remove(int(card))
        return _sort_cards(cards)

    def to_perfectdou_infoset(self, *, seed: int = 0):
        """Decode into a PerfectDou InfoSet with determinized opponent hands.

        This method does not import PerfectDou's Python 3.7-only encoder extension.
        """
        if str(PERFECTDOU_ROOT) not in sys.path:
            sys.path.append(str(PERFECTDOU_ROOT))
        from perfectdou.env.game import InfoSet  # type: ignore

        seat = self.view_player_id
        played_by_pos = self.played_by_position()

        public_cards_left = list(self.landlord_public_cards)
        for card in played_by_pos["landlord"]:
            if card in public_cards_left:
                public_cards_left.remove(card)

        current_hand = self.current_hand()
        other_hand_cards = _other_hand_cards(current_hand, self.history)
        cards_left = _cards_left(self.history, self.action_positions)

        rng = random.Random(int(seed))
        all_handcards: Dict[str, List[int]] = {pos: [] for pos in POSITION_ORDER}
        all_handcards[seat] = current_hand
        other_positions = [pos for pos in POSITION_ORDER if pos != seat]
        if self.known_initial_hands and all(pos in self.known_initial_hands for pos in other_positions):
            other_hand_cards = []
            for pos in other_positions:
                current = self._current_from_initial(self.known_initial_hands[pos], played_by_pos[pos])
                all_handcards[pos] = current
                other_hand_cards.extend(list(current))
            other_hand_cards = _sort_cards(other_hand_cards)
        else:
            bag = list(other_hand_cards)
            if seat == "landlord":
                rng.shuffle(bag)
                down_count = cards_left["landlord_down"]
                all_handcards["landlord_down"] = _sort_cards(bag[:down_count])
                all_handcards["landlord_up"] = _sort_cards(bag[down_count:])
            else:
                for card in public_cards_left:
                    bag.remove(card)
                rng.shuffle(bag)
                other_farmer = "landlord_up" if seat == "landlord_down" else "landlord_down"
                landlord_need = cards_left["landlord"] - len(public_cards_left)
                landlord_cards = list(public_cards_left) + bag[:landlord_need]
                all_handcards["landlord"] = _sort_cards(landlord_cards)
                all_handcards[other_farmer] = _sort_cards(bag[landlord_need:])

        info = InfoSet(seat)
        info.player_position = seat
        info.player_hand_cards = current_hand
        info.card_play_action_seq = [list(map(int, action)) for action in self.history]
        info.legal_actions = _legal_actions(current_hand, self.history)
        info.last_move = _last_move(self.history)
        info.last_two_moves = _last_two_moves(self.history)
        info.last_move_dict = _last_move_dict(self.history, self.action_positions)
        info.num_cards_left_dict = cards_left
        info.other_hand_cards = other_hand_cards
        info.played_cards = played_by_pos
        info.three_landlord_cards = public_cards_left
        info.all_handcards = all_handcards
        info.last_pid = _last_pid(self.history, self.action_positions)
        info.bomb_num = _bomb_num(self.history)
        return info
