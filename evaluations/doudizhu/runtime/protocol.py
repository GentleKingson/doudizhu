"""Action-string helpers for the DouDiZhu evaluator."""

from __future__ import annotations

from typing import Iterable, List

RANK_ORDER = "3456789TJQKA2BR"
CARD_ID_TO_RANK = {
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
    17: "2",
    20: "B",
    30: "R",
}
RANK_TO_CARD_ID = {rank: card_id for card_id, rank in CARD_ID_TO_RANK.items()}
RANK_INDEX = {rank: idx for idx, rank in enumerate(RANK_ORDER)}


def parse_action_string(action: str) -> List[int]:
    text = action.strip().upper()
    if not text or text == "PASS":
        return []
    cards: List[int] = []
    for ch in text:
        if ch.isspace():
            continue
        if ch in {"?", "+", "."}:
            raise ValueError(f"Action '{action}' is not committed.")
        if ch not in RANK_TO_CARD_ID:
            raise ValueError(f"Unknown rank symbol '{ch}' in action '{action}'.")
        cards.append(RANK_TO_CARD_ID[ch])
    cards.sort()
    return cards


def cards_to_action_string(cards: Iterable[int]) -> str:
    counts = {rank: 0 for rank in RANK_ORDER}
    total = 0
    for card in cards:
        rank = CARD_ID_TO_RANK[int(card)]
        counts[rank] += 1
        total += 1
    if total == 0:
        return "PASS"
    parts = []
    for rank in RANK_ORDER:
        count = counts[rank]
        if count:
            parts.append(rank * count)
    return "".join(parts)
