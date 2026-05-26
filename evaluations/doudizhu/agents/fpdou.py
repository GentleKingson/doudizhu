from __future__ import annotations

import collections
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from evaluations.core.socket_worker import PolicyWorkerRequest, run_policy_worker_server
from evaluations.doudizhu.runtime.config import DEFAULT_FPDOU_AGENT, DEFAULT_NETWORK_SERVER
from evaluations.doudizhu.runtime.infoset import History
from evaluations.doudizhu.runtime.protocol import CARD_ID_TO_RANK, RANK_INDEX, cards_to_action_string


REPO_ROOT = Path(__file__).resolve().parents[3]
FPDOU_ROOT = REPO_ROOT / "external" / "fpdou"

POSITIONS = ["landlord", "landlord_down", "landlord_up"]
FULL_DECK = [card for card in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]


@contextmanager
def _temporary_sys_path(path: Path):
    path_str = str(path)
    original = list(sys.path)
    try:
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
        yield
    finally:
        sys.path[:] = original


@contextmanager
def _temporary_cwd(path: Path):
    original_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


@contextmanager
def _temporary_cuda_visible_devices(value: str | None):
    if value is None:
        yield
        return

    had_original = "CUDA_VISIBLE_DEVICES" in os.environ
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    try:
        yield
    finally:
        if had_original and original is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)


@contextmanager
def _fpdou_runtime_context(*, cuda_visible_devices: str | None):
    with _temporary_sys_path(FPDOU_ROOT):
        with _temporary_cwd(FPDOU_ROOT):
            with _temporary_cuda_visible_devices(cuda_visible_devices):
                yield


def _sort_cards(cards: Iterable[int]) -> List[int]:
    return sorted([int(card) for card in cards], key = lambda card_id: RANK_INDEX[CARD_ID_TO_RANK[int(card_id)]])


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
    last = {pos: [] for pos in POSITIONS}
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
    from fpdou.env.game import MovesGener  # type: ignore
    from fpdou.env import move_detector as md  # type: ignore
    from fpdou.env import move_selector as ms  # type: ignore

    mg = MovesGener(hand_cards)
    rival_move = []
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


def _ensure_lightweight_dmc_imports() -> None:
    pkg_name = "fpdou.dmc"
    if pkg_name in sys.modules:
        return
    dmc_path = (FPDOU_ROOT / "fpdou" / "dmc").resolve()
    if not dmc_path.is_dir():
        return
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(dmc_path)]
    sys.modules[pkg_name] = pkg


def _parse_device(value: str):
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        text = "0" if torch.cuda.is_available() else "cpu"
    if text == "cpu":
        return "cpu"
    return int(text)


def _cuda_visible_devices_override(value: str) -> str | None:
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return None
    if text == "cpu":
        return ""
    return str(int(text))


def _default_ckpt(position: str, ckpt_dir: Path) -> Path:
    if position == "landlord":
        return ckpt_dir / "landlord_weights.ckpt"
    return ckpt_dir / f"{position}_weights.ckpt"


def _build_fpdou_infoset(seat: str, infoset: History):
    if infoset.view_player_id != seat:
        raise ValueError(f"Expected view_player_id={seat}, got {infoset.view_player_id}.")

    from fpdou.env.game import InfoSet  # type: ignore

    history = infoset.history
    positions = infoset.action_positions

    played_by_pos = {pos: [] for pos in POSITIONS}
    for pos, action in zip(positions, history):
        if action:
            played_by_pos[pos].extend(action)

    three_landlord_cards = list(infoset.landlord_public_cards)
    for card in played_by_pos["landlord"]:
        if card in three_landlord_cards:
            three_landlord_cards.remove(card)

    current_hand = infoset.current_hand()
    other_hand_cards = _other_hand_cards(current_hand, history)

    info = InfoSet(seat)
    info.player_position = seat
    info.player_hand_cards = current_hand
    info.card_play_action_seq = history
    info.legal_actions = _legal_actions(current_hand, history)
    info.last_move = _last_move(history)
    info.last_two_moves = _last_two_moves(history)
    info.last_move_dict = _last_move_dict(history, positions)
    info.num_cards_left_dict = _cards_left(history, positions)
    info.other_hand_cards = other_hand_cards
    info.played_cards = played_by_pos
    info.three_landlord_cards = three_landlord_cards
    info.last_pid = _last_pid(history, positions)
    info.bomb_num = _bomb_num(history)
    info.all_handcards = {pos: [] for pos in POSITIONS}
    info.all_handcards[seat] = current_hand
    return info


class _FpdouPolicy:
    def __init__(self) -> None:
        # Runner controls placement via CUDA_VISIBLE_DEVICES.
        cuda_device = "auto"
        self._cuda_visible_devices = _cuda_visible_devices_override(cuda_device)
        self._cuda_device = _parse_device(cuda_device)
        self._eval_criteria = str(DEFAULT_FPDOU_AGENT.eval_criteria)
        self._ckpt_dir = Path(DEFAULT_FPDOU_AGENT.checkpoint_dir).resolve()
        self._args = types.SimpleNamespace(
            out_conv = int(DEFAULT_FPDOU_AGENT.out_conv),
            out_res = int(DEFAULT_FPDOU_AGENT.out_res),
            blocks = int(DEFAULT_FPDOU_AGENT.blocks),
        )

        self._agents: Dict[str, object] = {}

        with _fpdou_runtime_context(cuda_visible_devices = self._cuda_visible_devices):
            _ensure_lightweight_dmc_imports()
            from fpdou.evaluation.deep_agent import DeepAgent  # type: ignore
            from fpdou.env.env import get_obs2d  # type: ignore

        self._DeepAgent = DeepAgent
        self._get_obs2d = get_obs2d

    def _get_agent(self, seat: str, *, runtime_context_active: bool = False) -> object:
        if seat in self._agents:
            return self._agents[seat]

        ckpt = _default_ckpt(seat, self._ckpt_dir)
        if not runtime_context_active:
            with _fpdou_runtime_context(cuda_visible_devices = self._cuda_visible_devices):
                agent = self._DeepAgent(
                    seat,
                    str(ckpt),
                    self._args,
                    self._cuda_device,
                    "distributional",
                    self._eval_criteria,
                )
        else:
            agent = self._DeepAgent(
                seat,
                str(ckpt),
                self._args,
                self._cuda_device,
                "distributional",
                self._eval_criteria,
            )
        self._agents[seat] = agent
        return agent

    def _batch_actions(
        self, seat: str, infosets: List[object], *, runtime_context_active: bool = False
    ) -> List[List[int]]:
        actions: List[List[int]] = [None] * len(infosets)  # type: ignore[list-item]
        to_eval: List[int] = []
        x_batches = []
        lengths: List[int] = []

        for idx, info in enumerate(infosets):
            if len(info.legal_actions) == 1:
                actions[idx] = info.legal_actions[0]
                continue
            obs = self._get_obs2d(info)
            x_batches.append(obs["x_batch"])
            lengths.append(len(info.legal_actions))
            to_eval.append(idx)

        if not to_eval:
            return actions  # type: ignore[return-value]

        x_batch = np.concatenate(x_batches, axis = 0)

        agent = self._get_agent(seat, runtime_context_active = runtime_context_active)
        model = agent.model
        device = next(model.parameters()).device

        x_t = torch.from_numpy(x_batch).float().to(device)

        with torch.inference_mode():
            out = model.forward(x_t, return_value = True)
            logits = out["values"]
            probs = torch.softmax(logits, dim = 1)
            if self._eval_criteria == "winrate":
                half = probs.shape[1] // 2
                metric = probs[:, half:].sum(dim = 1)
            else:
                score = model.score.to(probs.device)
                metric = (probs * score).sum(dim = 1)

        start = 0
        for idx, length in zip(to_eval, lengths):
            end = start + length
            slice_vals = metric[start:end]
            action_idx = int(torch.argmax(slice_vals).item())
            actions[idx] = infosets[idx].legal_actions[action_idx]
            start = end

        return actions  # type: ignore[return-value]

    def act_batch(self, requests: Sequence[PolicyWorkerRequest]) -> List[str]:
        infosets: List[History] = []
        for req in requests:
            info = History.from_string(req.infoset_str)
            if req.other_state_strings:
                known = History.known_initial_hands_from_state_strings(req.other_state_strings)
                info = info.with_known_hands(known)
            infosets.append(info)
        actions: List[str] = []
        if not infosets:
            return actions

        seat = infosets[0].view_player_id
        if any(info.view_player_id != seat for info in infosets):
            raise RuntimeError("FPDou policy expected a single seat per socket.")

        with _fpdou_runtime_context(cuda_visible_devices = self._cuda_visible_devices):
            batch_infosets = [_build_fpdou_infoset(seat, info) for info in infosets]
            if len(batch_infosets) > 1:
                action_cards_batch = self._batch_actions(
                    seat,
                    batch_infosets,
                    runtime_context_active = True,
                )
            else:
                agent = self._get_agent(seat, runtime_context_active = True)
                action_cards_batch = [agent.act(batch_infosets[0])]

        for action_cards in action_cards_batch:
            actions.append(cards_to_action_string(action_cards))
        return actions


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1:
        print("Usage: python -m evaluations.doudizhu.agents.fpdou <socket_path>", file = sys.stderr)
        return 2
    socket_path = Path(argv[0])
    policy = _FpdouPolicy()
    run_policy_worker_server(
        socket_path,
        policy = policy.act_batch,
        cfg = DEFAULT_NETWORK_SERVER,
        server_label = "Dou Dizhu policy worker",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
