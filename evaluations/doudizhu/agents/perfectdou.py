from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Sequence

from evaluations.core.socket_worker import PolicyWorkerRequest, run_policy_worker_server
from evaluations.doudizhu.runtime.config import DEFAULT_NETWORK_SERVER, DEFAULT_PERFECTDOU_AGENT
from evaluations.doudizhu.runtime.infoset import History
from evaluations.doudizhu.runtime.protocol import cards_to_action_string


REPO_ROOT = Path(__file__).resolve().parents[3]
PERFECTDOU_ROOT = REPO_ROOT / "external" / "PerfectDou"


def _require_py37() -> None:
    if sys.version_info[:2] != (3, 7):
        raise RuntimeError("PerfectDou agent must run with Python 3.7.")


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
def _perfectdou_runtime_context():
    with _temporary_sys_path(PERFECTDOU_ROOT):
        with _temporary_cwd(PERFECTDOU_ROOT):
            yield


class _PerfectDouPolicy:
    def __init__(self) -> None:
        _require_py37()
        with _perfectdou_runtime_context():
            from perfectdou.evaluation.perfectdou_agent import PerfectDouAgent  # type: ignore

        self._PerfectDouAgent = PerfectDouAgent
        self._seed = int(DEFAULT_PERFECTDOU_AGENT.seed)
        self._agents: Dict[str, object] = {}

    def _get_agent(self, seat: str) -> object:
        if seat in self._agents:
            return self._agents[seat]
        with _perfectdou_runtime_context():
            agent = self._PerfectDouAgent(seat)
        self._agents[seat] = agent
        return agent

    def _act_one(self, seat: str, infoset) -> List[int]:
        agent = self._get_agent(seat)
        with _perfectdou_runtime_context():
            return agent.act(infoset)

    def act_batch(self, requests: Sequence[PolicyWorkerRequest]) -> List[str]:
        infosets = []
        for req in requests:
            info = History.from_string(req.infoset_str)
            if req.other_state_strings:
                known = History.known_initial_hands_from_state_strings(req.other_state_strings)
                info = info.with_known_hands(known)
            infosets.append(info)
        if not infosets:
            return []
        seat = infosets[0].view_player_id
        if any(info.view_player_id != seat for info in infosets):
            raise RuntimeError("PerfectDou policy expected a single seat per socket.")

        pd_infosets = [
            info.to_perfectdou_infoset(seed = int(self._seed) + int(req.request_id)) for req, info in zip(requests, infosets)
        ]

        action_cards_batch = [self._act_one(seat, info) for info in pd_infosets]

        actions: List[str] = []
        for info, action_cards in zip(pd_infosets, action_cards_batch):
            if action_cards not in info.legal_actions:
                raise RuntimeError("PerfectDou returned illegal action.")
            actions.append(cards_to_action_string(action_cards))
        return actions


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1:
        print("Usage: python -m evaluations.doudizhu.agents.perfectdou <socket_path>", file = sys.stderr)
        return 2
    socket_path = Path(argv[0])
    policy = _PerfectDouPolicy()
    run_policy_worker_server(
        socket_path,
        policy = policy.act_batch,
        cfg = DEFAULT_NETWORK_SERVER,
        server_label = "Dou Dizhu policy worker",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
