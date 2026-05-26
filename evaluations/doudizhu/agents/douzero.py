from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from evaluations.core.socket_worker import PolicyWorkerRequest, run_policy_worker_server
from evaluations.doudizhu.runtime.config import DEFAULT_DOUZERO_AGENT, DEFAULT_NETWORK_SERVER
from evaluations.doudizhu.runtime.infoset import History
from evaluations.doudizhu.runtime.protocol import cards_to_action_string


REPO_ROOT = Path(__file__).resolve().parents[3]
PERFECTDOU_ROOT = REPO_ROOT / "external" / "PerfectDou"


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
def _douzero_runtime_context(*, cuda_visible_devices: str | None):
    with _temporary_sys_path(PERFECTDOU_ROOT):
        with _temporary_cuda_visible_devices(cuda_visible_devices):
            yield


def _default_ckpt(position: str) -> Path:
    return (
        PERFECTDOU_ROOT
        / "perfectdou"
        / "model"
        / "douzero"
        / "douzero_ADP"
        / f"{position}.ckpt"
    )


class _DouZeroPolicy:
    def __init__(self) -> None:
        # Runner controls placement via CUDA_VISIBLE_DEVICES.
        self._cuda_visible_devices: str | None = None

        with _douzero_runtime_context(cuda_visible_devices = self._cuda_visible_devices):
            from perfectdou.evaluation.deep_agent import DeepAgent  # type: ignore
            from perfectdou.env.env import get_obs  # type: ignore

        self._DeepAgent = DeepAgent
        self._get_obs = get_obs
        self._seed = int(DEFAULT_DOUZERO_AGENT.seed)
        self._agents: Dict[str, object] = {}

    def _get_agent(self, seat: str) -> object:
        if seat in self._agents:
            return self._agents[seat]

        ckpt = _default_ckpt(seat)
        with _douzero_runtime_context(cuda_visible_devices = self._cuda_visible_devices):
            agent = self._DeepAgent(seat, str(ckpt))
        self._agents[seat] = agent
        return agent

    def _batch_actions(self, seat: str, infosets: List[object]) -> List[List[int]]:
        actions: List[List[int]] = [None] * len(infosets)  # type: ignore[list-item]
        to_eval: List[int] = []
        z_batches = []
        x_batches = []
        lengths: List[int] = []

        for idx, info in enumerate(infosets):
            if len(info.legal_actions) == 1:
                actions[idx] = info.legal_actions[0]
                continue
            obs = self._get_obs(info)
            z_batches.append(obs["z_batch"])
            x_batches.append(obs["x_batch"])
            lengths.append(len(info.legal_actions))
            to_eval.append(idx)

        if not to_eval:
            return actions  # type: ignore[return-value]

        z_batch = np.concatenate(z_batches, axis = 0)
        x_batch = np.concatenate(x_batches, axis = 0)

        agent = self._get_agent(seat)
        model = agent.model
        device = next(model.parameters()).device

        z_t = torch.from_numpy(z_batch).float().to(device)
        x_t = torch.from_numpy(x_batch).float().to(device)

        with torch.inference_mode():
            values = model.forward(z_t, x_t, return_value = True)["values"]
            values = values.squeeze(-1)

        start = 0
        for idx, length in zip(to_eval, lengths):
            end = start + length
            slice_vals = values[start:end]
            action_idx = int(torch.argmax(slice_vals).item())
            actions[idx] = infosets[idx].legal_actions[action_idx]
            start = end

        return actions  # type: ignore[return-value]

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
            raise RuntimeError("DouZero policy expected a single seat per socket.")

        pd_infosets = [
            info.to_perfectdou_infoset(seed = int(self._seed) + int(req.request_id))
            for req, info in zip(requests, infosets)
        ]

        with _douzero_runtime_context(cuda_visible_devices = self._cuda_visible_devices):
            action_cards_batch = self._batch_actions(seat, pd_infosets)

        actions: List[str] = []
        for info, action_cards in zip(pd_infosets, action_cards_batch):
            if action_cards not in info.legal_actions:
                raise RuntimeError("DouZero returned illegal action.")
            actions.append(cards_to_action_string(action_cards))
        return actions


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1:
        print("Usage: python -m evaluations.doudizhu.agents.douzero <socket_path>", file = sys.stderr)
        return 2
    socket_path = Path(argv[0])
    policy = _DouZeroPolicy()
    run_policy_worker_server(
        socket_path,
        policy = policy.act_batch,
        cfg = DEFAULT_NETWORK_SERVER,
        server_label = "Dou Dizhu policy worker",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
