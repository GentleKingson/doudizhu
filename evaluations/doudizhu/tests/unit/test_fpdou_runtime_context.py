from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import types

from evaluations.core.socket_worker import PolicyWorkerRequest
from evaluations.doudizhu.agents import fpdou


class _DummyInfoset:
    view_player_id = "landlord"


class _DummyDeepAgent:
    def __init__(self, *args, **kwargs):
        del args
        del kwargs

    def act(self, infoset):
        del infoset
        return [3]


def _stub_policy(ckpt_dir: Path) -> fpdou._FpdouPolicy:
    policy = fpdou._FpdouPolicy.__new__(fpdou._FpdouPolicy)
    policy._cuda_visible_devices = None
    policy._cuda_device = "cpu"
    policy._eval_criteria = "winrate"
    policy._act_workers = 1
    policy._ckpt_dir = ckpt_dir
    policy._args = types.SimpleNamespace(out_conv = 64, out_res = 64, blocks = 5)
    policy._agents = {}
    policy._DeepAgent = _DummyDeepAgent
    policy._get_obs2d = lambda infoset: infoset
    return policy


def test_fpdou_cold_start_act_batch_avoids_nested_runtime_context(monkeypatch, tmp_path: Path):
    policy = _stub_policy(tmp_path)

    depth = 0
    entry_count = 0

    @contextmanager
    def fake_runtime_context(*, cuda_visible_devices: str | None):
        del cuda_visible_devices
        nonlocal depth, entry_count
        depth += 1
        entry_count += 1
        if depth > 1:
            raise AssertionError("nested _fpdou_runtime_context entry")
        try:
            yield
        finally:
            depth -= 1

    monkeypatch.setattr(fpdou, "_fpdou_runtime_context", fake_runtime_context)
    monkeypatch.setattr(fpdou.History, "from_string", lambda _state: _DummyInfoset())
    monkeypatch.setattr(fpdou, "_build_fpdou_infoset", lambda _seat, _info: object())
    monkeypatch.setattr(fpdou, "cards_to_action_string", lambda _cards: "3")

    out = policy.act_batch([PolicyWorkerRequest(request_id = 1, infoset_str = "dummy")])
    assert out == ["3"]
    assert entry_count == 1
