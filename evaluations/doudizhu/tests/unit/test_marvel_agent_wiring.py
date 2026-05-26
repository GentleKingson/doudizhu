from __future__ import annotations

from pathlib import Path
import types
from typing import Sequence

import pytest

from evaluations.core.socket_worker import PolicyWorkerRequest
from evaluations.doudizhu.agents import marvel


def test_policy_factory_maps_temperature_args(monkeypatch, tmp_path: Path) -> None:
    torch_mod = types.ModuleType("torch")
    torch_mod.device = lambda value: value
    torch_mod.cuda = types.SimpleNamespace(is_available = lambda: False)
    monkeypatch.setattr(marvel, "torch", torch_mod)

    wired: dict[str, object] = {}

    class FakeUpdateEquivalent:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeEngineConfig:
        UpdateEquivalent = FakeUpdateEquivalent

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeEngine:
        def __init__(self, *, cfg: FakeEngineConfig) -> None:
            wired["cfg"] = cfg

        def bind_marvel_model(self, **kwargs) -> None:  # noqa: ANN003
            wired["marvel_bind"] = dict(kwargs)

        def bind_yomi_model(self, **kwargs) -> None:  # noqa: ANN003
            wired["yomi_bind"] = dict(kwargs)

        def act_batch(self, *, infoset_strs):  # noqa: ANN001
            wired["sample_engine"] = self
            wired["infoset_strs"] = list(infoset_strs)
            return ["PASS" for _ in infoset_strs]

    monkeypatch.setattr(marvel, "DoudizhuInferenceEngine", FakeEngine)
    monkeypatch.setattr(marvel, "DoudizhuInferenceEngineConfig", FakeEngineConfig)
    monkeypatch.setattr(marvel, "_load_checkpoint_game_module", lambda **_kwargs: None)

    checkpoint_path = tmp_path / "ckpt.pt"
    policy = marvel._policy_factory(
        checkpoint = str(checkpoint_path),
        mode = "direct",
        action_temp = 1.0,
        belief_temp = 0.5,
        search_temp = 0.25,
    )
    out = policy(
        [
            PolicyWorkerRequest(request_id = 1, infoset_str = "s1"),
            PolicyWorkerRequest(request_id = 2, infoset_str = "s2"),
        ]
    )

    assert out == ["PASS", "PASS"]

    engine_cfg = wired["cfg"]
    assert engine_cfg.action_temp == pytest.approx(1.0)
    assert engine_cfg.belief_temp == pytest.approx(0.5)
    assert engine_cfg.search_temp == pytest.approx(0.25)
    assert getattr(engine_cfg, "update_equivalent", None) is None

    assert wired["marvel_bind"]["checkpoint"] == str(checkpoint_path)
    assert wired["marvel_bind"]["device"] == "cpu"
    assert "yomi_bind" not in wired
    assert wired["infoset_strs"] == ["s1", "s2"]


def test_policy_factory_wires_search_update_equivalent(monkeypatch) -> None:
    torch_mod = types.ModuleType("torch")
    torch_mod.device = lambda value: value
    torch_mod.cuda = types.SimpleNamespace(is_available = lambda: False)
    monkeypatch.setattr(marvel, "torch", torch_mod)

    wired: dict[str, object] = {}

    class FakeUpdateEquivalent:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeEngineConfig:
        UpdateEquivalent = FakeUpdateEquivalent

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeEngine:
        def __init__(self, *, cfg: FakeEngineConfig) -> None:
            wired["cfg"] = cfg

        def bind_marvel_model(self, **kwargs) -> None:  # noqa: ANN003
            wired["marvel_bind"] = dict(kwargs)

        def bind_yomi_model(self, **kwargs) -> None:  # noqa: ANN003
            wired["yomi_bind"] = dict(kwargs)

        def act_batch(self, *, infoset_strs):  # noqa: ANN001
            return ["PASS" for _ in infoset_strs]

    monkeypatch.setattr(marvel, "DoudizhuInferenceEngine", FakeEngine)
    monkeypatch.setattr(marvel, "DoudizhuInferenceEngineConfig", FakeEngineConfig)
    monkeypatch.setattr(marvel, "_load_checkpoint_game_module", lambda **_kwargs: None)
    policy = marvel._policy_factory(
        checkpoint = "wandb:nanomena/marvel/run123",
        mode = "search",
        belief_checkpoint = "wandb:nanomena/yomi/run456",
        action_temp = 0.0,
        belief_temp = 1.0,
        search_temp = 1.0,
        routine = "update-equiv",
        search_beta = 2.5,
        search_budget = 1234,
        search_floor = None,
    )
    out = policy([PolicyWorkerRequest(request_id = 1, infoset_str = "s1")])

    assert out == ["PASS"]

    engine_cfg = wired["cfg"]
    assert engine_cfg.action_temp == pytest.approx(0.0)
    assert engine_cfg.update_equivalent.enabled is True
    assert engine_cfg.update_equivalent.search_beta == pytest.approx(2.5)
    assert int(engine_cfg.update_equivalent.rollout_budget) == 1234
    assert wired["marvel_bind"]["checkpoint"] == "wandb:nanomena/marvel/run123"
    assert wired["yomi_bind"]["checkpoint"] == "wandb:nanomena/yomi/run456"


def test_policy_factory_wires_local_subgame_critic(monkeypatch) -> None:
    torch_mod = types.ModuleType("torch")
    torch_mod.device = lambda value: value
    torch_mod.cuda = types.SimpleNamespace(is_available = lambda: False)
    monkeypatch.setattr(marvel, "torch", torch_mod)

    wired: dict[str, object] = {}

    class FakeUpdateEquivalent:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeEngineConfig:
        UpdateEquivalent = FakeUpdateEquivalent

        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeEngine:
        def __init__(self, *, cfg: FakeEngineConfig) -> None:
            wired["cfg"] = cfg

        def bind_marvel_model(self, **kwargs) -> None:  # noqa: ANN003
            wired["marvel_bind"] = dict(kwargs)

        def bind_marvel_critic(self, **kwargs) -> None:  # noqa: ANN003
            wired["critic_bind"] = dict(kwargs)

        def bind_yomi_model(self, **kwargs) -> None:  # noqa: ANN003
            wired["yomi_bind"] = dict(kwargs)

        def act_batch(self, *, infoset_strs):  # noqa: ANN001
            return ["PASS" for _ in infoset_strs]

    monkeypatch.setattr(marvel, "DoudizhuInferenceEngine", FakeEngine)
    monkeypatch.setattr(marvel, "DoudizhuInferenceEngineConfig", FakeEngineConfig)
    monkeypatch.setattr(marvel, "_load_checkpoint_game_module", lambda **_kwargs: None)

    policy = marvel._policy_factory(
        checkpoint = "wandb:nanomena/marvel/run123",
        mode = "search",
        belief_checkpoint = "wandb:nanomena/yomi/run456",
        routine = "local-subgame",
        search_budget = 8,
        search_floor = None,
    )
    assert policy([PolicyWorkerRequest(request_id = 1, infoset_str = "s1")]) == ["PASS"]

    engine_cfg = wired["cfg"]
    assert engine_cfg.update_equivalent.routine == "local-subgame"
    assert engine_cfg.update_equivalent.search_floor == 8
    assert wired["critic_bind"]["checkpoint"] == "wandb:nanomena/marvel/run123"
    assert wired["yomi_bind"]["checkpoint"] == "wandb:nanomena/yomi/run456"


def test_main_rejects_negative_temperature(monkeypatch, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"x")

    with pytest.raises(SystemExit):
        _ = marvel.main(
            [
                "/tmp/sock",
                str(ckpt),
                "--action-temp",
                "-1",
            ]
        )


def test_main_rejects_non_positive_search_budget(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    belief = tmp_path / "belief.pt"
    ckpt.write_bytes(b"x")
    belief.write_bytes(b"y")

    with pytest.raises(SystemExit):
        _ = marvel.main(
            [
                "/tmp/sock",
                str(ckpt),
                "--mode",
                "search",
                "--belief-checkpoint",
                str(belief),
                "--search-budget",
                "0",
            ]
        )


def test_main_rejects_search_beta_outside_search_mode(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"x")

    with pytest.raises(SystemExit):
        _ = marvel.main(
            [
                "/tmp/sock",
                str(ckpt),
                "--search-beta",
                "1",
            ]
        )
