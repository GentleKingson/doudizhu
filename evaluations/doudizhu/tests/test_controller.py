from __future__ import annotations

from pathlib import Path

import pytest

import evaluations.core.socket_worker as socket_worker_module
from evaluations.core.controller_pool import WorkerLocalControllerPool
from evaluations.core.socket_worker import PolicyWorkerRequest
from evaluations.core.types import DecisionRequest, QueueEvent
from evaluations.doudizhu.controller import DoudizhuSocketController
from evaluations.doudizhu.specs import AgentSpec


class _FakeAgentClient:
    def __init__(self) -> None:
        self.closed = False
        self.requests: list[object] = []

    def act_batch(self, requests):
        self.requests.extend(list(requests))
        return ["PASS" for _ in requests]

    def close(self) -> None:
        self.closed = True


class _FakeSocketRuntime:
    created = 0
    commands: list[list[str]] = []
    socket_names: list[str] = []

    def __init__(
        self,
        *,
        command_factory,
        cwd,
        env,
        socket_name,
        log_path,
        connect_timeout_s,
        client_factory,
        runtime_label,
    ) -> None:  # noqa: ANN001
        del cwd, env, connect_timeout_s, runtime_label
        type(self).created += 1
        Path(log_path).parent.mkdir(parents = True, exist_ok = True)
        Path(log_path).write_text("", encoding = "utf-8")
        type(self).socket_names.append(socket_worker_module._bounded_socket_name(str(socket_name)))
        type(self).commands.append(list(command_factory(Path("/tmp/fake-agent.sock"))))
        self.log_path = Path(log_path)
        self.client = _FakeAgentClient()
        del client_factory

    def close(self) -> None:
        self.client.close()

    def failure_context(self) -> str:
        return f"Log: {self.log_path}\nState: process exited with code 7\nRecent output:\nboom"


class _ExplodingAgentClient(_FakeAgentClient):
    def act_batch(self, requests):  # noqa: ANN001
        self.requests.extend(list(requests))
        raise BrokenPipeError(32, "Broken pipe")


class _ExplodingSocketRuntime(_FakeSocketRuntime):
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self.client = _ExplodingAgentClient()


def _make_request() -> DecisionRequest:
    return DecisionRequest(
        controller_id = "landlord",
        acting_entity_id = "landlord",
        episode_id = "0",
        decision_index = 0,
        payload = PolicyWorkerRequest(request_id = 0, infoset_str = "dummy"),
    )


def test_doudizhu_controller_open_close_and_act_batch(monkeypatch, tmp_path: Path) -> None:
    events: list[QueueEvent] = []
    _FakeSocketRuntime.created = 0
    _FakeSocketRuntime.commands = []
    _FakeSocketRuntime.socket_names = []
    monkeypatch.setattr(socket_worker_module, "SubprocessSocketClientRuntime", _FakeSocketRuntime)
    pool = WorkerLocalControllerPool()

    controller = DoudizhuSocketController(
        pool = pool,
        logdir = tmp_path,
        worker_id = 2,
        job_id = "job-1",
        controller_id = "landlord",
        agent = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou"),
        gpu_token = None,
        matchup_name = "DouDiZhu_perfectdou_selfplay",
        role_key = "landlord_perfectdou__peasants_perfectdou",
        role_label = "Landlord=perfectdou, Peasants=perfectdou",
        scope_label = "DouDiZhu_perfectdou_selfplay [Landlord=perfectdou, Peasants=perfectdou]",
    )
    controller.bind_queue_event_emitter(events.append)
    controller.open()

    out = controller.act_batch([_make_request()])
    controller.close()
    pool.close()

    assert list(out) == ["PASS"]
    assert any(event.job_id == "job-1" for event in events)
    assert any(event.body.get("stage") == "controller_ready" for event in events)
    assert any(
        event.body.get("scope_label") == "DouDiZhu_perfectdou_selfplay [Landlord=perfectdou, Peasants=perfectdou]"
        for event in events
    )
    assert any(event.body.get("role_key") == "landlord_perfectdou__peasants_perfectdou" for event in events)
    assert (tmp_path / "workers" / "worker_002" / "controllers" / "perfectdou" / "perfectdou" / "controller.log").is_file()
    assert _FakeSocketRuntime.commands[0][:2] == ["bash", "evaluations/doudizhu/agents/perfectdou.sh"]
    assert _FakeSocketRuntime.socket_names == ["perfectdou_perfectdou"]


def test_doudizhu_controller_reuses_worker_local_runtime_across_jobs(monkeypatch, tmp_path: Path) -> None:
    _FakeSocketRuntime.created = 0
    _FakeSocketRuntime.commands = []
    _FakeSocketRuntime.socket_names = []
    monkeypatch.setattr(socket_worker_module, "SubprocessSocketClientRuntime", _FakeSocketRuntime)
    pool = WorkerLocalControllerPool()

    common_kwargs = {
        "pool": pool,
        "logdir": tmp_path,
        "worker_id": 1,
        "controller_id": "landlord",
        "agent": AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou"),
        "gpu_token": None,
        "matchup_name": "DouDiZhu_perfectdou_selfplay",
        "role_key": "landlord_perfectdou__peasants_perfectdou",
        "role_label": "Landlord=perfectdou, Peasants=perfectdou",
        "scope_label": "DouDiZhu_perfectdou_selfplay [Landlord=perfectdou, Peasants=perfectdou]",
    }
    first = DoudizhuSocketController(job_id = "job-a", **common_kwargs)
    second = DoudizhuSocketController(job_id = "job-b", **common_kwargs)

    first.open()
    assert list(first.act_batch([_make_request()])) == ["PASS"]
    first.close()

    second.open()
    assert list(second.act_batch([_make_request()])) == ["PASS"]
    second.close()
    pool.close()

    assert _FakeSocketRuntime.created == 1


def test_doudizhu_controller_reuses_worker_local_runtime_across_controller_ids(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _FakeSocketRuntime.created = 0
    _FakeSocketRuntime.commands = []
    _FakeSocketRuntime.socket_names = []
    monkeypatch.setattr(socket_worker_module, "SubprocessSocketClientRuntime", _FakeSocketRuntime)
    pool = WorkerLocalControllerPool()

    common_kwargs = {
        "pool": pool,
        "logdir": tmp_path,
        "worker_id": 1,
        "agent": AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou"),
        "gpu_token": None,
        "matchup_name": "DouDiZhu_perfectdou_selfplay",
        "role_key": "landlord_perfectdou__peasants_perfectdou",
        "role_label": "Landlord=perfectdou, Peasants=perfectdou",
        "scope_label": "DouDiZhu_perfectdou_selfplay [Landlord=perfectdou, Peasants=perfectdou]",
    }
    landlord = DoudizhuSocketController(
        job_id = "job-a",
        controller_id = "landlord",
        **common_kwargs,
    )
    landlord_up = DoudizhuSocketController(
        job_id = "job-a",
        controller_id = "landlord_up",
        **common_kwargs,
    )

    landlord.open()
    landlord_up.open()
    assert list(landlord.act_batch([_make_request()])) == ["PASS"]
    assert list(
        landlord_up.act_batch(
            [
                DecisionRequest(
                    controller_id = "landlord_up",
                    acting_entity_id = "landlord_up",
                    episode_id = "0",
                    decision_index = 0,
                    payload = PolicyWorkerRequest(request_id = 0, infoset_str = "dummy"),
                )
            ]
        )
    ) == ["PASS"]
    landlord.close()
    landlord_up.close()
    pool.close()

    assert _FakeSocketRuntime.created == 1
    assert _FakeSocketRuntime.socket_names == ["perfectdou_perfectdou"]


def test_doudizhu_controller_surfaces_runtime_failure_context(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(socket_worker_module, "SubprocessSocketClientRuntime", _ExplodingSocketRuntime)
    pool = WorkerLocalControllerPool()

    controller = DoudizhuSocketController(
        pool = pool,
        logdir = tmp_path,
        worker_id = 2,
        job_id = "job-1",
        controller_id = "landlord",
        agent = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou"),
        gpu_token = None,
        matchup_name = "DouDiZhu_perfectdou_selfplay",
        role_key = "landlord_perfectdou__peasants_perfectdou",
        role_label = "Landlord=perfectdou, Peasants=perfectdou",
        scope_label = "DouDiZhu_perfectdou_selfplay [Landlord=perfectdou, Peasants=perfectdou]",
    )
    controller.open()

    with pytest.raises(RuntimeError) as excinfo:
        _ = controller.act_batch([_make_request()])
    assert "controller_id=landlord" in str(excinfo.value)
    assert "controller.log" in str(excinfo.value)
    assert "Broken pipe" in str(excinfo.value)

    controller.close()
    pool.close()


def test_doudizhu_controller_bounds_long_socket_identifier(monkeypatch, tmp_path: Path) -> None:
    _FakeSocketRuntime.created = 0
    _FakeSocketRuntime.commands = []
    _FakeSocketRuntime.socket_names = []
    monkeypatch.setattr(socket_worker_module, "SubprocessSocketClientRuntime", _FakeSocketRuntime)
    pool = WorkerLocalControllerPool()

    agent = AgentSpec(
        kind = "marvel_search",
        canonical_spec = (
            "marvel@yomi=wandb:entity/marvel/very_long_policy_run_identifier"
            "@wandb:entity/yomi/very_long_belief_run_identifier"
            "?action_temp=0&search_temp=0&search_beta=1&search_budget=2048"
        ),
        checkpoint = "wandb:entity/marvel/very_long_policy_run_identifier",
        belief_checkpoint = "wandb:entity/yomi/very_long_belief_run_identifier",
    )
    controller = DoudizhuSocketController(
        pool = pool,
        logdir = tmp_path,
        worker_id = 0,
        job_id = "job-long",
        controller_id = "landlord",
        agent = agent,
        gpu_token = None,
        matchup_name = "DouDiZhu_long_matchup",
        role_key = "landlord_long__peasants_long",
        role_label = "Landlord=long, Peasants=long",
        scope_label = "DouDiZhu_long_matchup [Landlord=long, Peasants=long]",
    )
    controller.open()
    controller.close()
    pool.close()

    assert len(_FakeSocketRuntime.socket_names) == 1
    assert len(_FakeSocketRuntime.socket_names[0]) <= 24
