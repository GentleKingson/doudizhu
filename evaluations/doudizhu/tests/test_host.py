from __future__ import annotations

from pathlib import Path

from evaluations.core import SharedUnitAllocator
from evaluations.core.types import DecisionBatch, Finished, ProgressUpdate
from evaluations.doudizhu.host import DoudizhuHostSession
from evaluations.doudizhu.runtime.config import DoudizhuHostConfig
from evaluations.doudizhu.specs import AgentSpec, SeatingPlan


class _FakeInfoset:
    def __init__(self, text: str) -> None:
        self._text = text

    def to_string(self) -> str:
        return self._text


class _FakeEnv:
    def __init__(self) -> None:
        self.player_utility_dict = {"landlord": 1}


class _FakeGame:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = dict(payload)
        self._steps_remaining = int(self._payload["steps"])
        self._score = float(self._payload["score"])
        self.env = _FakeEnv()

    def current_player(self) -> str:
        return "landlord"

    def current_infoset(self, *, strict_turn: bool = True) -> _FakeInfoset:
        del strict_turn
        return _FakeInfoset(f"fake-{self._steps_remaining}")

    def step(self, action: str) -> None:
        del action
        self._steps_remaining -= 1

    def is_terminal(self) -> bool:
        return self._steps_remaining <= 0

    def landlord_score(self) -> float:
        return self._score


def _make_host(
    *,
    job_id: str,
    logical_job_id: str,
    plan: SeatingPlan,
    card_play_data: list[dict[str, object]],
    num_games: int,
    max_envs: int,
    progress_interval: int,
    work_allocator: SharedUnitAllocator,
) -> DoudizhuHostSession:
    return DoudizhuHostSession(
        job_id = job_id,
        logical_job_id = logical_job_id,
        plan = plan,
        card_play_data = card_play_data,
        num_games = num_games,
        host_cfg = DoudizhuHostConfig(
            num_games = num_games,
            max_envs = max_envs,
            progress_interval = progress_interval,
            ci_multiplier = 1.0,
            perfect_info = False,
        ),
        work_allocator = work_allocator,
    )


def test_host_emits_progress_once_per_completed_milestone(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evaluations.doudizhu.host.DoudizhuGame", _FakeGame)

    card_play_data = [
        {"steps": 1, "score": 2.0},
        {"steps": 2, "score": -2.0},
    ]
    agent = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    plan = SeatingPlan(
        matchup_id = "perfectdou_selfplay",
        matchup_name = "DouDiZhu_perfectdou_selfplay",
        mode = "selfplay",
        role_pass = "selfplay",
        left_agent = agent,
        right_agent = agent,
        landlord_agent = agent,
        landlord_down_agent = agent,
        landlord_up_agent = agent,
    )
    host = _make_host(
        job_id = "perfectdou_selfplay",
        logical_job_id = "perfectdou_selfplay",
        plan = plan,
        card_play_data = card_play_data,
        num_games = 2,
        max_envs = 1,
        progress_interval = 1,
        work_allocator = SharedUnitAllocator(total_units = 2),
    )

    host.open()
    opened = host.next_message()
    assert opened.body["stage"] == "host_opened"
    assert opened.body["logical_job_id"] == "perfectdou_selfplay"

    batch1 = host.next_message()
    assert isinstance(batch1, DecisionBatch)
    host.submit_actions(batch1.batch_id, ["PASS"])
    episode1 = host.next_message()
    assert episode1.body["episode_id"] == "1"
    assert episode1.body["logical_job_id"] == "perfectdou_selfplay"
    assert episode1.body["scope_label"] == "DouDiZhu_perfectdou_selfplay [Landlord=perfectdou, Peasants=perfectdou]"
    progress1 = host.next_message()
    assert isinstance(progress1, ProgressUpdate)
    assert progress1.body["progress_schema"] == "core.v1"
    assert progress1.body["completed"] == 1
    assert progress1.body["unit"] == "games"
    assert progress1.body["scope_label"] == "DouDiZhu_perfectdou_selfplay [Landlord=perfectdou, Peasants=perfectdou]"

    batch2 = host.next_message()
    assert isinstance(batch2, DecisionBatch)
    host.submit_actions(batch2.batch_id, ["PASS"])
    batch3 = host.next_message()
    assert isinstance(batch3, DecisionBatch)

    host.submit_actions(batch3.batch_id, ["PASS"])
    episode2 = host.next_message()
    assert episode2.body["episode_id"] == "2"
    assert episode2.body["logical_job_id"] == "perfectdou_selfplay"
    progress2 = host.next_message()
    assert isinstance(progress2, ProgressUpdate)
    assert progress2.body["completed"] == 2
    assert isinstance(host.next_message(), Finished)


def test_two_hosts_share_allocator_without_duplicate_game_ids(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evaluations.doudizhu.host.DoudizhuGame", _FakeGame)

    card_play_data = [
        {"steps": 1, "score": 1.0},
        {"steps": 1, "score": 2.0},
        {"steps": 1, "score": 3.0},
    ]
    agent = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    plan = SeatingPlan(
        matchup_id = "perfectdou_selfplay",
        matchup_name = "DouDiZhu_perfectdou_selfplay",
        mode = "selfplay",
        role_pass = "selfplay",
        left_agent = agent,
        right_agent = agent,
        landlord_agent = agent,
        landlord_down_agent = agent,
        landlord_up_agent = agent,
    )
    allocator = SharedUnitAllocator(total_units = 3)
    host1 = _make_host(
        job_id = "perfectdou_selfplay_worker001of002",
        logical_job_id = "perfectdou_selfplay",
        plan = plan,
        card_play_data = card_play_data,
        num_games = 3,
        max_envs = 1,
        progress_interval = 1,
        work_allocator = allocator,
    )
    host2 = _make_host(
        job_id = "perfectdou_selfplay_worker002of002",
        logical_job_id = "perfectdou_selfplay",
        plan = plan,
        card_play_data = card_play_data,
        num_games = 3,
        max_envs = 2,
        progress_interval = 1,
        work_allocator = allocator,
    )

    host1.open()
    opened1 = host1.next_message()
    assert opened1.body["stage"] == "host_opened"
    assert opened1.body["logical_job_id"] == "perfectdou_selfplay"
    host2.open()
    opened = host2.next_message()
    assert opened.body["stage"] == "host_opened"
    assert opened.body["logical_job_id"] == "perfectdou_selfplay"

    batch1 = host1.next_message()
    assert isinstance(batch1, DecisionBatch)
    assert [request.episode_id for request in batch1.requests] == ["1"]
    batch2 = host2.next_message()
    assert isinstance(batch2, DecisionBatch)
    assert {request.episode_id for request in batch2.requests} == {"2", "3"}
    host1.submit_actions(batch1.batch_id, ["PASS"])
    host2.submit_actions(batch2.batch_id, ["PASS", "PASS"])

    episode1 = host1.next_message()
    assert episode1.body["episode_id"] == "1"
    assert episode1.body["logical_job_id"] == "perfectdou_selfplay"
    progress1 = host1.next_message()
    assert isinstance(progress1, ProgressUpdate)
    assert progress1.body["completed"] == 1
    assert isinstance(host1.next_message(), Finished)
    episode_payloads = (host2.next_message().body, host2.next_message().body)
    episode_ids = {episode_payloads[0]["episode_id"], episode_payloads[1]["episode_id"]}
    assert episode_ids == {"2", "3"}
    assert all(payload["logical_job_id"] == "perfectdou_selfplay" for payload in episode_payloads)
    progress2 = host2.next_message()
    assert isinstance(progress2, ProgressUpdate)
    assert progress2.body["progress_schema"] == "core.v1"
    assert progress2.body["completed"] == 2
    assert progress2.body["logical_job_id"] == "perfectdou_selfplay"
    assert isinstance(host2.next_message(), Finished)


def test_host_emits_progress_when_batch_crosses_interval(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evaluations.doudizhu.host.DoudizhuGame", _FakeGame)

    card_play_data = [
        {"steps": 1, "score": 2.0},
        {"steps": 2, "score": -2.0},
        {"steps": 1, "score": 4.0},
    ]
    agent = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    plan = SeatingPlan(
        matchup_id = "perfectdou_selfplay",
        matchup_name = "DouDiZhu_perfectdou_selfplay",
        mode = "selfplay",
        role_pass = "selfplay",
        left_agent = agent,
        right_agent = agent,
        landlord_agent = agent,
        landlord_down_agent = agent,
        landlord_up_agent = agent,
    )
    host = _make_host(
        job_id = "perfectdou_selfplay",
        logical_job_id = "perfectdou_selfplay",
        plan = plan,
        card_play_data = card_play_data,
        num_games = 3,
        max_envs = 2,
        progress_interval = 2,
        work_allocator = SharedUnitAllocator(total_units = 3),
    )

    host.open()
    assert host.next_message().body["stage"] == "host_opened"

    batch1 = host.next_message()
    assert isinstance(batch1, DecisionBatch)
    host.submit_actions(batch1.batch_id, ["PASS", "PASS"])
    assert host.next_message().body["episode_id"] == "1"

    batch2 = host.next_message()
    assert isinstance(batch2, DecisionBatch)
    host.submit_actions(batch2.batch_id, ["PASS", "PASS"])
    episode_ids = {host.next_message().body["episode_id"], host.next_message().body["episode_id"]}
    assert episode_ids == {"2", "3"}
    progress = host.next_message()
    assert isinstance(progress, ProgressUpdate)
    assert progress.body["completed"] == 3
    assert isinstance(host.next_message(), Finished)


def test_host_zero_progress_interval_emits_final_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evaluations.doudizhu.host.DoudizhuGame", _FakeGame)

    card_play_data = [
        {"steps": 1, "score": 2.0},
        {"steps": 1, "score": -2.0},
    ]
    agent = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    plan = SeatingPlan(
        matchup_id = "perfectdou_selfplay",
        matchup_name = "DouDiZhu_perfectdou_selfplay",
        mode = "selfplay",
        role_pass = "selfplay",
        left_agent = agent,
        right_agent = agent,
        landlord_agent = agent,
        landlord_down_agent = agent,
        landlord_up_agent = agent,
    )
    host = _make_host(
        job_id = "perfectdou_selfplay",
        logical_job_id = "perfectdou_selfplay",
        plan = plan,
        card_play_data = card_play_data,
        num_games = 2,
        max_envs = 1,
        progress_interval = 0,
        work_allocator = SharedUnitAllocator(total_units = 2),
    )

    host.open()
    assert host.next_message().body["stage"] == "host_opened"

    batch1 = host.next_message()
    assert isinstance(batch1, DecisionBatch)
    host.submit_actions(batch1.batch_id, ["PASS"])
    assert host.next_message().body["episode_id"] == "1"

    batch2 = host.next_message()
    assert isinstance(batch2, DecisionBatch)
    host.submit_actions(batch2.batch_id, ["PASS"])
    assert host.next_message().body["episode_id"] == "2"

    progress = host.next_message()
    assert isinstance(progress, ProgressUpdate)
    assert progress.body["completed"] == 2
    assert isinstance(host.next_message(), Finished)


def test_host_negative_progress_interval_disables_final_progress(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evaluations.doudizhu.host.DoudizhuGame", _FakeGame)

    card_play_data = [{"steps": 1, "score": 2.0}]
    agent = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    plan = SeatingPlan(
        matchup_id = "perfectdou_selfplay",
        matchup_name = "DouDiZhu_perfectdou_selfplay",
        mode = "selfplay",
        role_pass = "selfplay",
        left_agent = agent,
        right_agent = agent,
        landlord_agent = agent,
        landlord_down_agent = agent,
        landlord_up_agent = agent,
    )
    host = _make_host(
        job_id = "perfectdou_selfplay",
        logical_job_id = "perfectdou_selfplay",
        plan = plan,
        card_play_data = card_play_data,
        num_games = 1,
        max_envs = 1,
        progress_interval = -1,
        work_allocator = SharedUnitAllocator(total_units = 1),
    )

    host.open()
    assert host.next_message().body["stage"] == "host_opened"
    batch = host.next_message()
    assert isinstance(batch, DecisionBatch)
    host.submit_actions(batch.batch_id, ["PASS"])
    assert host.next_message().body["episode_id"] == "1"
    assert isinstance(host.next_message(), Finished)
