from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluations.core.progress import build_normalized_progress
from evaluations.core import SharedUnitAllocator, WorkerAssignment
from evaluations.core.types import QueueEvent, WorkerSlot
from evaluations.doudizhu.runtime.config import DoudizhuHostConfig
from evaluations.doudizhu.specs import (
    AgentSpec,
    DoudizhuRunConfig,
    DoudizhuWandbConfig,
    SeatingPlan,
    WorkerLayout,
)
import evaluations.doudizhu.summary as summary_module
from evaluations.doudizhu.summary import DoudizhuAggregator, DoudizhuSummaryWriter


SELFPLAY_MATCHUP_NAME = "DouDiZhu_perfectdou_selfplay"
HEAD_TO_HEAD_MATCHUP_NAME = "DouDiZhu_perfectdou_vs_douzero"


class _Job:
    def __init__(self, job_id: str, artifacts_dir: Path, *, logical_job_id: str | None = None) -> None:
        self.job_id = job_id
        self.artifacts_dir = artifacts_dir
        self.logical_job_id = job_id if logical_job_id is None else logical_job_id


class _FakeWandbRun:
    def __init__(self, *, name: str, config: dict[str, object]) -> None:
        self.name = name
        self.config = dict(config)
        self.logged: list[tuple[dict[str, object], int | None]] = []
        self.summary: dict[str, object] = {}
        self.tables: list[tuple[str, tuple[str, ...], tuple[tuple[object, ...], ...]]] = []
        self.saved: list[str] = []
        self.defined_metrics: list[tuple[str, str]] = []
        self.closed = 0

    def define_metric(self, glob_pattern: str, *, step_metric: str) -> None:
        self.defined_metrics.append((str(glob_pattern), str(step_metric)))

    def log(self, payload: dict[str, object], *, step: int | None = None) -> None:
        self.logged.append((dict(payload), step))

    def update_summary(self, payload: dict[str, object]) -> None:
        self.summary.update(dict(payload))

    def log_table(self, key: str, *, columns, rows) -> None:  # noqa: ANN001
        self.tables.append((str(key), tuple(columns), tuple(tuple(row) for row in rows)))

    def save_file(self, path: Path) -> None:
        self.saved.append(str(Path(path).resolve()))

    def close(self) -> None:
        self.closed += 1


def _run_config(
    tmp_path: Path,
    *,
    input_agents: tuple[str, ...],
    unique_agents: tuple[AgentSpec, ...],
    wandb_enabled: bool = False,
    wandb_run_name: str | None = None,
) -> DoudizhuRunConfig:
    return DoudizhuRunConfig(
        input_agents = input_agents,
        unique_agents = unique_agents,
        eval_data = tmp_path / "eval.pkl",
        num_games = 2,
        batch_size = 4,
        host_cfg = DoudizhuHostConfig(
            num_games = 2,
            max_envs = 4,
            progress_interval = 1,
            ci_multiplier = 1.0,
            perfect_info = False,
        ),
        worker_layout = WorkerLayout(
            workers = 1,
            slot_tokens = (None,),
            workers_per_gpu = None,
            usable_gpu_tokens = (),
        ),
        wandb = DoudizhuWandbConfig(
            enabled = bool(wandb_enabled),
            project = "marvel",
            entity = None,
            run_name = wandb_run_name,
        ),
    )


def _make_assignment(
    job_id: str,
    artifacts_dir: Path,
    *,
    logical_job_id: str | None = None,
    worker_index: int = 0,
    worker_count: int = 1,
    total_units: int = 2,
) -> WorkerAssignment:
    job = _Job(job_id, artifacts_dir, logical_job_id = logical_job_id)
    return WorkerAssignment(
        assignment_id = job_id,
        logical_job = job,
        slot = WorkerSlot(worker_id = worker_index, gpu_token = None),
        worker_index = worker_index,
        worker_count = worker_count,
        work_allocator = SharedUnitAllocator(total_units = total_units),
    )


def test_doudizhu_aggregator_selfplay_result_contract(tmp_path: Path) -> None:
    agent = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    plan = SeatingPlan(
        matchup_id = "perfectdou_selfplay",
        matchup_name = SELFPLAY_MATCHUP_NAME,
        mode = "selfplay",
        role_pass = "selfplay",
        left_agent = agent,
        right_agent = agent,
        landlord_agent = agent,
        landlord_down_agent = agent,
        landlord_up_agent = agent,
    )
    aggregator = DoudizhuAggregator(
        assignment = _make_assignment("perfectdou_selfplay", tmp_path / "jobs" / "perfectdou_selfplay"),
        run_config = _run_config(
            tmp_path,
            input_agents = ("perfectdou",),
            unique_agents = (agent,),
        ),
        plan = plan,
    )
    aggregator.record_status({"hands_per_s": 123.0})
    aggregator.record_episode({"episode_id": "1", "landlord_win": 1, "landlord_score": 2.0})
    aggregator.record_episode({"episode_id": "2", "landlord_win": 0, "landlord_score": -2.0})

    result = aggregator.finalize()
    assert result.ok is True
    assert result.payload["mode"] == "selfplay"
    assert result.payload["role_key"] == "landlord_perfectdou__peasants_perfectdou"
    assert result.payload["role_label"] == "Landlord=perfectdou, Peasants=perfectdou"
    assert result.payload["left_view_role"] == "selfplay"
    assert result.payload["metrics"]["landlord_win_rate"]["mean"] == 0.5
    assert result.payload["metrics"]["peasants_team_win_rate"]["mean"] == 0.5
    assert result.payload["metrics"]["peasants_team_avg_gain"]["mean"] == 0.0


def test_doudizhu_summary_writer_aggregates_logical_progress_across_workers(tmp_path: Path, monkeypatch) -> None:
    agent_left = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    agent_right = AgentSpec(kind = "douzero", canonical_spec = "douzero")
    plan = SeatingPlan(
        matchup_id = "perfectdou_vs_douzero",
        matchup_name = HEAD_TO_HEAD_MATCHUP_NAME,
        mode = "head_to_head",
        role_pass = "as_landlord",
        left_agent = agent_left,
        right_agent = agent_right,
        landlord_agent = agent_left,
        landlord_down_agent = agent_right,
        landlord_up_agent = agent_right,
    )
    writer = DoudizhuSummaryWriter(
        logdir = tmp_path,
        run_config = _run_config(
            tmp_path,
            input_agents = ("perfectdou", "douzero"),
            unique_agents = (agent_left, agent_right),
        ),
        planned_matchups = (plan,),
    )
    scope_label = "DouDiZhu_perfectdou_vs_douzero [Landlord=perfectdou, Peasants=douzero]"
    logical_job_id = "perfectdou_vs_douzero:landlord_perfectdou__peasants_douzero"
    perf_counter_values = iter([10.0, 12.0, 14.0])
    monkeypatch.setattr(summary_module.time, "perf_counter", lambda: next(perf_counter_values))

    status_event = QueueEvent(
        kind = "status",
        job_id = "worker-1",
        body = {
            "logical_job_id": logical_job_id,
            "scope_label": scope_label,
            "message": "Opened Dou Dizhu role pass with Landlord=perfectdou, Peasants=douzero.",
        },
    )
    writer.record_event(status_event)
    assert writer.format_event(status_event) == f"[{scope_label}] Opened Dou Dizhu role pass with Landlord=perfectdou, Peasants=douzero."

    episode_event1 = QueueEvent(
        kind = "episode",
        job_id = "worker-1",
        body = {
            "logical_job_id": logical_job_id,
            "matchup_id": "perfectdou_vs_douzero",
            "role_key": "landlord_perfectdou__peasants_douzero",
            "landlord_win": 1,
            "landlord_score": 1.0,
        },
    )
    writer.record_event(episode_event1)
    assert writer.format_event(episode_event1) is None
    episode_event2 = QueueEvent(
        kind = "episode",
        job_id = "worker-2",
        body = {
            "logical_job_id": logical_job_id,
            "matchup_id": "perfectdou_vs_douzero",
            "role_key": "landlord_perfectdou__peasants_douzero",
            "landlord_win": 0,
            "landlord_score": -1.0,
        },
    )
    writer.record_event(episode_event2)
    assert writer.format_event(episode_event2) is None

    progress_event1 = QueueEvent(
        kind = "progress",
        job_id = "worker-1",
        body = build_normalized_progress(
            scope_kind = "physical_job",
            scope_label = scope_label,
            completed = 1,
            total = 10,
            unit = "games",
            throughput_label = "Hands/s",
            throughput_value = 30.0,
            logical_job_id = logical_job_id,
            job_id = "worker-1",
            extra = {
                "matchup_id": "perfectdou_vs_douzero",
                "role_key": "landlord_perfectdou__peasants_douzero",
            },
        ),
    )
    writer.record_event(progress_event1)
    rendered1 = writer.format_event(progress_event1)
    assert rendered1 == (
        "DouDiZhu_perfectdou_vs_douzero [Landlord=perfectdou, Peasants=douzero] | 2 / 10 games | "
        "Landlord Win Rate: 0.500 ± 0.354 | Landlord Avg Gain: 0.000 ± 0.707 | Hands/s: 1.00"
    )

    progress_event2 = QueueEvent(
        kind = "progress",
        job_id = "worker-2",
        body = build_normalized_progress(
            scope_kind = "physical_job",
            scope_label = scope_label,
            completed = 1,
            total = 10,
            unit = "games",
            throughput_label = "Hands/s",
            throughput_value = 20.0,
            logical_job_id = logical_job_id,
            job_id = "worker-2",
            extra = {
                "matchup_id": "perfectdou_vs_douzero",
                "role_key": "landlord_perfectdou__peasants_douzero",
            },
        ),
    )
    writer.record_event(progress_event2)
    rendered2 = writer.format_event(progress_event2)
    assert rendered2 == (
        "DouDiZhu_perfectdou_vs_douzero [Landlord=perfectdou, Peasants=douzero] | 2 / 10 games | "
        "Landlord Win Rate: 0.500 ± 0.354 | Landlord Avg Gain: 0.000 ± 0.707 | Hands/s: 0.50"
    )


def test_doudizhu_summary_writer_pairs_head_to_head_roles(tmp_path: Path) -> None:
    left = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    right = AgentSpec(kind = "douzero", canonical_spec = "douzero")
    writer = DoudizhuSummaryWriter(
        logdir = tmp_path,
        run_config = _run_config(
            tmp_path,
            input_agents = ("perfectdou", "douzero"),
            unique_agents = (left, right),
        ),
    )
    as_landlord = {
        "schema_version": 1,
        "scenario": "doudizhu",
        "runner": "evaluations_doudizhu",
        "job_kind": "seating",
        "matchup_id": "perfectdou_vs_douzero",
        "matchup_name": HEAD_TO_HEAD_MATCHUP_NAME,
        "mode": "head_to_head",
        "role_key": "landlord_perfectdou__peasants_douzero",
        "role_label": "Landlord=perfectdou, Peasants=douzero",
        "left_agent": "perfectdou",
        "right_agent": "douzero",
        "left_view_role": "as_landlord",
        "ok": True,
        "agents": {
            "landlord": "perfectdou",
            "landlord_down": "douzero",
            "landlord_up": "douzero",
        },
        "warnings": [],
        "hands_per_s": 10.0,
        "metrics": {
            "landlord_win_rate": {"mean": 0.5, "ci": 0.1, "n": 2, "std": 0.5, "sem": 0.35, "unit": "probability"},
            "landlord_avg_gain": {"mean": 1.0, "ci": 0.2, "n": 2, "std": 1.0, "sem": 0.7, "unit": "landlord_score"},
            "peasants_team_win_rate": {"mean": 0.5, "ci": 0.1, "n": 2, "std": 0.5, "sem": 0.35, "unit": "probability"},
            "peasants_team_avg_gain": {"mean": -1.0, "ci": 0.2, "n": 2, "std": 1.0, "sem": 0.7, "unit": "landlord_score"},
        },
        "landlord_win_by_game": [1, 0],
        "landlord_score_by_game": [2.0, 0.0],
    }
    as_peasants = {
        **as_landlord,
        "role_key": "landlord_douzero__peasants_perfectdou",
        "role_label": "Landlord=douzero, Peasants=perfectdou",
        "left_view_role": "as_peasants",
        "agents": {
            "landlord": "douzero",
            "landlord_down": "perfectdou",
            "landlord_up": "perfectdou",
        },
        "landlord_win_by_game": [0, 1],
        "landlord_score_by_game": [-1.0, 1.0],
        "metrics": {
            **as_landlord["metrics"],
            "landlord_avg_gain": {"mean": 0.0, "ci": 0.2, "n": 2, "std": 1.0, "sem": 0.7, "unit": "landlord_score"},
        },
    }

    path = writer.write_run(
        [
            type("Result", (), {"job_id": "as_landlord", "payload": as_landlord, "ok": True, "artifacts": {}})(),
            type("Result", (), {"job_id": "as_peasants", "payload": as_peasants, "ok": True, "artifacts": {}})(),
        ]
    )
    payload = json.loads(path.read_text())
    assert payload["mode"] == "head_to_head"
    assert len(payload["matchups"]) == 1
    matchup = payload["matchups"][0]
    assert matchup["role_results"][0]["left_view_role"] == "as_landlord"
    assert matchup["role_results"][1]["left_view_role"] == "as_peasants"
    assert matchup["left_view_metrics"]["role_avg_win_rate"]["mean"] == 0.5
    assert matchup["left_view_metrics"]["as_landlord_avg_gain"]["mean"] == 1.0
    assert matchup["left_view_metrics"]["as_peasants_avg_gain"]["mean"] == 0.0


def test_doudizhu_summary_writer_merges_partial_results_by_logical_job(tmp_path: Path) -> None:
    left = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    right = AgentSpec(kind = "douzero", canonical_spec = "douzero")
    writer = DoudizhuSummaryWriter(
        logdir = tmp_path,
        run_config = _run_config(
            tmp_path,
            input_agents = ("perfectdou", "douzero"),
            unique_agents = (left, right),
        ),
    )

    def _partial_payload(
        *,
        logical_job_id: str,
        worker_index: int,
        role_key: str,
        left_view_role: str,
        game_ids,
        wins,
        scores,
        agents,
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "scenario": "doudizhu",
            "runner": "evaluations_doudizhu",
            "job_kind": "seating_partial",
            "logical_job_id": logical_job_id,
            "worker_index": worker_index,
            "worker_count": 2,
            "matchup_id": "perfectdou_vs_douzero",
            "matchup_name": HEAD_TO_HEAD_MATCHUP_NAME,
            "mode": "head_to_head",
            "role_key": role_key,
            "role_label": "Landlord=perfectdou, Peasants=douzero"
            if agents["landlord"] == "perfectdou"
            else "Landlord=douzero, Peasants=perfectdou",
            "left_agent": "perfectdou",
            "right_agent": "douzero",
            "left_view_role": left_view_role,
            "num_games": len(game_ids),
            "logical_num_games": 2,
            "agents": agents,
            "warnings": [],
            "hands_per_s": 10.0,
            "game_ids": list(game_ids),
            "metrics": {
                "landlord_win_rate": {"mean": float(sum(wins)) / len(wins), "ci": 0.0, "n": len(wins), "std": 0.0, "sem": 0.0, "unit": "probability"},
                "landlord_avg_gain": {"mean": float(sum(scores)) / len(scores), "ci": 0.0, "n": len(scores), "std": 0.0, "sem": 0.0, "unit": "landlord_score"},
                "peasants_team_win_rate": {"mean": 0.0, "ci": 0.0, "n": len(wins), "std": 0.0, "sem": 0.0, "unit": "probability"},
                "peasants_team_avg_gain": {"mean": 0.0, "ci": 0.0, "n": len(scores), "std": 0.0, "sem": 0.0, "unit": "landlord_score"},
            },
            "landlord_win_by_game": list(wins),
            "landlord_score_by_game": list(scores),
        }

    as_landlord_agents = {
        "landlord": "perfectdou",
        "landlord_down": "douzero",
        "landlord_up": "douzero",
    }
    as_peasants_agents = {
        "landlord": "douzero",
        "landlord_down": "perfectdou",
        "landlord_up": "perfectdou",
    }
    path = writer.write_run(
        [
            type("Result", (), {"job_id": "as_landlord-w2", "payload": _partial_payload(logical_job_id = "as_landlord", worker_index = 1, role_key = "landlord_perfectdou__peasants_douzero", left_view_role = "as_landlord", game_ids = [1], wins = [0], scores = [0.0], agents = as_landlord_agents), "ok": True, "artifacts": {}})(),
            type("Result", (), {"job_id": "as_landlord-w1", "payload": _partial_payload(logical_job_id = "as_landlord", worker_index = 0, role_key = "landlord_perfectdou__peasants_douzero", left_view_role = "as_landlord", game_ids = [2], wins = [1], scores = [2.0], agents = as_landlord_agents), "ok": True, "artifacts": {}})(),
            type("Result", (), {"job_id": "as_peasants-w2", "payload": _partial_payload(logical_job_id = "as_peasants", worker_index = 1, role_key = "landlord_douzero__peasants_perfectdou", left_view_role = "as_peasants", game_ids = [1], wins = [1], scores = [1.0], agents = as_peasants_agents), "ok": True, "artifacts": {}})(),
            type("Result", (), {"job_id": "as_peasants-w1", "payload": _partial_payload(logical_job_id = "as_peasants", worker_index = 0, role_key = "landlord_douzero__peasants_perfectdou", left_view_role = "as_peasants", game_ids = [2], wins = [0], scores = [-1.0], agents = as_peasants_agents), "ok": True, "artifacts": {}})(),
        ]
    )

    payload = json.loads(path.read_text())
    matchup = payload["matchups"][0]
    assert len(matchup["role_results"]) == 2
    assert matchup["role_results"][0]["metrics"]["landlord_win_rate"]["mean"] == 0.5
    assert matchup["role_results"][1]["metrics"]["landlord_win_rate"]["mean"] == 0.5
    assert "left_view_metrics" in matchup


def test_doudizhu_summary_writer_marks_failed_matchup_from_result_flag(tmp_path: Path) -> None:
    left = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    right = AgentSpec(kind = "douzero", canonical_spec = "douzero")
    writer = DoudizhuSummaryWriter(
        logdir = tmp_path,
        run_config = _run_config(
            tmp_path,
            input_agents = ("perfectdou", "douzero"),
            unique_agents = (left, right),
        ),
    )
    as_landlord = {
        "schema_version": 1,
        "scenario": "doudizhu",
        "runner": "evaluations_doudizhu",
        "job_kind": "seating",
        "matchup_id": "perfectdou_vs_douzero",
        "matchup_name": HEAD_TO_HEAD_MATCHUP_NAME,
        "mode": "head_to_head",
        "role_key": "landlord_perfectdou__peasants_douzero",
        "role_label": "Landlord=perfectdou, Peasants=douzero",
        "left_agent": "perfectdou",
        "right_agent": "douzero",
        "left_view_role": "as_landlord",
        "agents": {
            "landlord": "perfectdou",
            "landlord_down": "douzero",
            "landlord_up": "douzero",
        },
        "warnings": [],
        "hands_per_s": 10.0,
        "metrics": {
            "landlord_win_rate": {"mean": 0.5, "ci": 0.1, "n": 2, "std": 0.5, "sem": 0.35, "unit": "probability"},
            "landlord_avg_gain": {"mean": 1.0, "ci": 0.2, "n": 2, "std": 1.0, "sem": 0.7, "unit": "landlord_score"},
            "peasants_team_win_rate": {"mean": 0.5, "ci": 0.1, "n": 2, "std": 0.5, "sem": 0.35, "unit": "probability"},
            "peasants_team_avg_gain": {"mean": -1.0, "ci": 0.2, "n": 2, "std": 1.0, "sem": 0.7, "unit": "landlord_score"},
        },
        "landlord_win_by_game": [1, 0],
        "landlord_score_by_game": [2.0, 0.0],
    }
    failed_role = {
        "job_id": "as_peasants",
        "scenario_name": "doudizhu",
        "mode": "head_to_head",
        "matchup_id": "perfectdou_vs_douzero",
        "matchup_name": HEAD_TO_HEAD_MATCHUP_NAME,
        "role_key": "landlord_douzero__peasants_perfectdou",
        "role_label": "Landlord=douzero, Peasants=perfectdou",
        "left_agent": "perfectdou",
        "right_agent": "douzero",
        "left_view_role": "as_peasants",
        "error": "controller startup failed",
    }

    path = writer.write_run(
        [
            type("Result", (), {"job_id": "as_landlord", "payload": as_landlord, "ok": True, "artifacts": {}})(),
            type("Result", (), {"job_id": "as_peasants", "payload": failed_role, "ok": False, "artifacts": {}})(),
        ]
    )
    payload = json.loads(path.read_text())
    matchup = payload["matchups"][0]
    assert matchup["ok"] is False
    assert "left_view_metrics" not in matchup
    assert matchup["role_results"][1]["error"] == "controller startup failed"

    report = writer.render_report(
        [
            type("Result", (), {"job_id": "as_landlord", "payload": as_landlord, "ok": True, "artifacts": {}})(),
            type("Result", (), {"job_id": "as_peasants", "payload": failed_role, "ok": False, "artifacts": {}})(),
        ]
    )
    assert report is not None
    assert "Landlord=douzero, Peasants=perfectdou failed: controller startup failed" in report


def test_doudizhu_summary_writer_logs_live_and_final_wandb_metrics(monkeypatch, tmp_path: Path) -> None:
    left = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    right = AgentSpec(kind = "douzero", canonical_spec = "douzero")
    created_runs: dict[str, _FakeWandbRun] = {}

    def _open_wandb_run(*, project: str, entity: str | None, name: str, config: dict[str, object]) -> _FakeWandbRun:
        assert project == "marvel"
        assert entity is None
        run = _FakeWandbRun(name = name, config = config)
        created_runs[name] = run
        return run

    monkeypatch.setattr(summary_module.EvaluationWandbRun, "open", staticmethod(_open_wandb_run))

    writer = DoudizhuSummaryWriter(
        logdir = tmp_path,
        run_config = _run_config(
            tmp_path,
            input_agents = ("perfectdou", "douzero"),
            unique_agents = (left, right),
            wandb_enabled = True,
        ),
    )
    writer.record_event(
        QueueEvent(
            kind = "episode",
            job_id = "as_peasants",
            body = {
                "matchup_id": "perfectdou_vs_douzero",
                "matchup_name": HEAD_TO_HEAD_MATCHUP_NAME,
                "mode": "head_to_head",
                "role_key": "landlord_douzero__peasants_perfectdou",
                "role_label": "Landlord=douzero, Peasants=perfectdou",
                "left_agent": "perfectdou",
                "right_agent": "douzero",
                "left_view_role": "as_peasants",
                "episode_id": "1",
                "landlord_win": 1,
                "landlord_score": 1.0,
            },
        )
    )
    writer.record_event(
        QueueEvent(
            kind = "episode",
            job_id = "as_landlord",
            body = {
                "matchup_id": "perfectdou_vs_douzero",
                "matchup_name": HEAD_TO_HEAD_MATCHUP_NAME,
                "mode": "head_to_head",
                "role_key": "landlord_perfectdou__peasants_douzero",
                "role_label": "Landlord=perfectdou, Peasants=douzero",
                "left_agent": "perfectdou",
                "right_agent": "douzero",
                "left_view_role": "as_landlord",
                "episode_id": "2",
                "landlord_win": 1,
                "landlord_score": 2.0,
            },
        )
    )
    writer.record_event(
        QueueEvent(
            kind = "episode",
            job_id = "as_peasants",
            body = {
                "matchup_id": "perfectdou_vs_douzero",
                "matchup_name": HEAD_TO_HEAD_MATCHUP_NAME,
                "mode": "head_to_head",
                "role_key": "landlord_douzero__peasants_perfectdou",
                "role_label": "Landlord=douzero, Peasants=perfectdou",
                "left_agent": "perfectdou",
                "right_agent": "douzero",
                "left_view_role": "as_peasants",
                "episode_id": "2",
                "landlord_win": 0,
                "landlord_score": -1.0,
            },
        )
    )
    run = created_runs[HEAD_TO_HEAD_MATCHUP_NAME]
    role_avg_logs = [payload for payload, _ in run.logged if "role_avg/games" in payload]
    assert len(role_avg_logs) == 1
    assert role_avg_logs[0] == {
        "role_avg/games": 1,
        "role_avg/win_rate_mean": 1.0,
        "role_avg/win_rate_ci": 0.0,
        "role_avg/avg_gain_mean": 1.5,
        "role_avg/avg_gain_ci": 0.0,
    }
    assert any("as_landlord/games" in payload for payload, _ in run.logged)
    assert any("as_peasants/games" in payload for payload, _ in run.logged)
    assert run.logged[-1][0] == {
        "role_avg/games": 1,
        "role_avg/win_rate_mean": 1.0,
        "role_avg/win_rate_ci": 0.0,
        "role_avg/avg_gain_mean": 1.5,
        "role_avg/avg_gain_ci": 0.0,
    }
    (tmp_path / "events.jsonl").write_text("", encoding = "utf-8")

    as_landlord = {
        "schema_version": 1,
        "scenario": "doudizhu",
        "runner": "evaluations_doudizhu",
        "job_kind": "seating",
        "matchup_id": "perfectdou_vs_douzero",
        "matchup_name": HEAD_TO_HEAD_MATCHUP_NAME,
        "mode": "head_to_head",
        "role_key": "landlord_perfectdou__peasants_douzero",
        "role_label": "Landlord=perfectdou, Peasants=douzero",
        "left_agent": "perfectdou",
        "right_agent": "douzero",
        "left_view_role": "as_landlord",
        "agents": {
            "landlord": "perfectdou",
            "landlord_down": "douzero",
            "landlord_up": "douzero",
        },
        "warnings": [],
        "hands_per_s": 10.0,
        "metrics": {
            "landlord_win_rate": {"mean": 0.5, "ci": 0.5, "n": 2, "std": 0.5, "sem": 0.5, "unit": "probability"},
            "landlord_avg_gain": {"mean": 1.0, "ci": 1.0, "n": 2, "std": 1.0, "sem": 1.0, "unit": "landlord_score"},
            "peasants_team_win_rate": {"mean": 0.5, "ci": 0.5, "n": 2, "std": 0.5, "sem": 0.5, "unit": "probability"},
            "peasants_team_avg_gain": {"mean": -1.0, "ci": 1.0, "n": 2, "std": 1.0, "sem": 1.0, "unit": "landlord_score"},
        },
        "landlord_win_by_game": [1, 0],
        "landlord_score_by_game": [2.0, 0.0],
    }
    as_peasants = {
        **as_landlord,
        "role_key": "landlord_douzero__peasants_perfectdou",
        "role_label": "Landlord=douzero, Peasants=perfectdou",
        "left_view_role": "as_peasants",
        "agents": {
            "landlord": "douzero",
            "landlord_down": "perfectdou",
            "landlord_up": "perfectdou",
        },
        "landlord_win_by_game": [0, 1],
        "landlord_score_by_game": [-1.0, 1.0],
    }

    summary_path = writer.write_run(
        [
            type("Result", (), {"job_id": "as_landlord", "payload": as_landlord, "ok": True, "artifacts": {}})(),
            type("Result", (), {"job_id": "as_peasants", "payload": as_peasants, "ok": True, "artifacts": {}})(),
        ]
    )
    assert run.summary["role_avg/win_rate_mean"] == 0.5
    assert run.summary["role_avg/win_rate_ci"] == pytest.approx(0.3535533905932738)
    assert run.summary["as_landlord/avg_gain_mean"] == 1.0
    assert run.summary["as_peasants/avg_gain_mean"] == 0.0
    assert any(table[0] == "final_metrics" for table in run.tables)
    assert str(summary_path.resolve()) in run.saved


def test_doudizhu_summary_writer_logs_live_wandb_metrics_despite_head_to_head_gap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    left = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    right = AgentSpec(kind = "douzero", canonical_spec = "douzero")
    created_runs: dict[str, _FakeWandbRun] = {}

    def _open_wandb_run(*, project: str, entity: str | None, name: str, config: dict[str, object]) -> _FakeWandbRun:
        assert project == "marvel"
        assert entity is None
        run = _FakeWandbRun(name = name, config = config)
        created_runs[name] = run
        return run

    monkeypatch.setattr(summary_module.EvaluationWandbRun, "open", staticmethod(_open_wandb_run))

    writer = DoudizhuSummaryWriter(
        logdir = tmp_path,
        run_config = _run_config(
            tmp_path,
            input_agents = ("perfectdou", "douzero"),
            unique_agents = (left, right),
            wandb_enabled = True,
        ),
    )
    writer.record_event(
        QueueEvent(
            kind = "episode",
            job_id = "as_landlord",
            body = {
                "matchup_id": "perfectdou_vs_douzero",
                "matchup_name": HEAD_TO_HEAD_MATCHUP_NAME,
                "mode": "head_to_head",
                "role_key": "landlord_perfectdou__peasants_douzero",
                "role_label": "Landlord=perfectdou, Peasants=douzero",
                "left_agent": "perfectdou",
                "right_agent": "douzero",
                "left_view_role": "as_landlord",
                "episode_id": "31",
                "landlord_win": 1,
                "landlord_score": 2.0,
            },
        )
    )
    writer.record_event(
        QueueEvent(
            kind = "episode",
            job_id = "as_peasants",
            body = {
                "matchup_id": "perfectdou_vs_douzero",
                "matchup_name": HEAD_TO_HEAD_MATCHUP_NAME,
                "mode": "head_to_head",
                "role_key": "landlord_douzero__peasants_perfectdou",
                "role_label": "Landlord=douzero, Peasants=perfectdou",
                "left_agent": "perfectdou",
                "right_agent": "douzero",
                "left_view_role": "as_peasants",
                "episode_id": "31",
                "landlord_win": 0,
                "landlord_score": -1.0,
            },
        )
    )

    run = created_runs[HEAD_TO_HEAD_MATCHUP_NAME]
    role_avg_logs = [payload for payload, _ in run.logged if "role_avg/games" in payload]
    assert len(role_avg_logs) == 1
    assert role_avg_logs[0]["role_avg/games"] == 1
    assert role_avg_logs[0]["role_avg/win_rate_mean"] == 1.0
    assert role_avg_logs[0]["role_avg/avg_gain_mean"] == 1.5


def test_doudizhu_summary_writer_logs_live_wandb_metrics_despite_selfplay_gap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    agent = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    created_runs: dict[str, _FakeWandbRun] = {}

    def _open_wandb_run(*, project: str, entity: str | None, name: str, config: dict[str, object]) -> _FakeWandbRun:
        assert project == "marvel"
        assert entity is None
        run = _FakeWandbRun(name = name, config = config)
        created_runs[name] = run
        return run

    monkeypatch.setattr(summary_module.EvaluationWandbRun, "open", staticmethod(_open_wandb_run))

    writer = DoudizhuSummaryWriter(
        logdir = tmp_path,
        run_config = _run_config(
            tmp_path,
            input_agents = ("perfectdou",),
            unique_agents = (agent,),
            wandb_enabled = True,
        ),
    )
    writer.record_event(
        QueueEvent(
            kind = "episode",
            job_id = "selfplay",
            body = {
                "matchup_id": "perfectdou_selfplay",
                "matchup_name": SELFPLAY_MATCHUP_NAME,
                "mode": "selfplay",
                "role_key": "landlord_perfectdou__peasants_perfectdou",
                "role_label": "Landlord=perfectdou, Peasants=perfectdou",
                "left_agent": "perfectdou",
                "right_agent": "perfectdou",
                "left_view_role": "selfplay",
                "episode_id": "31",
                "landlord_win": 1,
                "landlord_score": 2.0,
            },
        )
    )

    run = created_runs[SELFPLAY_MATCHUP_NAME]
    assert len(run.logged) == 1
    assert run.logged[0][1] is None
    assert run.logged[0][0] == {
        "as_landlord/games": 1,
        "as_peasants/games": 1,
        "as_landlord/win_rate_mean": 1.0,
        "as_landlord/win_rate_ci": 0.0,
        "as_landlord/avg_gain_mean": 2.0,
        "as_landlord/avg_gain_ci": 0.0,
        "as_peasants/win_rate_mean": 0.0,
        "as_peasants/win_rate_ci": 0.0,
        "as_peasants/avg_gain_mean": -2.0,
        "as_peasants/avg_gain_ci": 0.0,
    }


def test_doudizhu_summary_writer_eagerly_opens_wandb_runs(monkeypatch, tmp_path: Path) -> None:
    left = AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    right = AgentSpec(kind = "douzero", canonical_spec = "douzero")
    as_landlord = SeatingPlan(
        matchup_id = "perfectdou_vs_douzero",
        matchup_name = HEAD_TO_HEAD_MATCHUP_NAME,
        mode = "head_to_head",
        role_pass = "as_landlord",
        left_agent = left,
        right_agent = right,
        landlord_agent = left,
        landlord_down_agent = right,
        landlord_up_agent = right,
    )
    as_peasants = SeatingPlan(
        matchup_id = "perfectdou_vs_douzero",
        matchup_name = HEAD_TO_HEAD_MATCHUP_NAME,
        mode = "head_to_head",
        role_pass = "as_peasants",
        left_agent = left,
        right_agent = right,
        landlord_agent = right,
        landlord_down_agent = left,
        landlord_up_agent = left,
    )

    created_runs: dict[str, _FakeWandbRun] = {}

    def _open_wandb_run(*, project: str, entity: str | None, name: str, config: dict[str, object]) -> _FakeWandbRun:
        assert project == "marvel"
        assert entity is None
        run = _FakeWandbRun(name = name, config = config)
        created_runs[name] = run
        return run

    monkeypatch.setattr(summary_module.EvaluationWandbRun, "open", staticmethod(_open_wandb_run))

    writer = DoudizhuSummaryWriter(
        logdir = tmp_path,
        run_config = _run_config(
            tmp_path,
            input_agents = ("perfectdou", "douzero"),
            unique_agents = (left, right),
            wandb_enabled = True,
        ),
        planned_matchups = (as_landlord, as_peasants),
    )

    assert list(created_runs) == [HEAD_TO_HEAD_MATCHUP_NAME]
    run = created_runs[HEAD_TO_HEAD_MATCHUP_NAME]
    assert run.config["matchup_id"] == "perfectdou_vs_douzero"
    assert run.config["left_agent"] == "perfectdou"
    assert run.config["right_agent"] == "douzero"

    writer.record_event(
        QueueEvent(
            kind = "episode",
            job_id = "as_landlord",
            body = {
                "matchup_id": "perfectdou_vs_douzero",
                "matchup_name": HEAD_TO_HEAD_MATCHUP_NAME,
                "mode": "head_to_head",
                "role_key": "landlord_perfectdou__peasants_douzero",
                "role_label": "Landlord=perfectdou, Peasants=douzero",
                "left_agent": "perfectdou",
                "right_agent": "douzero",
                "left_view_role": "as_landlord",
                "episode_id": "1",
                "landlord_win": 1,
                "landlord_score": 2.0,
            },
        )
    )

    assert list(created_runs) == [HEAD_TO_HEAD_MATCHUP_NAME]
    assert len(run.logged) == 1
    assert run.logged[0][0]["as_landlord/games"] == 1
