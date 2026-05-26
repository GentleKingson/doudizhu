from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import threading
from types import SimpleNamespace

import pytest

from evaluations.core import EvaluationJob, SharedUnitAllocator, WorkerAssignment, WorkerLayoutPlanner, WorkerSlot
from evaluations.doudizhu import specs
import evaluations.doudizhu.scenario as doudizhu_scenario_module
from evaluations.doudizhu.scenario import DoudizhuRuntimeFactory, DoudizhuScenario
from marvel.runtime.checkpoints import CheckpointStore


class _NoopEvalDataProvider:
    def ensure_eval_data(self, path: Path, *, min_games: int, python_cmd) -> None:  # noqa: ANN001
        del path, min_games, python_cmd
        return None


def _planner(
    *,
    visible_tokens = (),
    usable_tokens = (),
) -> specs.DoudizhuRunPlanner:
    return specs.DoudizhuRunPlanner(
        eval_data_provider = _NoopEvalDataProvider(),
        layout_planner = WorkerLayoutPlanner(
            discover_visible_tokens = lambda: list(visible_tokens),
            probe_token = lambda token: token in set(usable_tokens),
        ),
    )


def test_agent_spec_parser_parses_canonical_baselines() -> None:
    parser = specs.DoudizhuAgentSpecParser()
    assert parser.parse("perfectdou").kind == "perfectdou"
    assert parser.parse("douzero").kind == "douzero"
    assert parser.parse("fpdou").kind == "fpdou"


def test_agent_spec_parser_parses_marvel_direct_and_search_variants() -> None:
    parser = specs.DoudizhuAgentSpecParser()
    direct = parser.parse("marvel=ckpts/policy.pt?action_temp=0&belief_temp=2")
    assert direct.kind == "marvel_direct"
    assert direct.canonical_spec.endswith("?action_temp=0&belief_temp=2")
    assert direct.action_temp == 0.0
    assert direct.belief_temp == 2.0

    search = parser.parse("marvel@yomi=policy.pt@belief.pt?action_temp=1&search_beta=1&search_budget=2048")
    assert search.kind == "marvel_search"
    assert search.canonical_spec.endswith("?action_temp=1&search_beta=1&search_budget=2048")
    assert search.search_routine == "local-subgame"
    assert search.search_beta == 1.0
    assert search.search_budget == 2048
    assert search.search_floor == 16

    local = parser.parse(
        "marvel@yomi=policy.pt@belief.pt?routine=local-subgame&search_budget=64&search_floor=8"
    )
    assert local.canonical_spec.endswith("?routine=local-subgame&search_budget=64&search_floor=8")
    assert local.search_routine == "local-subgame"
    assert local.search_budget == 64
    assert local.search_floor == 8

    low_budget = parser.parse(
        "marvel@yomi=policy.pt@belief.pt?routine=local-subgame&search_budget=8"
    )
    assert low_budget.canonical_spec.endswith("?routine=local-subgame&search_budget=8")
    assert low_budget.search_budget == 8
    assert low_budget.search_floor == 8


def test_agent_spec_parser_rejects_legacy_aliases() -> None:
    parser = specs.DoudizhuAgentSpecParser()
    with pytest.raises(ValueError, match = "Unknown agent spec"):
        _ = parser.parse("baseline.douzero")

    with pytest.raises(ValueError, match = "Unknown agent spec"):
        _ = parser.parse("marvel.ckpt=/tmp/a.pt")


def test_agent_spec_parser_rejects_non_marvel_query() -> None:
    parser = specs.DoudizhuAgentSpecParser()
    with pytest.raises(ValueError, match = "Only MARVEL specs support"):
        _ = parser.parse("perfectdou?bad=1")


def test_agent_spec_parser_rejects_unknown_or_duplicate_override_keys() -> None:
    parser = specs.DoudizhuAgentSpecParser()
    with pytest.raises(ValueError, match = "Unknown MARVEL override key"):
        _ = parser.parse("marvel=wandb:e/p/r?temperature=1")

    with pytest.raises(ValueError, match = "Duplicate override key"):
        _ = parser.parse("marvel=wandb:e/p/r?action_temp=1&action_temp=0")


def test_agent_spec_parser_rejects_invalid_search_budget_rules() -> None:
    parser = specs.DoudizhuAgentSpecParser()
    with pytest.raises(ValueError, match = "Unknown MARVEL override key"):
        _ = parser.parse("marvel=wandb:e/p/r?search_budget=1024")

    with pytest.raises(ValueError, match = "search_budget must be a positive integer"):
        _ = parser.parse("marvel@yomi=wandb:e/p/r1@wandb:e/p/r2?search_budget=0")

    with pytest.raises(ValueError, match = "Unknown MARVEL override key"):
        _ = parser.parse("marvel=wandb:e/p/r?search_beta=1")

    with pytest.raises(ValueError, match = "search_beta must be finite and non-negative"):
        _ = parser.parse("marvel@yomi=wandb:e/p/r1@wandb:e/p/r2?search_beta=-1")

    with pytest.raises(ValueError, match = "search_floor"):
        _ = parser.parse("marvel@yomi=wandb:e/p/r1@wandb:e/p/r2?routine=update-equiv&search_floor=4")

    with pytest.raises(ValueError, match = "search_floor"):
        _ = parser.parse(
            "marvel@yomi=wandb:e/p/r1@wandb:e/p/r2"
            "?routine=local-subgame&search_budget=8&search_floor=16"
        )


def test_agent_command_builder_uses_evaluations_modules(tmp_path: Path) -> None:
    socket_path = tmp_path / "agent.sock"
    builder = specs.DoudizhuAgentCommandBuilder()

    perfectdou_cmd = builder.build(
        specs.AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou"),
        socket_path = socket_path,
        device = "cpu",
    )
    douzero_cmd = builder.build(
        specs.AgentSpec(kind = "douzero", canonical_spec = "douzero"),
        socket_path = socket_path,
        device = "cpu",
    )

    assert perfectdou_cmd[:2] == ["bash", "evaluations/doudizhu/agents/perfectdou.sh"]
    assert douzero_cmd[3] == "evaluations.doudizhu.agents.douzero"


def test_agent_log_name_and_role_label_use_readable_wandb_specs() -> None:
    parser = specs.DoudizhuAgentSpecParser()
    agent = parser.parse(
        "marvel@yomi=wandb:entity/marvel/policy_run@wandb:entity/yomi/belief_run?action_temp=0&search_temp=0&search_beta=1"
    )
    opponent = parser.parse("marvel=wandb:entity/marvel/opp_run?action_temp=0")

    assert agent.log_name() == "marvel@yomi[policy_run+belief_run](action_temp=0,search_temp=0,search_beta=1)"
    assert opponent.log_name() == "marvel[opp_run](action_temp=0)"

    plan = specs.SeatingPlan(
        matchup_id = "matchup",
        matchup_name = "DouDiZhu_matchup",
        mode = "head_to_head",
        role_pass = "as_landlord",
        left_agent = agent,
        right_agent = opponent,
        landlord_agent = agent,
        landlord_down_agent = opponent,
        landlord_up_agent = opponent,
    )
    assert plan.progress_scope_label() == (
        "DouDiZhu_matchup [Landlord=marvel@yomi[policy_run+belief_run](action_temp=0,search_temp=0,search_beta=1), "
        "Peasants=marvel[opp_run](action_temp=0)]"
    )
    assert plan.role_key() == (
        "landlord_marvel_yomi_policy_run_belief_run_action_temp_0_search_temp_0_search_beta_1"
        "__peasants_marvel_opp_run_action_temp_0"
    )
    assert plan.left_view_role() == "as_landlord"


def test_doudizhu_run_planner_reports_seating_workload_from_num_games(tmp_path: Path) -> None:
    planner = _planner()
    args = argparse.Namespace(
        agents = ["perfectdou"],
        num_games = 50,
        workers_per_gpu = 2,
        batch_size = 4,
        progress_interval = 100,
        eval_data = tmp_path / "eval.pkl",
        logdir = tmp_path,
        wandb = True,
        wandb_project = None,
        wandb_entity = None,
        wandb_run_name = None,
    )

    planned = planner.plan(args)

    workload = planner._matchup_planner.describe_job_workload(planned.logical_jobs[0])  # noqa: SLF001
    assert workload.total_units == 50
    assert workload.unit_name == "games"


def test_doudizhu_run_planner_accepts_negative_progress_interval(tmp_path: Path) -> None:
    planner = _planner()
    args = argparse.Namespace(
        agents = ["perfectdou"],
        num_games = 50,
        workers_per_gpu = 2,
        batch_size = 4,
        progress_interval = -1,
        eval_data = tmp_path / "eval.pkl",
        logdir = tmp_path,
        wandb = True,
        wandb_project = None,
        wandb_entity = None,
        wandb_run_name = None,
    )

    run_cfg = planner._build_run_config(args, unique_agents = ())  # noqa: SLF001

    assert run_cfg.host_cfg.progress_interval == -1


def test_doudizhu_run_planner_expands_two_agent_run_to_fill_worker_slots(tmp_path: Path) -> None:
    planner = _planner(
        visible_tokens = ("0", "1", "2", "3"),
        usable_tokens = ("0", "1", "2", "3"),
    )
    args = argparse.Namespace(
        agents = ["douzero", "fpdou"],
        num_games = 4,
        workers_per_gpu = 2,
        batch_size = 1,
        progress_interval = 100,
        eval_data = tmp_path / "eval.pkl",
        logdir = tmp_path,
        wandb = True,
        wandb_project = None,
        wandb_entity = None,
        wandb_run_name = None,
    )

    planned = planner.plan(args)

    assert len(planned.slots) == 8
    assert len(planned.logical_jobs) == 2
    logical_job_ids = {job.logical_job_id for job in planned.logical_jobs}
    assert len(logical_job_ids) == 2
    assert all(job.config["plan"].matchup_name == "DouDiZhu_douzero_vs_fpdou" for job in planned.logical_jobs)
    assert tuple(job.logical_job_id for job in planned.logical_jobs) == tuple(sorted(logical_job_ids))


def test_doudizhu_run_planner_resolves_marvel_checkpoints_during_planning(monkeypatch, tmp_path: Path) -> None:
    planner = _planner()
    policy_checkpoint = str((tmp_path / "resolved" / "policy.pt").resolve())
    belief_checkpoint = str((tmp_path / "resolved" / "belief.pt").resolve())
    Path(policy_checkpoint).parent.mkdir(parents = True, exist_ok = True)
    Path(policy_checkpoint).write_bytes(b"policy")
    Path(belief_checkpoint).write_bytes(b"belief")
    resolve_calls: list[tuple[str, Path]] = []

    def _mock_checkpoint_file_path(locator: str, **kwargs) -> Path:
        resolve_calls.append(str(locator))
        if str(locator) == "wandb:entity/marvel/policy_run":
            return Path(policy_checkpoint)
        if str(locator) == "wandb:entity/yomi/belief_run":
            return Path(belief_checkpoint)
        raise AssertionError(f"Unexpected locator: {locator}")

    monkeypatch.setattr(CheckpointStore, "resolve_checkpoint_file_path", staticmethod(_mock_checkpoint_file_path))
    args = argparse.Namespace(
        agents = [
            "marvel=wandb:entity/marvel/policy_run?action_temp=1",
            "marvel@yomi=wandb:entity/marvel/policy_run@wandb:entity/yomi/belief_run?action_temp=1&search_beta=2&search_budget=32",
        ],
        num_games = 12,
        workers_per_gpu = 2,
        batch_size = 1,
        progress_interval = 100,
        eval_data = tmp_path / "eval.pkl",
        logdir = tmp_path,
        wandb = True,
        wandb_project = None,
        wandb_entity = None,
        wandb_run_name = None,
    )

    planned = planner.plan(args)

    assert resolve_calls == [
        "wandb:entity/marvel/policy_run",
        "wandb:entity/yomi/belief_run",
    ]
    prepared_agents = planned.run_config.unique_agents
    assert prepared_agents[0].display_name() == "marvel=wandb:entity/marvel/policy_run?action_temp=1"
    assert prepared_agents[0].checkpoint == policy_checkpoint
    assert prepared_agents[1].checkpoint == policy_checkpoint
    assert prepared_agents[1].belief_checkpoint == belief_checkpoint
    assert prepared_agents[1].search_beta == 2.0


def test_doudizhu_run_planner_defaults_wandb_to_marvel_project(monkeypatch, tmp_path: Path) -> None:
    planner = _planner()
    monkeypatch.delenv("WANDB_PROJECT", raising = False)
    monkeypatch.delenv("WANDB_ENTITY", raising = False)
    args = argparse.Namespace(
        agents = ["perfectdou", "douzero"],
        num_games = 8,
        workers_per_gpu = 2,
        batch_size = 2,
        progress_interval = 100,
        eval_data = tmp_path / "eval.pkl",
        logdir = tmp_path,
        wandb = True,
        wandb_project = None,
        wandb_entity = None,
        wandb_run_name = None,
    )

    planned = planner.plan(args)
    run_config = planned.run_config
    assert run_config.wandb.enabled is True
    assert run_config.wandb.project == "marvel"
    assert run_config.wandb.entity is None
    assert run_config.wandb.run_name is None


def test_doudizhu_run_planner_rejects_custom_wandb_run_name_for_multiple_matchups(tmp_path: Path) -> None:
    planner = _planner()
    args = argparse.Namespace(
        agents = ["perfectdou", "douzero", "fpdou"],
        num_games = 8,
        workers_per_gpu = 2,
        batch_size = 2,
        progress_interval = 100,
        eval_data = tmp_path / "eval.pkl",
        logdir = tmp_path,
        wandb = True,
        wandb_project = None,
        wandb_entity = None,
        wandb_run_name = "custom",
    )

    with pytest.raises(ValueError, match = "exactly one matchup"):
        planner.plan(args)


def test_doudizhu_scenario_produces_run_plan(tmp_path: Path) -> None:
    (tmp_path / "eval.pkl").write_bytes(pickle.dumps([{"steps": 1}], protocol = 4))
    parser = argparse.ArgumentParser()
    scenario = DoudizhuScenario()
    scenario.configure_parser(parser)
    args = parser.parse_args(
        [
            "perfectdou",
            "--num-games",
            "4",
            "--batch-size",
            "2",
            "--no-wandb",
            "--eval-data",
            str(tmp_path / "eval.pkl"),
            "--logdir",
            str(tmp_path),
        ]
    )

    scenario._planner = _planner()  # noqa: SLF001
    run_plan = scenario.plan_run(args)

    assert run_plan.logdir == tmp_path.resolve()
    assert len(run_plan.logical_jobs) == 1
    assert run_plan.workloads[0].total_units == 4


def test_doudizhu_runtime_factory_reuses_one_controller_pool_under_parallel_session_creation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    created_pools: list[object] = []
    captured_pools: list[object] = []

    class _FakePool:
        def __init__(self) -> None:
            self.closed = False
            created_pools.append(self)

        def close(self) -> None:
            self.closed = True

    class _FakeController:
        def __init__(self, *, pool, **kwargs) -> None:  # noqa: ANN001
            del kwargs
            captured_pools.append(pool)

    monkeypatch.setattr(doudizhu_scenario_module, "WorkerLocalControllerPool", _FakePool)
    monkeypatch.setattr(doudizhu_scenario_module, "DoudizhuSocketController", _FakeController)
    monkeypatch.setattr(doudizhu_scenario_module, "DoudizhuHostSession", lambda **kwargs: object())
    monkeypatch.setattr(doudizhu_scenario_module, "DoudizhuAggregator", lambda **kwargs: object())

    eval_data = tmp_path / "eval.pkl"
    eval_data.write_bytes(pickle.dumps([{"steps": 1}], protocol = 4))

    factory = DoudizhuRuntimeFactory(
        run_config = SimpleNamespace(eval_data = eval_data),
        logdir = tmp_path,
    )
    assert len(created_pools) == 1

    agent = specs.AgentSpec(kind = "perfectdou", canonical_spec = "perfectdou")
    plan = specs.SeatingPlan(
        matchup_id = "matchup",
        matchup_name = "DouDiZhu_matchup",
        mode = "selfplay",
        role_pass = "selfplay",
        left_agent = agent,
        right_agent = agent,
        landlord_agent = agent,
        landlord_down_agent = agent,
        landlord_up_agent = agent,
    )

    jobs = [
        EvaluationJob(
            job_id = f"job_{idx}",
            scenario_name = "doudizhu",
            mode = "selfplay",
            artifacts_dir = tmp_path / "jobs" / f"job_{idx}",
            requires_gpu = False,
            logical_job_id = f"logical_{idx}",
            config = {
                "plan": plan,
                "eval_data": tmp_path / "eval.pkl",
                "num_games": 1,
                "host_cfg": object(),
            },
        )
        for idx in range(2)
    ]
    slots = [WorkerSlot(worker_id = idx, gpu_token = None) for idx in range(2)]
    assignments = [
        WorkerAssignment(
            assignment_id = f"{job.logical_job_id}_worker{idx + 1:03d}of001",
            logical_job = job,
            slot = slot,
            worker_index = 0,
            worker_count = 1,
            work_allocator = SharedUnitAllocator(total_units = 1),
        )
        for idx, (job, slot) in enumerate(zip(jobs, slots, strict = False))
    ]

    start = threading.Event()
    errors: list[Exception] = []

    def _thread_target(assignment) -> None:  # noqa: ANN001
        start.wait()
        try:
            factory.create_session(assignment = assignment, emit = lambda event: None)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target = _thread_target, args = (assignment,), daemon = True)
        for assignment in assignments
    ]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(timeout = 5.0)
        assert not thread.is_alive()

    assert not errors
    assert len(created_pools) == 1
    assert len(captured_pools) == 6
    assert all(pool is created_pools[0] for pool in captured_pools)

    factory.close()
    assert created_pools[0].closed is True
