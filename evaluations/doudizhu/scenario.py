from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from evaluations.core import EvaluationRunPlan, RunScopedControllerRuntimeFactory, WorkerAssignment, WorkerLocalControllerPool
from evaluations.doudizhu.controller import DoudizhuSocketController
from evaluations.doudizhu.host import DoudizhuHostSession
from evaluations.doudizhu.specs import DoudizhuPlannedRun, DoudizhuRunConfig, DoudizhuRunPlanner, SeatingPlan
from evaluations.doudizhu.summary import DoudizhuAggregator, DoudizhuSummaryWriter


class DoudizhuRuntimeFactory(RunScopedControllerRuntimeFactory):
    def __init__(
        self,
        *,
        run_config: DoudizhuRunConfig,
        logdir: Path,
    ) -> None:
        super().__init__(controller_pool = WorkerLocalControllerPool())
        self._run_config = run_config
        self._logdir = Path(logdir).resolve()
        with Path(run_config.eval_data).resolve().open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Eval data must be a list, got {type(data).__name__}.")
        self._card_play_data = tuple(dict(item) for item in data)

    def _create_host(self, assignment: WorkerAssignment) -> DoudizhuHostSession:
        logical_job = assignment.logical_job
        plan = logical_job.config["plan"]
        return DoudizhuHostSession(
            job_id = assignment.assignment_id,
            logical_job_id = logical_job.logical_job_id,
            plan = plan,
            card_play_data = self._card_play_data,
            num_games = int(logical_job.config["num_games"]),
            host_cfg = logical_job.config["host_cfg"],
            work_allocator = assignment.work_allocator,
        )

    def _create_controllers(
        self,
        assignment: WorkerAssignment,
    ) -> dict[str, DoudizhuSocketController]:
        logical_job = assignment.logical_job
        plan = logical_job.config["plan"]
        return {
            seat: DoudizhuSocketController(
                pool = self.controller_pool,
                logdir = self._logdir,
                worker_id = int(assignment.slot.worker_id),
                job_id = assignment.assignment_id,
                controller_id = seat,
                agent = agent,
                gpu_token = assignment.slot.gpu_token,
                matchup_name = plan.matchup_name,
                role_key = plan.role_key(),
                role_label = plan.role_label(),
                scope_label = plan.progress_scope_label(),
            )
            for seat, agent in plan.seat_to_agent().items()
        }

    def _create_aggregator(self, assignment: WorkerAssignment) -> DoudizhuAggregator:
        return DoudizhuAggregator(
            assignment = assignment,
            run_config = self._run_config,
            plan = assignment.logical_job.config["plan"],
        )


class DoudizhuScenario:
    name = "doudizhu"

    def __init__(self) -> None:
        self._parser: argparse.ArgumentParser | None = None
        self._planner = DoudizhuRunPlanner()

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        self._parser = parser
        self._planner.configure_parser(parser)

    def plan_run(self, args) -> EvaluationRunPlan:
        if self._parser is None:
            raise RuntimeError("configure_parser() must be called before plan_run().")
        try:
            planned = self._planner.plan(args)
        except ValueError as exc:
            self._parser.error(str(exc))
        return self._build_run_plan(planned)

    @staticmethod
    def _build_run_plan(planned: DoudizhuPlannedRun) -> EvaluationRunPlan:
        planned_matchups: list[SeatingPlan] = []
        seen_matchup_ids: set[str] = set()
        for job in planned.logical_jobs:
            plan = job.config["plan"]
            matchup_id = str(plan.matchup_id)
            if matchup_id in seen_matchup_ids:
                continue
            seen_matchup_ids.add(matchup_id)
            planned_matchups.append(plan)
        return EvaluationRunPlan.create(
            logdir = planned.logdir,
            logical_jobs = planned.logical_jobs,
            workloads = planned.workloads,
            slots = planned.slots,
            summary_writer = DoudizhuSummaryWriter(
                logdir = planned.logdir,
                run_config = planned.run_config,
                planned_matchups = planned_matchups,
            ),
            runtime_factory = DoudizhuRuntimeFactory(
                run_config = planned.run_config,
                logdir = planned.logdir,
            ),
        )


__all__ = ["DoudizhuScenario"]
