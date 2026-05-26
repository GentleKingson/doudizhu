from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import pickle
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qsl

from evaluations.core import EvaluationJob, JobWorkload, WorkerLayout, WorkerLayoutPlanner, WorkerSlot
from evaluations.core.plan import EvaluationRunPlan
from evaluations.doudizhu.runtime import (
    DEFAULT_EVAL_DATA,
    DEFAULT_HOST,
    DEFAULT_MARVEL_AGENT,
    DoudizhuHostConfig,
    REPO_ROOT,
)
from kagami.core.specs import PolicySpec, PolicySpecParser
from kagami.search.routines import (
    SEARCH_ROUTINE_LOCAL_SUBGAME,
    default_local_search_floor,
    normalize_search_routine,
)
from marvel.game_identity import resolve_nanopokerlib_game_identifier
from marvel.runtime.checkpoints import CheckpointStore


POSITIONS = ("landlord", "landlord_down", "landlord_up")
_GPU_REQUIRED_KINDS = {"marvel_direct", "marvel_search", "douzero", "fpdou"}
_TEMP_KEY_ORDER = ("action_temp", "belief_temp", "search_temp")
_SEARCH_KEY_ORDER = ("routine", "search_beta", "search_budget", "search_floor")
_MARVEL_QUERY_KEY_ORDER = _TEMP_KEY_ORDER + _SEARCH_KEY_ORDER
DEFAULT_WORKERS_PER_GPU = 2
DOUDIZHU_GAME_IDENTIFIER = resolve_nanopokerlib_game_identifier("doudizhu")


@dataclass(frozen = True)
class AgentSpec(PolicySpec):
    checkpoint: str | None = None
    belief_checkpoint: str | None = None
    action_temp: float = float(DEFAULT_MARVEL_AGENT.action_temp)
    belief_temp: float = float(DEFAULT_MARVEL_AGENT.belief_temp)
    search_temp: float = float(DEFAULT_MARVEL_AGENT.search_temp)
    search_routine: str = str(DEFAULT_MARVEL_AGENT.search.routine)
    search_beta: float = float(DEFAULT_MARVEL_AGENT.search.search_beta)
    search_budget: int = int(DEFAULT_MARVEL_AGENT.search.rollout_budget)
    search_floor: int | None = DEFAULT_MARVEL_AGENT.search.search_floor

    def display_name(self) -> str:
        return self.canonical_spec

    def log_name(self) -> str:
        return _compact_agent_log_name(self.canonical_spec)

    def requires_gpu(self) -> bool:
        return self.kind in _GPU_REQUIRED_KINDS


@dataclass(frozen = True)
class SeatingPlan:
    matchup_id: str
    matchup_name: str
    mode: str
    role_pass: str
    left_agent: AgentSpec
    right_agent: AgentSpec
    landlord_agent: AgentSpec
    landlord_down_agent: AgentSpec
    landlord_up_agent: AgentSpec

    def seat_to_agent(self) -> dict[str, AgentSpec]:
        return {
            "landlord": self.landlord_agent,
            "landlord_down": self.landlord_down_agent,
            "landlord_up": self.landlord_up_agent,
        }

    def requires_gpu(self) -> bool:
        return any(agent.requires_gpu() for agent in self.seat_to_agent().values())

    def role_label(self, *, compact: bool = False) -> str:
        if compact:
            landlord = self.landlord_agent.log_name()
            landlord_down = self.landlord_down_agent.log_name()
            landlord_up = self.landlord_up_agent.log_name()
        else:
            landlord = self.landlord_agent.display_name()
            landlord_down = self.landlord_down_agent.display_name()
            landlord_up = self.landlord_up_agent.display_name()
        return format_role_assignment(
            landlord = landlord,
            landlord_down = landlord_down,
            landlord_up = landlord_up,
        )

    def display_label(self) -> str:
        return f"{self.matchup_name} [{self.role_label(compact = True)}]"

    def progress_scope_label(self) -> str:
        return self.display_label()

    def role_key(self) -> str:
        landlord = _slugify_role_component(self.landlord_agent.log_name())
        if self.landlord_down_agent.canonical_spec == self.landlord_up_agent.canonical_spec:
            peasants = _slugify_role_component(self.landlord_down_agent.log_name())
            return f"landlord_{landlord}__peasants_{peasants}"
        landlord_down = _slugify_role_component(self.landlord_down_agent.log_name())
        landlord_up = _slugify_role_component(self.landlord_up_agent.log_name())
        return f"landlord_{landlord}__peasant_down_{landlord_down}__peasant_up_{landlord_up}"

    def left_view_role(self) -> str:
        if self.mode != "head_to_head":
            return "selfplay"
        if self.landlord_agent.canonical_spec == self.left_agent.canonical_spec:
            return "as_landlord"
        if (
            self.landlord_agent.canonical_spec == self.right_agent.canonical_spec
            and self.landlord_down_agent.canonical_spec == self.left_agent.canonical_spec
            and self.landlord_up_agent.canonical_spec == self.left_agent.canonical_spec
        ):
            return "as_peasants"
        raise RuntimeError(
            "Unexpected Dou Dizhu seating plan for left-view metric mapping: "
            f"{self.role_label()}."
        )


@dataclass(frozen = True)
class DoudizhuWandbConfig:
    enabled: bool
    project: str
    entity: str | None
    run_name: str | None


@dataclass(frozen = True)
class DoudizhuRunConfig:
    input_agents: tuple[str, ...]
    unique_agents: tuple[AgentSpec, ...]
    eval_data: Path
    num_games: int
    batch_size: int
    host_cfg: DoudizhuHostConfig
    worker_layout: WorkerLayout
    wandb: DoudizhuWandbConfig


@dataclass(frozen = True)
class DoudizhuPlannedRun:
    logdir: Path
    run_config: DoudizhuRunConfig
    logical_jobs: tuple[EvaluationJob, ...]
    workloads: tuple[JobWorkload, ...]
    slots: tuple[WorkerSlot, ...]


def parse_doudizhu_args(parser: argparse.ArgumentParser) -> None:
    DoudizhuRunPlanner().configure_parser(parser)


def resolve_logdir(args) -> Path:
    return DoudizhuRunPlanner().resolve_logdir(args)


def format_role_assignment(*, landlord: str, landlord_down: str, landlord_up: str) -> str:
    if str(landlord_down) == str(landlord_up):
        return f"Landlord={landlord}, Peasants={landlord_down}"
    return (
        f"Landlord={landlord}, "
        f"PeasantDown={landlord_down}, "
        f"PeasantUp={landlord_up}"
    )


def format_matchup_name(*, left: AgentSpec, right: AgentSpec | None = None, mode: str) -> str:
    if str(mode) == "selfplay":
        return f"{DOUDIZHU_GAME_IDENTIFIER}_{left.log_name()}_selfplay"
    if right is None:
        raise ValueError("Head-to-head matchup naming requires both left and right agents.")
    return f"{DOUDIZHU_GAME_IDENTIFIER}_{left.log_name()}_vs_{right.log_name()}"


class DoudizhuAgentSpecParser(PolicySpecParser):
    def parse(self, raw: str) -> AgentSpec:
        spec = str(raw).strip()
        if not spec:
            raise ValueError("Empty agent spec.")
        base_no_query = spec.partition("?")[0]
        if base_no_query in {"perfectdou", "douzero", "fpdou"}:
            if "?" in spec:
                raise ValueError("Only MARVEL specs support '?key=value' overrides.")
            return AgentSpec(kind = spec, canonical_spec = spec, checkpoint_locators = ())
        if "=" not in spec:
            raise ValueError(f"Unknown agent spec: {spec!r}")

        kinds, locators, raw_overrides = self.split_spec(raw)
        kind_key = "@".join(kinds)

        if kind_key == "marvel@yomi":
            if len(locators) != 2:
                raise ValueError(
                    "MARVEL+Yomi spec must be in the form "
                    "'marvel@yomi=<policy_checkpoint>@<belief_checkpoint>[?... ]'."
                )
            policy_ckpt = self.normalize_checkpoint_locator(locators[0], resolve_base = REPO_ROOT)
            belief_ckpt = self.normalize_checkpoint_locator(locators[1], resolve_base = REPO_ROOT)
            overrides = self._parse_marvel_overrides(raw_overrides, allow_search_budget = True)
            action_temp = float(overrides.get("action_temp", DEFAULT_MARVEL_AGENT.action_temp))
            belief_temp = float(overrides.get("belief_temp", DEFAULT_MARVEL_AGENT.belief_temp))
            search_temp = float(overrides.get("search_temp", DEFAULT_MARVEL_AGENT.search_temp))
            search_routine = str(overrides.get("routine", DEFAULT_MARVEL_AGENT.search.routine))
            search_beta = float(overrides.get("search_beta", DEFAULT_MARVEL_AGENT.search.search_beta))
            search_budget = int(overrides.get("search_budget", DEFAULT_MARVEL_AGENT.search.rollout_budget))
            if "search_floor" in overrides:
                search_floor_raw = overrides["search_floor"]
            elif search_routine == SEARCH_ROUTINE_LOCAL_SUBGAME:
                search_floor_raw = default_local_search_floor(search_budget)
            else:
                search_floor_raw = None
            search_floor = None if search_floor_raw is None else int(search_floor_raw)
            canonical = f"marvel@yomi={policy_ckpt}@{belief_ckpt}"
            canonical_query = self._canonical_marvel_query(overrides)
            if canonical_query:
                canonical += f"?{canonical_query}"
            return AgentSpec(
                kind = "marvel_search",
                canonical_spec = canonical,
                checkpoint_locators = (policy_ckpt, belief_ckpt),
                checkpoint = policy_ckpt,
                belief_checkpoint = belief_ckpt,
                action_temp = action_temp,
                belief_temp = belief_temp,
                search_temp = search_temp,
                search_routine = search_routine,
                search_beta = search_beta,
                search_budget = search_budget,
                search_floor = search_floor,
            )

        if kind_key == "marvel":
            if len(locators) != 1:
                raise ValueError("MARVEL direct spec requires exactly one checkpoint locator.")
            checkpoint = self.normalize_checkpoint_locator(locators[0], resolve_base = REPO_ROOT)
            overrides = self._parse_marvel_overrides(raw_overrides, allow_search_budget = False)
            action_temp = float(overrides.get("action_temp", DEFAULT_MARVEL_AGENT.action_temp))
            belief_temp = float(overrides.get("belief_temp", DEFAULT_MARVEL_AGENT.belief_temp))
            search_temp = float(overrides.get("search_temp", DEFAULT_MARVEL_AGENT.search_temp))
            canonical = f"marvel={checkpoint}"
            canonical_query = self._canonical_marvel_query(overrides)
            if canonical_query:
                canonical += f"?{canonical_query}"
            return AgentSpec(
                kind = "marvel_direct",
                canonical_spec = canonical,
                checkpoint_locators = (checkpoint,),
                checkpoint = checkpoint,
                action_temp = action_temp,
                belief_temp = belief_temp,
                search_temp = search_temp,
            )

        raise ValueError(f"Unknown agent spec: {spec!r}")

    def _parse_marvel_overrides(
        self,
        raw_overrides: dict[str, str],
        *,
        allow_search_budget: bool,
    ) -> dict[str, float | int | str]:
        allowed_order = _MARVEL_QUERY_KEY_ORDER if allow_search_budget else _TEMP_KEY_ORDER
        parsed: dict[str, float | int | str] = {}
        for key, raw_value in raw_overrides.items():
            if key not in allowed_order:
                choices = ", ".join(allowed_order)
                raise ValueError(f"Unknown MARVEL override key {key!r}. Allowed keys: {choices}.")
            if key == "routine":
                parsed[key] = normalize_search_routine(raw_value)
            elif key in {"search_budget", "search_floor"}:
                parsed[key] = self.parse_positive_int(raw_value, name = key)
            else:
                parsed[key] = self.parse_non_negative_float(raw_value, name = key)
        routine = str(parsed.get("routine", DEFAULT_MARVEL_AGENT.search.routine))
        search_budget = int(parsed.get("search_budget", DEFAULT_MARVEL_AGENT.search.rollout_budget))
        search_floor = parsed.get("search_floor")
        if search_floor is not None:
            if routine != SEARCH_ROUTINE_LOCAL_SUBGAME:
                raise ValueError("search_floor is only valid when routine='local-subgame'.")
            if int(search_floor) > search_budget:
                raise ValueError("search_floor must be <= search_budget.")
        return parsed

    def _canonical_marvel_query(self, overrides: dict[str, float | int | str]) -> str:
        if not overrides:
            return ""
        parts: list[str] = []
        for key in _MARVEL_QUERY_KEY_ORDER:
            if key not in overrides:
                continue
            if key in {"search_budget", "search_floor"}:
                parts.append(f"{key}={int(overrides[key])}")
            elif key == "routine":
                parts.append(f"{key}={str(overrides[key])}")
            else:
                parts.append(f"{key}={self.format_float(float(overrides[key]))}")
        return "&".join(parts)


class DoudizhuEvalDataProvider:
    def ensure_eval_data(self, path: Path, *, min_games: int, python_cmd: Sequence[str]) -> None:
        path = path.resolve()

        def _pickle_protocol(data_path: Path) -> int | None:
            try:
                with data_path.open("rb") as handle:
                    head = handle.read(2)
            except FileNotFoundError:
                return None
            if len(head) >= 2 and head[0] == 0x80:
                return int(head[1])
            return None

        if path.exists():
            try:
                with path.open("rb") as handle:
                    data = pickle.load(handle)
                if isinstance(data, list) and len(data) >= min_games:
                    protocol = _pickle_protocol(path)
                    if protocol is not None and protocol > 4:
                        with path.open("wb") as handle:
                            pickle.dump(data, handle, protocol = 4)
                    return
            except (pickle.PickleError, EOFError, OSError) as exc:
                raise RuntimeError(f"Failed to load existing eval data: {path}") from exc

        output_stem = str(path.with_suffix(""))
        cmd = list(python_cmd) + [
            str(REPO_ROOT / "external" / "PerfectDou" / "generate_eval_data.py"),
            "--output",
            output_stem,
            "--num_games",
            str(max(min_games, 10_000)),
        ]
        subprocess.run(cmd, cwd = REPO_ROOT, check = True)
        if not path.exists():
            raise RuntimeError(f"Eval data generation succeeded but file missing: {path}")
        try:
            with path.open("rb") as handle:
                data = pickle.load(handle)
            if isinstance(data, list) and len(data) >= min_games:
                with path.open("wb") as handle:
                    pickle.dump(data, handle, protocol = 4)
        except (pickle.PickleError, EOFError, OSError) as exc:
            raise RuntimeError(f"Failed to rewrite eval data with protocol=4: {path}") from exc


class DoudizhuCheckpointResolver:
    def resolve_agents(self, agents: Sequence[AgentSpec]) -> tuple[AgentSpec, ...]:
        resolved: dict[str, str] = {}
        for agent in agents:
            for locator in (agent.checkpoint, agent.belief_checkpoint):
                if locator is None:
                    continue
                text = str(locator).strip()
                if not text or text in resolved:
                    continue
                checkpoint_path = Path(
                    CheckpointStore.resolve_checkpoint_file_path(text)
                ).expanduser()
                if not checkpoint_path.is_absolute():
                    checkpoint_path = checkpoint_path.resolve()
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                resolved[text] = os.fspath(checkpoint_path)

        return tuple(self._substitute_agent_locators(agent, resolved_locators = resolved) for agent in agents)

    @staticmethod
    def _substitute_agent_locators(agent: AgentSpec, *, resolved_locators: dict[str, str]) -> AgentSpec:
        checkpoint = agent.checkpoint
        belief_checkpoint = agent.belief_checkpoint
        resolved_checkpoint = None if checkpoint is None else resolved_locators.get(str(checkpoint), str(checkpoint))
        resolved_belief_checkpoint = (
            None
            if belief_checkpoint is None
            else resolved_locators.get(str(belief_checkpoint), str(belief_checkpoint))
        )
        return replace(
            agent,
            checkpoint = resolved_checkpoint,
            belief_checkpoint = resolved_belief_checkpoint,
        )


class DoudizhuAgentCommandBuilder:
    def build(self, agent: AgentSpec, *, socket_path: Path, device: str) -> list[str]:
        if agent.kind == "perfectdou":
            return ["bash", "evaluations/doudizhu/agents/perfectdou.sh", str(socket_path)]
        if agent.kind == "douzero":
            return [sys.executable, "-u", "-m", "evaluations.doudizhu.agents.douzero", str(socket_path)]
        if agent.kind == "fpdou":
            return [sys.executable, "-u", "-m", "evaluations.doudizhu.agents.fpdou", str(socket_path)]
        if agent.kind in {"marvel_direct", "marvel_search"}:
            if agent.checkpoint is None:
                raise ValueError("MARVEL agent spec missing checkpoint.")
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "evaluations.doudizhu.agents.marvel",
                str(socket_path),
                str(agent.checkpoint),
                "--device",
                str(device),
                "--action-temp",
                _format_float(agent.action_temp),
                "--belief-temp",
                _format_float(agent.belief_temp),
                "--search-temp",
                _format_float(agent.search_temp),
            ]
            if agent.kind == "marvel_search":
                if agent.belief_checkpoint is None:
                    raise ValueError("MARVEL search agent spec missing belief checkpoint.")
                cmd.extend(
                    [
                        "--mode",
                        "search",
                        "--belief-checkpoint",
                        str(agent.belief_checkpoint),
                        "--routine",
                        str(agent.search_routine),
                        "--search-beta",
                        _format_float(agent.search_beta),
                        "--search-budget",
                        str(int(agent.search_budget)),
                    ]
                )
                if agent.search_floor is not None:
                    cmd.extend(["--search-floor", str(int(agent.search_floor))])
            return cmd
        raise ValueError(f"Unknown agent kind: {agent.kind}")


class DoudizhuMatchupPlanner:
    def build_jobs(
        self,
        *,
        unique_agents: Sequence[AgentSpec],
        run_config: DoudizhuRunConfig,
        logdir: Path,
    ) -> list[EvaluationJob]:
        if len(unique_agents) <= 0:
            raise RuntimeError("Expected at least one agent.")

        jobs: list[EvaluationJob] = []
        if len(unique_agents) == 1:
            only = unique_agents[0]
            matchup_id = f"{_spec_dir_component(only)}_selfplay"
            plan = SeatingPlan(
                matchup_id = matchup_id,
                matchup_name = format_matchup_name(left = only, mode = "selfplay"),
                mode = "selfplay",
                role_pass = "selfplay",
                left_agent = only,
                right_agent = only,
                landlord_agent = only,
                landlord_down_agent = only,
                landlord_up_agent = only,
            )
            jobs.append(
                self._build_job_from_plan(
                    plan = plan,
                    job_id = matchup_id,
                    logdir = logdir,
                    run_config = run_config,
                )
            )
            return jobs

        for pair_index, (left, right) in enumerate(itertools.combinations(unique_agents, 2)):
            matchup_id = f"{_spec_dir_component(left)}_vs_{_spec_dir_component(right)}"
            matchup_name = format_matchup_name(left = left, right = right, mode = "head_to_head")
            as_landlord = SeatingPlan(
                matchup_id = matchup_id,
                matchup_name = matchup_name,
                mode = "head_to_head",
                role_pass = "as_landlord",
                left_agent = left,
                right_agent = right,
                landlord_agent = left,
                landlord_down_agent = right,
                landlord_up_agent = right,
            )
            as_peasants = SeatingPlan(
                matchup_id = matchup_id,
                matchup_name = matchup_name,
                mode = "head_to_head",
                role_pass = "as_peasants",
                left_agent = left,
                right_agent = right,
                landlord_agent = right,
                landlord_down_agent = left,
                landlord_up_agent = left,
            )
            jobs.append(
                self._build_job_from_plan(
                    plan = as_landlord,
                    job_id = f"{matchup_id}_as_landlord",
                    logdir = logdir,
                    run_config = run_config,
                    pair_index = pair_index,
                )
            )
            jobs.append(
                self._build_job_from_plan(
                    plan = as_peasants,
                    job_id = f"{matchup_id}_as_peasants",
                    logdir = logdir,
                    run_config = run_config,
                    pair_index = pair_index,
                )
            )
        return jobs

    def describe_job_workload(self, job: EvaluationJob) -> JobWorkload:
        return JobWorkload(
            total_units = int(job.config["num_games"]),
            unit_name = "games",
            shardable = True,
        )

    def build_slots(self, run_config: DoudizhuRunConfig) -> tuple[WorkerSlot, ...]:
        return tuple(
            WorkerSlot(worker_id = idx, gpu_token = token)
            for idx, token in enumerate(run_config.worker_layout.slot_tokens)
        )

    def _build_job_from_plan(
        self,
        *,
        plan: SeatingPlan,
        job_id: str,
        logdir: Path,
        run_config: DoudizhuRunConfig,
        pair_index: int = 0,
    ) -> EvaluationJob:
        config: dict[str, object] = {
            "plan": plan,
            "pair_index": int(pair_index),
            "display_name": plan.display_label(),
            "status_context": {
                "matchup_id": str(plan.matchup_id),
                "matchup_name": str(plan.matchup_name),
                "role_key": str(plan.role_key()),
                "role_label": str(plan.role_label()),
                "left_view_role": str(plan.left_view_role()),
                "scope_label": str(plan.progress_scope_label()),
            },
            "run_config": run_config,
            "eval_data": run_config.eval_data,
            "num_games": int(run_config.num_games),
            "host_cfg": run_config.host_cfg,
        }
        return EvaluationJob(
            job_id = job_id,
            scenario_name = "doudizhu",
            mode = plan.mode,
            artifacts_dir = logdir / "jobs" / job_id,
            requires_gpu = plan.requires_gpu(),
            logical_job_id = job_id,
            config = config,
        )


class DoudizhuRunPlanner:
    def __init__(
        self,
        *,
        agent_parser: DoudizhuAgentSpecParser | None = None,
        eval_data_provider: DoudizhuEvalDataProvider | None = None,
        checkpoint_resolver: DoudizhuCheckpointResolver | None = None,
        layout_planner: WorkerLayoutPlanner | None = None,
        matchup_planner: DoudizhuMatchupPlanner | None = None,
    ) -> None:
        self._agent_parser = DoudizhuAgentSpecParser() if agent_parser is None else agent_parser
        self._eval_data_provider = DoudizhuEvalDataProvider() if eval_data_provider is None else eval_data_provider
        self._checkpoint_resolver = (
            DoudizhuCheckpointResolver() if checkpoint_resolver is None else checkpoint_resolver
        )
        self._layout_planner = WorkerLayoutPlanner() if layout_planner is None else layout_planner
        self._matchup_planner = DoudizhuMatchupPlanner() if matchup_planner is None else matchup_planner

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("agents", nargs = "+", help = "Agent specs.")
        parser.add_argument("--num-games", "--num_games", dest = "num_games", type = int, default = int(DEFAULT_HOST.num_games))
        parser.add_argument(
            "--workers-per-gpu",
            "--workers_per_gpu",
            dest = "workers_per_gpu",
            type = int,
            default = int(DEFAULT_WORKERS_PER_GPU),
        )
        parser.add_argument("--batch-size", "--batch_size", dest = "batch_size", type = int, required = True)
        parser.add_argument(
            "--progress-interval",
            "--progress_interval",
            dest = "progress_interval",
            type = int,
            default = int(DEFAULT_HOST.progress_interval),
        )
        parser.add_argument("--eval-data", type = Path, default = DEFAULT_EVAL_DATA)
        parser.add_argument("--logdir", type = Path, default = None)
        parser.add_argument("--wandb", action = argparse.BooleanOptionalAction, default = True)
        parser.add_argument("--wandb-project", type = str, default = None)
        parser.add_argument("--wandb-entity", type = str, default = None)
        parser.add_argument("--wandb-run-name", type = str, default = None)

    def resolve_logdir(self, args) -> Path:
        if args.logdir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            return REPO_ROOT / "tmp" / "evaluations" / "doudizhu" / stamp
        logdir = Path(args.logdir)
        if not logdir.is_absolute():
            logdir = (REPO_ROOT / logdir).resolve()
        return logdir

    def plan(self, args) -> DoudizhuPlannedRun:
        logdir = self.resolve_logdir(args)
        unique_agents = self._prepare_unique_agents(args.agents)
        self._validate_agent_checkpoints(unique_agents)
        run_config = self._build_run_config(args, unique_agents = unique_agents)
        resolved_agents = self._checkpoint_resolver.resolve_agents(run_config.unique_agents)
        run_config = replace(run_config, unique_agents = resolved_agents)

        logical_jobs = tuple(
            self._matchup_planner.build_jobs(
                unique_agents = run_config.unique_agents,
                run_config = run_config,
                logdir = logdir,
            )
        )
        workloads = tuple(
            self._matchup_planner.describe_job_workload(job)
            for job in logical_jobs
        )
        slots = self._matchup_planner.build_slots(run_config)
        return DoudizhuPlannedRun(
            logdir = logdir,
            run_config = run_config,
            logical_jobs = logical_jobs,
            workloads = workloads,
            slots = slots,
        )

    def _prepare_unique_agents(self, raw_agents: Sequence[str]) -> tuple[AgentSpec, ...]:
        parsed_agents = [self._agent_parser.parse(item) for item in raw_agents]
        unique_agents: list[AgentSpec] = []
        seen: set[str] = set()
        for agent in parsed_agents:
            if agent.canonical_spec in seen:
                print(f"[Warning] duplicate agent spec ignored: {agent.canonical_spec}", file = sys.stderr)
                continue
            unique_agents.append(agent)
            seen.add(agent.canonical_spec)
        if not unique_agents:
            raise ValueError("No unique agents provided.")
        return tuple(unique_agents)

    def _validate_agent_checkpoints(self, agents: Sequence[AgentSpec]) -> None:
        for agent in agents:
            if agent.kind in {"marvel_direct", "marvel_search"} and not _checkpoint_locator_exists(agent.checkpoint):
                raise ValueError(f"MARVEL checkpoint not found: {agent.checkpoint}")
            if agent.kind == "marvel_search" and not _checkpoint_locator_exists(agent.belief_checkpoint):
                raise ValueError(f"MARVEL belief checkpoint not found: {agent.belief_checkpoint}")

    def _build_run_config(
        self,
        args,
        *,
        unique_agents: Sequence[AgentSpec],
    ) -> DoudizhuRunConfig:
        num_games = int(args.num_games)
        if num_games <= 0:
            raise ValueError("--num-games must be positive.")
        batch_size = int(args.batch_size)
        if batch_size <= 0:
            raise ValueError("--batch-size must be a positive integer.")
        workers_per_gpu = int(args.workers_per_gpu)
        if workers_per_gpu <= 0:
            raise ValueError("--workers-per-gpu must be a positive integer.")
        progress_interval = int(args.progress_interval)
        wandb_run_name = None if args.wandb_run_name is None else str(args.wandb_run_name).strip()
        if wandb_run_name == "":
            raise ValueError("--wandb-run-name must be non-empty when provided.")
        if wandb_run_name is not None and _count_matchups(unique_agents) != 1:
            raise ValueError("--wandb-run-name requires an invocation that resolves to exactly one matchup.")

        eval_data = Path(args.eval_data)
        if not eval_data.is_absolute():
            eval_data = (REPO_ROOT / eval_data).resolve()
        self._eval_data_provider.ensure_eval_data(eval_data, min_games = num_games, python_cmd = [sys.executable])

        requires_gpu_jobs = any(agent.requires_gpu() for agent in unique_agents)
        usable_tokens = self._layout_planner.discover_usable_gpu_tokens()
        visible_tokens = self._layout_planner.discover_visible_gpu_tokens()
        unusable_tokens = [token for token in visible_tokens if token not in usable_tokens]
        for token in unusable_tokens:
            print(f"[Warning] GPU token '{token}' is not usable; excluding from scheduler.", file = sys.stderr)

        worker_layout = self._layout_planner.build_for_workers_per_gpu(
            workers_per_gpu = workers_per_gpu,
            prefer_gpu = requires_gpu_jobs,
        )

        host_cfg = DoudizhuHostConfig(
            num_games = int(num_games),
            max_envs = int(batch_size),
            progress_interval = int(progress_interval),
            ci_multiplier = float(DEFAULT_HOST.ci_multiplier),
            perfect_info = bool(DEFAULT_HOST.perfect_info),
        )
        project = str(args.wandb_project).strip() if args.wandb_project is not None else ""
        entity = str(args.wandb_entity).strip() if args.wandb_entity is not None else ""
        if not project:
            project = os.environ.get("WANDB_PROJECT", "").strip() or "marvel"
        if not entity:
            entity = os.environ.get("WANDB_ENTITY", "").strip()
        wandb = DoudizhuWandbConfig(
            enabled = bool(args.wandb),
            project = str(project),
            entity = None if not entity else str(entity),
            run_name = wandb_run_name,
        )
        return DoudizhuRunConfig(
            input_agents = tuple(str(item) for item in args.agents),
            unique_agents = tuple(unique_agents),
            eval_data = eval_data,
            num_games = int(num_games),
            batch_size = int(batch_size),
            host_cfg = host_cfg,
            worker_layout = worker_layout,
            wandb = wandb,
        )


def _format_float(value: float) -> str:
    return PolicySpecParser.format_float(value)


def _compact_checkpoint_locator(locator: str) -> str:
    text = str(locator).strip()
    if not text:
        return "<unknown>"
    if text.startswith("wandb:"):
        return text.rsplit("/", maxsplit = 1)[-1]
    basename = Path(text).name.strip()
    if basename:
        return basename
    return text.rsplit("/", maxsplit = 1)[-1] or text


def _compact_agent_log_name(canonical_spec: str) -> str:
    spec = str(canonical_spec).strip()
    if "=" not in spec or spec.startswith(("perfectdou", "douzero", "fpdou")):
        return spec
    base, has_query, query = spec.partition("?")
    kind, _, payload = base.partition("=")
    compact_params: list[str] = []
    if has_query:
        for key, raw_value in parse_qsl(query, keep_blank_values = True):
            compact_params.append(f"{key}={raw_value}")
    if kind == "marvel@yomi":
        policy_ref, sep, belief_ref = payload.partition("@")
        compact_payload = _compact_checkpoint_locator(policy_ref)
        if sep:
            compact_payload += f"+{_compact_checkpoint_locator(belief_ref)}"
    else:
        compact_payload = _compact_checkpoint_locator(payload)
    suffix = "" if not compact_params else f"({','.join(compact_params)})"
    return f"{kind}[{compact_payload}]{suffix}"


def _checkpoint_locator_exists(locator: str | None) -> bool:
    if locator is None:
        return False
    if locator.startswith("wandb:"):
        return True
    return Path(locator).exists()


def _count_matchups(unique_agents: Sequence[AgentSpec]) -> int:
    count = len(unique_agents)
    if count <= 1:
        return 1
    return (count * (count - 1)) // 2


def _slugify_role_component(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    return normalized or "unknown"


def _sanitize_dir_component(text: str) -> str:
    return PolicySpecParser.sanitize_dir_component(text)


def _spec_dir_component(agent: AgentSpec) -> str:
    normalized = _sanitize_dir_component(agent.canonical_spec)
    suffix = hashlib.sha1(agent.canonical_spec.encode("utf-8")).hexdigest()[:8]
    if not normalized:
        return suffix
    return f"{normalized[:48]}_{suffix}"


__all__ = [
    "AgentSpec",
    "DoudizhuAgentCommandBuilder",
    "DoudizhuAgentSpecParser",
    "DoudizhuCheckpointResolver",
    "DoudizhuEvalDataProvider",
    "DoudizhuMatchupPlanner",
    "DoudizhuPlannedRun",
    "DoudizhuRunConfig",
    "DoudizhuRunPlanner",
    "DoudizhuWandbConfig",
    "POSITIONS",
    "SeatingPlan",
    "WorkerLayout",
    "format_matchup_name",
    "format_role_assignment",
    "parse_doudizhu_args",
    "resolve_logdir",
]
