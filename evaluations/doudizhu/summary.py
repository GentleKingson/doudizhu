from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Sequence

from evaluations.core.dispatch import WorkerAssignment
from evaluations.core.progress import is_normalized_progress
from evaluations.core.stats import MetricStats, RunningStats, metric_from_mean_ci, metric_from_running_stats
from evaluations.core.types import JobResult, QueueEvent
from evaluations.core.wandb import EvaluationWandbRun
from evaluations.doudizhu.specs import DoudizhuRunConfig, SeatingPlan, format_role_assignment


_ROLE_VIEW_ORDER = {"as_landlord": 0, "as_peasants": 1, "selfplay": 0}
_PUBLIC_WANDB_METRIC_PREFIX = {
    "role_avg_win_rate": "role_avg/win_rate",
    "role_avg_avg_gain": "role_avg/avg_gain",
    "as_landlord_win_rate": "as_landlord/win_rate",
    "as_landlord_avg_gain": "as_landlord/avg_gain",
    "as_peasants_win_rate": "as_peasants/win_rate",
    "as_peasants_avg_gain": "as_peasants/avg_gain",
}


@dataclass
class _DoudizhuLiveScopeState:
    scope_label: str
    total: int
    start_time_s: float
    completed: int = 0
    num_landlord_wins: int = 0
    sum_landlord_scores: float = 0.0
    sum_sq_landlord_scores: float = 0.0


def _metric_to_dict(metric: MetricStats) -> dict[str, float | int | str]:
    return {
        "n": int(metric.n),
        "mean": float(metric.mean),
        "std": float(metric.std),
        "sem": float(metric.sem),
        "ci": float(metric.ci),
        "unit": str(metric.unit),
    }


def _metric_from_samples(
    samples: Sequence[float],
    *,
    ci_multiplier: float,
    unit: str,
) -> MetricStats:
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("samples must be non-empty.")
    mean = sum(values) / float(len(values))
    mean_sq = sum(value * value for value in values) / float(len(values))
    variance = max(mean_sq - mean * mean, 0.0)
    ci = float(ci_multiplier) * math.sqrt(variance / float(len(values)))
    return metric_from_mean_ci(
        n = len(values),
        mean = float(mean),
        ci = float(ci),
        ci_multiplier = float(ci_multiplier),
        unit = unit,
    )


def _metric_display(metric: dict[str, object]) -> str:
    return f"{float(metric['mean']):.4f} (± {float(metric['ci']):.4f})"


def _metric_from_running_stats_for_logging(
    stats: RunningStats,
    *,
    ci_multiplier: float,
    unit: str,
) -> MetricStats:
    if int(stats.n) <= 0:
        raise ValueError("RunningStats must contain at least one sample.")
    if int(stats.n) == 1:
        return metric_from_mean_ci(
            n = 1,
            mean = float(stats.mean),
            ci = 0.0,
            ci_multiplier = float(ci_multiplier),
            unit = unit,
        )
    return metric_from_running_stats(
        stats,
        ci_multiplier = float(ci_multiplier),
        unit = unit,
    )


def _role_label_from_agents(agents: dict[str, object]) -> str:
    return format_role_assignment(
        landlord = str(agents["landlord"]),
        landlord_down = str(agents["landlord_down"]),
        landlord_up = str(agents["landlord_up"]),
    )


def _build_success_payload(
    *,
    run_config: DoudizhuRunConfig,
    job_kind: str,
    reported_num_games: int,
    logical_job_id: str,
    worker_index: int | None,
    worker_count: int | None,
    matchup_id: str,
    matchup_name: str,
    mode: str,
    role_key: str,
    role_label: str,
    left_agent: str,
    right_agent: str,
    left_view_role: str,
    agents: dict[str, str],
    game_ids: Sequence[int],
    wins: Sequence[int],
    scores: Sequence[float],
    warnings: Sequence[str],
    hands_per_s: float,
) -> dict[str, object]:
    landlord_win_rate = _metric_from_samples(
        wins,
        ci_multiplier = float(run_config.host_cfg.ci_multiplier),
        unit = "probability",
    )
    landlord_avg_gain = _metric_from_samples(
        scores,
        ci_multiplier = float(run_config.host_cfg.ci_multiplier),
        unit = "landlord_score",
    )
    peasants_win_rate = metric_from_mean_ci(
        n = landlord_win_rate.n,
        mean = 1.0 - float(landlord_win_rate.mean),
        ci = float(landlord_win_rate.ci),
        ci_multiplier = float(run_config.host_cfg.ci_multiplier),
        unit = "probability",
    )
    peasants_avg_gain = metric_from_mean_ci(
        n = landlord_avg_gain.n,
        mean = -float(landlord_avg_gain.mean),
        ci = float(landlord_avg_gain.ci),
        ci_multiplier = float(run_config.host_cfg.ci_multiplier),
        unit = "landlord_score",
    )
    payload = {
        "schema_version": 1,
        "scenario": "doudizhu",
        "runner": "evaluations_doudizhu",
        "job_kind": str(job_kind),
        "logical_job_id": str(logical_job_id),
        "matchup_id": str(matchup_id),
        "matchup_name": str(matchup_name),
        "mode": str(mode),
        "role_key": str(role_key),
        "role_label": str(role_label),
        "left_agent": str(left_agent),
        "right_agent": str(right_agent),
        "left_view_role": str(left_view_role),
        "num_games": int(reported_num_games),
        "logical_num_games": int(run_config.num_games),
        "batch_size": int(run_config.batch_size),
        "progress_interval": int(run_config.host_cfg.progress_interval),
        "ci_multiplier": float(run_config.host_cfg.ci_multiplier),
        "eval_data": str(run_config.eval_data),
        "agents": dict(agents),
        "warnings": list(warnings),
        "hands_per_s": float(hands_per_s),
        "metrics": {
            "landlord_win_rate": _metric_to_dict(landlord_win_rate),
            "landlord_avg_gain": _metric_to_dict(landlord_avg_gain),
            "peasants_team_win_rate": _metric_to_dict(peasants_win_rate),
            "peasants_team_avg_gain": _metric_to_dict(peasants_avg_gain),
        },
        "game_ids": list(int(value) for value in game_ids),
        "landlord_win_by_game": list(int(value) for value in wins),
        "landlord_score_by_game": list(float(value) for value in scores),
    }
    if worker_index is not None:
        payload["worker_index"] = int(worker_index)
    if worker_count is not None:
        payload["worker_count"] = int(worker_count)
    return payload


def _role_profile_metrics(payload: dict[str, object]) -> dict[str, dict[str, float | int | str]]:
    metrics = dict(payload["metrics"])
    return {
        "as_landlord_win_rate": dict(metrics["landlord_win_rate"]),
        "as_landlord_avg_gain": dict(metrics["landlord_avg_gain"]),
        "as_peasants_win_rate": dict(metrics["peasants_team_win_rate"]),
        "as_peasants_avg_gain": dict(metrics["peasants_team_avg_gain"]),
    }


def _left_view_metrics(
    *,
    as_landlord: dict[str, object],
    as_peasants: dict[str, object],
    ci_multiplier: float,
) -> dict[str, dict[str, float | int | str]]:
    landlord_wins = [int(value) for value in as_landlord["landlord_win_by_game"]]
    landlord_scores = [float(value) for value in as_landlord["landlord_score_by_game"]]
    peasants_ll_wins = [int(value) for value in as_peasants["landlord_win_by_game"]]
    peasants_ll_scores = [float(value) for value in as_peasants["landlord_score_by_game"]]
    if len(landlord_wins) != len(peasants_ll_wins) or len(landlord_scores) != len(peasants_ll_scores):
        raise RuntimeError("Head-to-head role results must have equal vector lengths for left-view metrics.")

    as_landlord_win_samples = [float(value) for value in landlord_wins]
    as_landlord_gain_samples = [float(value) for value in landlord_scores]
    as_peasants_win_samples = [1.0 - float(value) for value in peasants_ll_wins]
    as_peasants_gain_samples = [-float(value) for value in peasants_ll_scores]
    role_avg_win_samples = [
        0.5 * (float(left_ll_win) + float(left_peasant_win))
        for left_ll_win, left_peasant_win in zip(as_landlord_win_samples, as_peasants_win_samples, strict = False)
    ]
    role_avg_gain_samples = [
        0.5 * (float(left_ll_gain) + float(left_peasant_gain))
        for left_ll_gain, left_peasant_gain in zip(as_landlord_gain_samples, as_peasants_gain_samples, strict = False)
    ]
    return {
        "role_avg_win_rate": _metric_to_dict(
            _metric_from_samples(
                role_avg_win_samples,
                ci_multiplier = ci_multiplier,
                unit = "probability",
            )
        ),
        "role_avg_avg_gain": _metric_to_dict(
            _metric_from_samples(
                role_avg_gain_samples,
                ci_multiplier = ci_multiplier,
                unit = "landlord_score",
            )
        ),
        "as_landlord_win_rate": _metric_to_dict(
            _metric_from_samples(
                as_landlord_win_samples,
                ci_multiplier = ci_multiplier,
                unit = "probability",
            )
        ),
        "as_landlord_avg_gain": _metric_to_dict(
            _metric_from_samples(
                as_landlord_gain_samples,
                ci_multiplier = ci_multiplier,
                unit = "landlord_score",
            )
        ),
        "as_peasants_win_rate": _metric_to_dict(
            _metric_from_samples(
                as_peasants_win_samples,
                ci_multiplier = ci_multiplier,
                unit = "probability",
            )
        ),
        "as_peasants_avg_gain": _metric_to_dict(
            _metric_from_samples(
                as_peasants_gain_samples,
                ci_multiplier = ci_multiplier,
                unit = "landlord_score",
            )
        ),
    }


def _find_left_view_role_payloads(
    payloads: Sequence[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    as_landlord = None
    as_peasants = None
    for payload in payloads:
        role = str(payload.get("left_view_role", ""))
        if role == "as_landlord":
            as_landlord = payload
        elif role == "as_peasants":
            as_peasants = payload
    if as_landlord is None or as_peasants is None:
        raise RuntimeError("Expected both as_landlord and as_peasants role results for head-to-head matchup.")
    return as_landlord, as_peasants


def _role_sort_key(payload: dict[str, object]) -> tuple[int, str]:
    left_view_role = str(payload.get("left_view_role", ""))
    return (
        int(_ROLE_VIEW_ORDER.get(left_view_role, 99)),
        str(payload.get("role_key", "")),
    )


def _flatten_summary_metrics(metrics: dict[str, dict[str, float | int | str]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, metric in metrics.items():
        public_key = str(_PUBLIC_WANDB_METRIC_PREFIX[key])
        payload[f"{public_key}_mean"] = float(metric["mean"])
        payload[f"{public_key}_ci"] = float(metric["ci"])
    return payload


class _MatchupWandbLogger:
    _ROLE_STEP_METRICS: tuple[tuple[str, str], ...] = (
        ("as_landlord/*", "as_landlord/games"),
        ("as_peasants/*", "as_peasants/games"),
        ("role_avg/*", "role_avg/games"),
    )

    def __init__(self, *, run: EvaluationWandbRun, mode: str, ci_multiplier: float) -> None:
        self._run = run
        self._mode = str(mode)
        self._ci_multiplier = float(ci_multiplier)
        self._selfplay_buffer: dict[int, tuple[int, float]] = {}
        self._as_landlord_buffer: dict[int, tuple[int, float]] = {}
        self._as_peasants_buffer: dict[int, tuple[int, float]] = {}
        self._as_landlord_completed = 0
        self._as_peasants_completed = 0
        self._paired_completed = 0
        self._selfplay_completed = 0
        self._stats: dict[str, RunningStats] = {
            "role_avg_win_rate": RunningStats(),
            "role_avg_avg_gain": RunningStats(),
            "as_landlord_win_rate": RunningStats(),
            "as_landlord_avg_gain": RunningStats(),
            "as_peasants_win_rate": RunningStats(),
            "as_peasants_avg_gain": RunningStats(),
        }
        for glob_pattern, step_metric in self._ROLE_STEP_METRICS:
            run.define_metric(glob_pattern, step_metric = step_metric)

    def record_episode(self, body: dict[str, object]) -> None:
        episode_id = int(body["episode_id"])
        landlord_win = int(body["landlord_win"])
        landlord_score = float(body["landlord_score"])
        if self._mode == "head_to_head":
            role = str(body.get("left_view_role", ""))
            if role == "as_landlord":
                target = self._as_landlord_buffer
            elif role == "as_peasants":
                target = self._as_peasants_buffer
            else:
                raise RuntimeError(f"Unexpected left_view_role for head-to-head W&B logging: {role!r}.")
            if episode_id in target:
                raise RuntimeError(f"Duplicate episode payload for matched-role logging at episode_id={episode_id}.")
            target[episode_id] = (landlord_win, landlord_score)
            self._drain_head_to_head(role, landlord_win, landlord_score)
            return

        if episode_id in self._selfplay_buffer:
            raise RuntimeError(f"Duplicate self-play episode payload for W&B logging at episode_id={episode_id}.")
        self._selfplay_buffer[episode_id] = (landlord_win, landlord_score)
        self._drain_selfplay()

    def _drain_head_to_head(self, role: str, landlord_win: int, landlord_score: float) -> None:
        # Log per-seating metrics immediately with per-role step counters.
        if role == "as_landlord":
            self._stats["as_landlord_win_rate"].update(float(landlord_win))
            self._stats["as_landlord_avg_gain"].update(float(landlord_score))
            self._as_landlord_completed += 1
            self._run.log(
                {
                    "as_landlord/games": self._as_landlord_completed,
                    **self._logging_payload(keys = ("as_landlord_win_rate", "as_landlord_avg_gain")),
                },
            )
        else:
            self._stats["as_peasants_win_rate"].update(1.0 - float(landlord_win))
            self._stats["as_peasants_avg_gain"].update(-float(landlord_score))
            self._as_peasants_completed += 1
            self._run.log(
                {
                    "as_peasants/games": self._as_peasants_completed,
                    **self._logging_payload(keys = ("as_peasants_win_rate", "as_peasants_avg_gain")),
                },
            )

        # When both seatings are paired, also update role-averaged metrics.
        ready_episode_ids = sorted(self._as_landlord_buffer.keys() & self._as_peasants_buffer.keys())
        for episode_id in ready_episode_ids:
            ll_win, ll_gain = self._as_landlord_buffer.pop(episode_id)
            opp_ll_win, opp_ll_gain = self._as_peasants_buffer.pop(episode_id)
            as_landlord_win = float(ll_win)
            as_landlord_gain = float(ll_gain)
            as_peasants_win = 1.0 - float(opp_ll_win)
            as_peasants_gain = -float(opp_ll_gain)
            self._stats["role_avg_win_rate"].update(0.5 * (as_landlord_win + as_peasants_win))
            self._stats["role_avg_avg_gain"].update(0.5 * (as_landlord_gain + as_peasants_gain))
            self._paired_completed += 1
            self._run.log(
                {
                    "role_avg/games": self._paired_completed,
                    **self._logging_payload(keys = ("role_avg_win_rate", "role_avg_avg_gain")),
                },
            )

    def _drain_selfplay(self) -> None:
        for episode_id in sorted(self._selfplay_buffer):
            ll_win, ll_gain = self._selfplay_buffer.pop(episode_id)
            self._selfplay_completed += 1
            self._stats["as_landlord_win_rate"].update(float(ll_win))
            self._stats["as_landlord_avg_gain"].update(float(ll_gain))
            self._stats["as_peasants_win_rate"].update(1.0 - float(ll_win))
            self._stats["as_peasants_avg_gain"].update(-float(ll_gain))
            self._run.log(
                {
                    "as_landlord/games": self._selfplay_completed,
                    "as_peasants/games": self._selfplay_completed,
                    **self._logging_payload(
                        keys = (
                            "as_landlord_win_rate",
                            "as_landlord_avg_gain",
                            "as_peasants_win_rate",
                            "as_peasants_avg_gain",
                        )
                    ),
                },
            )

    def _logging_payload(self, *, keys: Sequence[str]) -> dict[str, float]:
        payload: dict[str, float] = {}
        for key in keys:
            metric = _metric_from_running_stats_for_logging(
                self._stats[key],
                ci_multiplier = self._ci_multiplier,
                unit = "probability" if "win_rate" in key else "landlord_score",
            )
            public_key = str(_PUBLIC_WANDB_METRIC_PREFIX[key])
            payload[f"{public_key}_mean"] = float(metric.mean)
            payload[f"{public_key}_ci"] = float(metric.ci)
        return payload


class DoudizhuResultBuilder:
    def __init__(self, *, assignment: WorkerAssignment, run_config: DoudizhuRunConfig, plan: SeatingPlan) -> None:
        self._assignment = assignment
        self._job = assignment.logical_job
        self._run_config = run_config
        self._plan = plan

    def build_success(
        self,
        *,
        game_ids: Sequence[int],
        wins: Sequence[int],
        scores: Sequence[float],
        warnings: Sequence[str],
        hands_per_s: float,
    ) -> JobResult:
        payload = _build_success_payload(
            run_config = self._run_config,
            job_kind = "seating_partial",
            reported_num_games = len(game_ids),
            logical_job_id = str(self._job.logical_job_id),
            worker_index = int(self._assignment.worker_index),
            worker_count = int(self._assignment.worker_count),
            matchup_id = self._plan.matchup_id,
            matchup_name = self._plan.matchup_name,
            mode = self._plan.mode,
            role_key = self._plan.role_key(),
            role_label = self._plan.role_label(),
            left_agent = self._plan.left_agent.display_name(),
            right_agent = self._plan.right_agent.display_name(),
            left_view_role = self._plan.left_view_role(),
            agents = self._agents_payload(),
            game_ids = game_ids,
            wins = wins,
            scores = scores,
            warnings = warnings,
            hands_per_s = hands_per_s,
        )
        return JobResult(
            job_id = self._assignment.assignment_id,
            ok = True,
            payload = payload,
            artifacts = {"job_dir": str(self._assignment_artifacts_dir())},
        )

    def build_failure(self, *, exc: Exception, warnings: Sequence[str]) -> JobResult:
        payload: dict[str, object] = {
            "schema_version": 1,
            "scenario": "doudizhu",
            "runner": "evaluations_doudizhu",
            "job_kind": "seating_partial",
            "logical_job_id": str(self._job.logical_job_id),
            "worker_index": int(self._assignment.worker_index),
            "worker_count": int(self._assignment.worker_count),
            "matchup_id": self._plan.matchup_id,
            "matchup_name": self._plan.matchup_name,
            "mode": self._plan.mode,
            "role_key": self._plan.role_key(),
            "role_label": self._plan.role_label(),
            "left_agent": self._plan.left_agent.display_name(),
            "right_agent": self._plan.right_agent.display_name(),
            "left_view_role": self._plan.left_view_role(),
            "num_games": 0,
            "logical_num_games": int(self._run_config.num_games),
            "ok": False,
            "error": str(exc),
            "agents": self._agents_payload(),
            "warnings": list(warnings),
        }
        return JobResult(
            job_id = self._assignment.assignment_id,
            ok = False,
            payload = payload,
            artifacts = {"job_dir": str(self._assignment_artifacts_dir())},
        )

    def _agents_payload(self) -> dict[str, str]:
        return {
            "landlord": self._plan.landlord_agent.display_name(),
            "landlord_down": self._plan.landlord_down_agent.display_name(),
            "landlord_up": self._plan.landlord_up_agent.display_name(),
        }

    def _assignment_artifacts_dir(self) -> Path:
        return self._job.artifacts_dir.parent / self._assignment.assignment_id


class DoudizhuAggregator:
    def __init__(self, *, assignment: WorkerAssignment, run_config: DoudizhuRunConfig, plan: SeatingPlan) -> None:
        self._result_builder = DoudizhuResultBuilder(assignment = assignment, run_config = run_config, plan = plan)
        self._wins_by_game: dict[int, int] = {}
        self._scores_by_game: dict[int, float] = {}
        self._warnings: list[str] = []
        self._hands_per_s = 0.0

    def record_episode(self, body: dict[str, object]) -> None:
        game_idx = int(body["episode_id"])
        if game_idx in self._wins_by_game or game_idx in self._scores_by_game:
            raise RuntimeError(f"Duplicate episode payload for game index {game_idx}.")
        self._wins_by_game[game_idx] = int(body["landlord_win"])
        self._scores_by_game[game_idx] = float(body["landlord_score"])

    def record_status(self, body: dict[str, object]) -> None:
        if "hands_per_s" in body:
            self._hands_per_s = float(body["hands_per_s"])
        level = str(body.get("level", "")).lower()
        if level in {"warning", "error"}:
            message = str(body.get("message", "")).strip()
            if message:
                self._warnings.append(message)

    def finalize(self) -> JobResult:
        game_ids, wins, scores = self._ordered_samples()
        return self._result_builder.build_success(
            game_ids = game_ids,
            wins = wins,
            scores = scores,
            warnings = self._warnings,
            hands_per_s = self._hands_per_s,
        )

    def build_failed_result(self, exc: Exception) -> JobResult:
        return self._result_builder.build_failure(exc = exc, warnings = self._warnings)

    def _ordered_samples(self) -> tuple[list[int], list[int], list[float]]:
        game_ids = sorted(self._wins_by_game)
        if len(game_ids) != len(self._scores_by_game):
            raise RuntimeError("Dou Dizhu seating completed with inconsistent sample maps.")
        missing = [idx for idx in game_ids if idx not in self._scores_by_game]
        if missing:
            raise RuntimeError(f"Dou Dizhu seating missing results for game indices: {missing[:8]}.")
        wins = [int(self._wins_by_game[idx]) for idx in game_ids]
        scores = [float(self._scores_by_game[idx]) for idx in game_ids]
        return game_ids, wins, scores


class _DoudizhuEventFormatter:
    def __init__(self) -> None:
        self._live_by_logical_job: dict[str, _DoudizhuLiveScopeState] = {}

    def format(self, event: QueueEvent) -> str | None:
        body = event.body
        if event.kind == "status":
            self._record_status_scope(event)
            message = str(body.get("message", "")).strip()
            if not message:
                return None
            scope_label = str(body.get("scope_label", "")).strip()
            if scope_label:
                prefix = f"[{scope_label}]"
            else:
                matchup_name = str(body.get("matchup_name", body.get("matchup_id", event.job_id)))
                role_label = str(body.get("role_label", body.get("role_key", ""))).strip()
                prefix = f"[{matchup_name}]"
                if role_label:
                    prefix += f"[{role_label}]"
            return f"{prefix} {message}"
        if event.kind == "episode":
            self._record_episode(event)
            return None
        if event.kind == "progress" and is_normalized_progress(body):
            return self._render_live_progress(event)
        if event.kind == "progress":
            matchup_name = str(body.get("matchup_name", body.get("matchup_id", event.job_id)))
            role_label = str(body.get("role_label", body.get("role_key", ""))).strip()
            prefix = f"{matchup_name} {role_label}".strip()
            return (
                f"{prefix} | {int(body['completed'])} / {int(body['total'])} | "
                f"Landlord Win Rate: {float(body['landlord_win_rate']):.3f} ± {float(body['landlord_win_rate_ci']):.3f} | "
                f"Landlord Avg Gain: {float(body['landlord_avg_gain']):.3f} ± {float(body['landlord_avg_gain_ci']):.3f} | "
                f"Hands/s: {float(body['hands_per_s']):.2f}"
            )
        return None

    def _record_status_scope(self, event: QueueEvent) -> None:
        body = event.body
        scope_label = str(body.get("scope_label", "")).strip()
        if not scope_label:
            return
        self._require_live_scope(
            logical_job_id = self._logical_job_id(body, fallback = event.job_id),
            scope_label = scope_label,
            total = self._scope_total(logical_job_id = self._logical_job_id(body, fallback = event.job_id)),
        )

    def _record_episode(self, event: QueueEvent) -> None:
        body = event.body
        logical_job_id = self._logical_job_id(body, fallback = event.job_id)
        state = self._require_live_scope(
            logical_job_id = logical_job_id,
            scope_label = self._scope_label(body, fallback = logical_job_id),
            total = self._scope_total(logical_job_id = logical_job_id),
        )
        state.completed += 1
        state.num_landlord_wins += int(body["landlord_win"])
        score = float(body["landlord_score"])
        state.sum_landlord_scores += score
        state.sum_sq_landlord_scores += score * score

    def _render_live_progress(self, event: QueueEvent) -> str | None:
        body = event.body
        logical_job_id = self._logical_job_id(body, fallback = event.job_id)
        state = self._require_live_scope(
            logical_job_id = logical_job_id,
            scope_label = self._scope_label(body, fallback = logical_job_id),
            total = int(body["total"]),
        )
        parts = [f"{int(state.completed)} / {int(state.total)} games"]
        if int(state.completed) > 0:
            completed = float(state.completed)
            landlord_win_rate = float(state.num_landlord_wins) / completed
            landlord_win_rate_ci = math.sqrt(landlord_win_rate * (1.0 - landlord_win_rate) / completed)
            mean_landlord_gain = float(state.sum_landlord_scores) / completed
            variance_landlord_gain = (float(state.sum_sq_landlord_scores) / completed) - mean_landlord_gain**2
            variance_landlord_gain = max(variance_landlord_gain, 0.0)
            landlord_gain_ci = math.sqrt(variance_landlord_gain / completed)
            parts.append(f"Landlord Win Rate: {landlord_win_rate:.3f} ± {landlord_win_rate_ci:.3f}")
            parts.append(f"Landlord Avg Gain: {mean_landlord_gain:.3f} ± {landlord_gain_ci:.3f}")
        elapsed_s = max(1e-6, time.perf_counter() - float(state.start_time_s))
        parts.append(f"Hands/s: {float(state.completed) / elapsed_s:.2f}")
        return f"{state.scope_label} | " + " | ".join(parts)

    def _require_live_scope(self, *, logical_job_id: str, scope_label: str, total: int) -> _DoudizhuLiveScopeState:
        state = self._live_by_logical_job.get(str(logical_job_id))
        if state is not None:
            if int(total) > 0:
                state.total = int(total)
            if str(scope_label).strip():
                state.scope_label = str(scope_label)
            return state
        state = _DoudizhuLiveScopeState(
            scope_label = str(scope_label),
            total = int(total),
            start_time_s = time.perf_counter(),
        )
        self._live_by_logical_job[str(logical_job_id)] = state
        return state

    def _scope_total(self, *, logical_job_id: str) -> int:
        state = self._live_by_logical_job.get(str(logical_job_id))
        return 0 if state is None else int(state.total)

    def _logical_job_id(self, body: dict[str, object], *, fallback: str) -> str:
        logical_job_id = str(body.get("logical_job_id", "")).strip()
        if logical_job_id:
            return logical_job_id
        matchup_id = str(body.get("matchup_id", "")).strip()
        role_key = str(body.get("role_key", "")).strip()
        if matchup_id and role_key:
            return f"{matchup_id}:{role_key}"
        if matchup_id:
            return matchup_id
        return str(fallback)

    def _scope_label(self, body: dict[str, object], *, fallback: str) -> str:
        scope_label = str(body.get("scope_label", "")).strip()
        if scope_label:
            return scope_label
        matchup_name = str(body.get("matchup_name", body.get("matchup_id", fallback))).strip()
        role_label = str(body.get("role_label", body.get("role_key", ""))).strip()
        return f"{matchup_name} [{role_label}]".strip() if role_label else matchup_name


class DoudizhuLogicalResultMerger:
    def __init__(self, *, run_config: DoudizhuRunConfig) -> None:
        self._run_config = run_config

    def merge(self, results: Sequence[JobResult]) -> list[JobResult]:
        ordered_ids: list[str] = []
        grouped: dict[str, list[JobResult]] = {}
        for result in results:
            logical_job_id = str(result.payload.get("logical_job_id", result.job_id))
            if logical_job_id not in grouped:
                ordered_ids.append(logical_job_id)
            grouped.setdefault(logical_job_id, []).append(result)

        merged: list[JobResult] = []
        for logical_job_id in ordered_ids:
            group = grouped[logical_job_id]
            if len(group) == 1 and "worker_count" not in group[0].payload:
                merged.append(group[0])
                continue
            merged.append(self._merge_partial_group(group))
        return merged

    def _merge_partial_group(self, results: Sequence[JobResult]) -> JobResult:
        results_sorted = sorted(
            results,
            key = lambda result: (
                int(result.payload.get("worker_index", 0)),
                str(result.job_id),
            ),
        )
        first = results_sorted[0]
        first_payload = first.payload
        logical_job_id = str(first_payload.get("logical_job_id", first.job_id))
        if any(not bool(result.ok) for result in results_sorted):
            warnings: list[str] = []
            errors: list[str] = []
            for result in results_sorted:
                payload = result.payload
                warnings.extend(str(item) for item in payload.get("warnings", []))
                if not bool(result.ok):
                    label = str(payload.get("role_label", result.job_id))
                    errors.append(f"{label}: {payload.get('error', 'unknown error')}")
            return JobResult(
                job_id = logical_job_id,
                ok = False,
                payload = {
                    "schema_version": 1,
                    "scenario": "doudizhu",
                    "runner": "evaluations_doudizhu",
                    "job_kind": "seating",
                    "logical_job_id": logical_job_id,
                    "matchup_id": first_payload.get("matchup_id", logical_job_id),
                    "matchup_name": first_payload.get("matchup_name", logical_job_id),
                    "mode": first_payload.get("mode", "unknown"),
                    "role_key": first_payload.get("role_key", logical_job_id),
                    "role_label": first_payload.get("role_label"),
                    "left_agent": first_payload.get("left_agent"),
                    "right_agent": first_payload.get("right_agent"),
                    "left_view_role": first_payload.get("left_view_role", "unknown"),
                    "num_games": int(self._run_config.num_games),
                    "logical_num_games": int(self._run_config.num_games),
                    "ok": False,
                    "error": "; ".join(errors),
                    "agents": first_payload.get("agents"),
                    "warnings": warnings,
                },
                artifacts = {"job_dir": str(first.artifacts.get("job_dir", first.job_id))},
            )

        merged_samples: dict[int, tuple[int, float]] = {}
        warnings: list[str] = []
        hands_per_s = 0.0
        for result in results_sorted:
            payload = result.payload
            game_ids = [int(value) for value in payload.get("game_ids", [])]
            partial_wins = [int(value) for value in payload["landlord_win_by_game"]]
            partial_scores = [float(value) for value in payload["landlord_score_by_game"]]
            if len(game_ids) != len(partial_wins) or len(game_ids) != len(partial_scores):
                raise RuntimeError(
                    f"Partial sample count mismatch for {logical_job_id}: "
                    f"{len(game_ids)} ids but {len(partial_wins)} wins and {len(partial_scores)} scores."
                )
            for game_id, win, score in zip(game_ids, partial_wins, partial_scores, strict = False):
                if int(game_id) in merged_samples:
                    raise RuntimeError(f"Duplicate game id {game_id} while merging {logical_job_id}.")
                merged_samples[int(game_id)] = (int(win), float(score))
            warnings.extend(str(item) for item in payload.get("warnings", []))
            hands_per_s += float(payload.get("hands_per_s", 0.0))
        expected_game_ids = list(range(1, int(self._run_config.num_games) + 1))
        missing = [game_id for game_id in expected_game_ids if game_id not in merged_samples]
        if missing:
            raise RuntimeError(f"Missing game ids while merging {logical_job_id}: {missing[:8]}.")
        merged_game_ids = sorted(merged_samples)
        merged_wins = [int(merged_samples[game_id][0]) for game_id in merged_game_ids]
        merged_scores = [float(merged_samples[game_id][1]) for game_id in merged_game_ids]

        merged_payload = _build_success_payload(
            run_config = self._run_config,
            job_kind = "seating",
            reported_num_games = int(self._run_config.num_games),
            logical_job_id = logical_job_id,
            worker_index = None,
            worker_count = None,
            matchup_id = str(first_payload.get("matchup_id", logical_job_id)),
            matchup_name = str(first_payload.get("matchup_name", logical_job_id)),
            mode = str(first_payload.get("mode", "unknown")),
            role_key = str(first_payload.get("role_key", logical_job_id)),
            role_label = str(first_payload.get("role_label", _role_label_from_agents(dict(first_payload.get("agents", {}))))),
            left_agent = str(first_payload.get("left_agent", "")),
            right_agent = str(first_payload.get("right_agent", "")),
            left_view_role = str(first_payload.get("left_view_role", "unknown")),
            agents = dict(first_payload.get("agents", {})),
            game_ids = merged_game_ids,
            wins = merged_wins,
            scores = merged_scores,
            warnings = warnings,
            hands_per_s = hands_per_s,
        )
        return JobResult(
            job_id = logical_job_id,
            ok = True,
            payload = merged_payload,
            artifacts = {"job_dir": str(first.artifacts.get("job_dir", first.job_id))},
        )


class DoudizhuMatchupSummaryBuilder:
    def __init__(self, *, run_config: DoudizhuRunConfig) -> None:
        self._run_config = run_config

    def build_payload(self, logical_results: Sequence[JobResult]) -> dict[str, object]:
        grouped: dict[str, list[JobResult]] = {}
        for result in logical_results:
            matchup_id = str(result.payload.get("matchup_id", result.job_id))
            grouped.setdefault(matchup_id, []).append(result)

        matchups: list[dict[str, object]] = []
        for matchup_id, grouped_results in sorted(grouped.items()):
            payloads = [result.payload for result in grouped_results]
            payloads_sorted = sorted(payloads, key = _role_sort_key)
            first = payloads_sorted[0]
            matchup_payload: dict[str, object] = {
                "matchup_id": matchup_id,
                "matchup_name": first.get("matchup_name", matchup_id),
                "mode": first.get("mode", "unknown"),
                "left_agent": first.get("left_agent"),
                "right_agent": first.get("right_agent"),
                "ok": all(bool(result.ok) for result in grouped_results),
                "role_results": [
                    {
                        "role_key": payload.get("role_key", result.job_id),
                        "role_label": payload.get("role_label"),
                        "left_view_role": payload.get("left_view_role"),
                        "agents": payload.get("agents"),
                        "hands_per_s": payload.get("hands_per_s"),
                        "warnings": payload.get("warnings", []),
                        **({"metrics": payload["metrics"]} if "metrics" in payload else {}),
                        **({"error": payload["error"]} if "error" in payload else {}),
                    }
                    for result, payload in sorted(
                        zip(grouped_results, payloads, strict = False),
                        key = lambda pair: _role_sort_key(pair[1]),
                    )
                ],
            }
            if str(first.get("mode", "")) == "head_to_head" and matchup_payload["ok"]:
                as_landlord, as_peasants = _find_left_view_role_payloads(payloads_sorted)
                matchup_payload["left_view_metrics"] = _left_view_metrics(
                    as_landlord = as_landlord,
                    as_peasants = as_peasants,
                    ci_multiplier = float(self._run_config.host_cfg.ci_multiplier),
                )
            if str(first.get("mode", "")) == "selfplay" and matchup_payload["ok"]:
                matchup_payload["role_profile_metrics"] = _role_profile_metrics(first)
            matchups.append(matchup_payload)

        return {
            "schema_version": 1,
            "scenario": "doudizhu",
            "runner": "evaluations_doudizhu",
            "mode": "selfplay" if len(self._run_config.unique_agents) == 1 else "head_to_head",
            "num_input_agents": int(len(self._run_config.input_agents)),
            "num_unique_agents": int(len(self._run_config.unique_agents)),
            "input_agents": list(self._run_config.input_agents),
            "unique_agents": [agent.display_name() for agent in self._run_config.unique_agents],
            "num_games_per_role": int(self._run_config.num_games),
            "workers": int(self._run_config.worker_layout.workers),
            "batch_size": int(self._run_config.batch_size),
            "usable_gpu_tokens": list(self._run_config.worker_layout.usable_gpu_tokens),
            "workers_per_gpu": (
                int(self._run_config.worker_layout.workers_per_gpu)
                if self._run_config.worker_layout.workers_per_gpu is not None
                else None
            ),
            "progress_interval": int(self._run_config.host_cfg.progress_interval),
            "ci_multiplier": float(self._run_config.host_cfg.ci_multiplier),
            "eval_data": str(self._run_config.eval_data),
            "matchups": matchups,
        }

    def render_report(self, logical_results: Sequence[JobResult]) -> str | None:
        grouped: dict[str, list[JobResult]] = {}
        for result in logical_results:
            grouped.setdefault(str(result.payload.get("matchup_id", result.job_id)), []).append(result)

        lines: list[str] = ["Note: self-play reports a role profile, not a strength comparison."]
        for matchup_id in sorted(grouped):
            results_sorted = sorted(grouped[matchup_id], key = lambda result: _role_sort_key(result.payload))
            payloads = [result.payload for result in results_sorted]
            first = payloads[0]
            lines.append(f"- {first.get('matchup_name', matchup_id)} ({first.get('mode', 'unknown')})")
            for result, payload in zip(results_sorted, payloads, strict = False):
                if not bool(result.ok):
                    label = str(payload.get("role_label", payload.get("role_key", result.job_id)))
                    lines.append(f"    - {label} failed: {payload.get('error', 'unknown error')}")
                    continue
                metrics = payload["metrics"]
                label = str(payload.get("role_label", payload.get("role_key", result.job_id)))
                lines.append(
                    "    - "
                    f"{label} "
                    f"LL WR {_metric_display(metrics['landlord_win_rate'])}, "
                    f"LL Avg Gain {_metric_display(metrics['landlord_avg_gain'])}"
                )
            if str(first.get("mode", "")) == "head_to_head" and all(bool(result.ok) for result in results_sorted):
                as_landlord, as_peasants = _find_left_view_role_payloads(payloads)
                left_view_metrics = _left_view_metrics(
                    as_landlord = as_landlord,
                    as_peasants = as_peasants,
                    ci_multiplier = float(self._run_config.host_cfg.ci_multiplier),
                )
                lines.append(
                    "    - Role Avg: "
                    f"Win Rate {_metric_display(left_view_metrics['role_avg_win_rate'])}, "
                    f"Avg Gain {_metric_display(left_view_metrics['role_avg_avg_gain'])}"
                )
        return "\n".join(lines) + "\n"


class DoudizhuWandbSink:
    def __init__(
        self,
        *,
        logdir: Path,
        run_config: DoudizhuRunConfig,
        planned_matchups: Sequence[SeatingPlan] = (),
    ) -> None:
        self._logdir = Path(logdir).resolve()
        self._run_config = run_config
        self._planned_matchups = self._dedupe_planned_matchups(planned_matchups)
        self._wandb_runs: dict[str, EvaluationWandbRun] = {}
        self._wandb_loggers: dict[str, _MatchupWandbLogger] = {}
        self._initialize_planned_runs()

    def record_event(self, event: QueueEvent) -> None:
        if not self._run_config.wandb.enabled or event.kind != "episode":
            return
        body = dict(event.body)
        matchup_id = str(body.get("matchup_id", "")).strip()
        if not matchup_id:
            return
        logger = self._require_wandb_logger(
            matchup_id = matchup_id,
            matchup_name = str(body.get("matchup_name", matchup_id)),
            mode = str(body.get("mode", "unknown")),
            left_agent = str(body.get("left_agent", "")),
            right_agent = str(body.get("right_agent", "")),
        )
        logger.record_episode(body)

    def write_matchup_summaries(self, *, matchups: Sequence[dict[str, object]], summary_path: Path) -> None:
        if not self._run_config.wandb.enabled:
            return
        events_path = self._logdir / "events.jsonl"
        for matchup in matchups:
            self._write_matchup_summary(
                matchup = matchup,
                summary_path = summary_path,
                events_path = events_path,
            )

    def close(self) -> None:
        for run in list(self._wandb_runs.values()):
            run.close()
        self._wandb_runs.clear()
        self._wandb_loggers.clear()

    @staticmethod
    def _dedupe_planned_matchups(planned_matchups: Sequence[SeatingPlan]) -> tuple[SeatingPlan, ...]:
        unique: list[SeatingPlan] = []
        seen_matchup_ids: set[str] = set()
        for plan in planned_matchups:
            matchup_id = str(plan.matchup_id)
            if matchup_id in seen_matchup_ids:
                continue
            seen_matchup_ids.add(matchup_id)
            unique.append(plan)
        return tuple(unique)

    def _initialize_planned_runs(self) -> None:
        if not self._run_config.wandb.enabled:
            return
        try:
            for plan in self._planned_matchups:
                _ = self._require_wandb_logger(
                    matchup_id = str(plan.matchup_id),
                    matchup_name = str(plan.matchup_name),
                    mode = str(plan.mode),
                    left_agent = str(plan.left_agent.display_name()),
                    right_agent = str(plan.right_agent.display_name()),
                )
        except Exception:
            self.close()
            raise

    def _require_wandb_logger(
        self,
        *,
        matchup_id: str,
        matchup_name: str,
        mode: str,
        left_agent: str,
        right_agent: str,
    ) -> _MatchupWandbLogger:
        logger = self._wandb_loggers.get(str(matchup_id))
        if logger is not None:
            return logger
        run = self._wandb_runs.get(str(matchup_id))
        if run is None:
            run = EvaluationWandbRun.open(
                project = self._run_config.wandb.project,
                entity = self._run_config.wandb.entity,
                name = self._build_wandb_run_name(matchup_name = matchup_name),
                config = {
                    "scenario": "doudizhu",
                    "matchup_id": str(matchup_id),
                    "matchup_name": str(matchup_name),
                    "mode": str(mode),
                    "left_agent": str(left_agent),
                    "right_agent": str(right_agent),
                    "num_games": int(self._run_config.num_games),
                    "batch_size": int(self._run_config.batch_size),
                    "workers": int(self._run_config.worker_layout.workers),
                    "workers_per_gpu": (
                        int(self._run_config.worker_layout.workers_per_gpu)
                        if self._run_config.worker_layout.workers_per_gpu is not None
                        else None
                    ),
                    "progress_interval": int(self._run_config.host_cfg.progress_interval),
                    "input_agents": list(self._run_config.input_agents),
                },
            )
            self._wandb_runs[str(matchup_id)] = run
        logger = _MatchupWandbLogger(
            run = run,
            mode = mode,
            ci_multiplier = float(self._run_config.host_cfg.ci_multiplier),
        )
        self._wandb_loggers[str(matchup_id)] = logger
        return logger

    def _build_wandb_run_name(self, *, matchup_name: str) -> str:
        if self._run_config.wandb.run_name is not None:
            return str(self._run_config.wandb.run_name)
        return str(matchup_name)

    def _write_matchup_summary(
        self,
        *,
        matchup: dict[str, object],
        summary_path: Path,
        events_path: Path,
    ) -> None:
        matchup_id = str(matchup["matchup_id"])
        _ = self._require_wandb_logger(
            matchup_id = matchup_id,
            matchup_name = str(matchup.get("matchup_name", matchup_id)),
            mode = str(matchup.get("mode", "unknown")),
            left_agent = str(matchup.get("left_agent", "")),
            right_agent = str(matchup.get("right_agent", "")),
        )
        run = self._wandb_runs[matchup_id]

        summary_payload: dict[str, object] = {
            "ok": bool(matchup.get("ok", False)),
            "matchup_name": str(matchup.get("matchup_name", matchup_id)),
            "left_agent": matchup.get("left_agent"),
            "right_agent": matchup.get("right_agent"),
        }
        table_rows: list[list[object]] = []
        role_results = list(matchup.get("role_results", []))
        for role_result in role_results:
            row = dict(role_result)
            metrics = row.get("metrics")
            if isinstance(metrics, dict):
                table_rows.append(
                    [
                        "role",
                        str(row.get("role_label", row.get("role_key", ""))),
                        str(row.get("left_view_role", "")),
                        _metric_display(dict(metrics["landlord_win_rate"])),
                        _metric_display(dict(metrics["landlord_avg_gain"])),
                        "",
                    ]
                )
            else:
                table_rows.append(
                    [
                        "role",
                        str(row.get("role_label", row.get("role_key", ""))),
                        str(row.get("left_view_role", "")),
                        "",
                        "",
                        str(row.get("error", "")),
                    ]
                )

        if bool(matchup.get("ok", False)):
            if "left_view_metrics" in matchup:
                left_view_metrics = dict(matchup["left_view_metrics"])
                summary_payload.update(_flatten_summary_metrics(left_view_metrics))
                table_rows.append(
                    [
                        "aggregate",
                        "Role average",
                        "left_view",
                        _metric_display(dict(left_view_metrics["role_avg_win_rate"])),
                        _metric_display(dict(left_view_metrics["role_avg_avg_gain"])),
                        "",
                    ]
                )
            if "role_profile_metrics" in matchup:
                role_profile_metrics = dict(matchup["role_profile_metrics"])
                summary_payload.update(_flatten_summary_metrics(role_profile_metrics))
                table_rows.append(
                    [
                        "aggregate",
                        "Role profile",
                        "selfplay",
                        _metric_display(dict(role_profile_metrics["as_landlord_win_rate"])),
                        _metric_display(dict(role_profile_metrics["as_landlord_avg_gain"])),
                        "",
                    ]
                )
        else:
            for role_result in role_results:
                if "error" in role_result:
                    summary_payload["error"] = str(role_result["error"])
                    break

        run.update_summary(summary_payload)
        run.log_table(
            "final_metrics",
            columns = ("row_kind", "label", "view", "win_rate", "avg_gain", "error"),
            rows = table_rows,
        )
        run.save_file(summary_path)
        if events_path.exists():
            run.save_file(events_path)


class DoudizhuSummaryWriter:
    def __init__(
        self,
        *,
        logdir: Path,
        run_config: DoudizhuRunConfig,
        planned_matchups: Sequence[SeatingPlan] = (),
    ) -> None:
        self._logdir = Path(logdir).resolve()
        self._merger = DoudizhuLogicalResultMerger(run_config = run_config)
        self._summary_builder = DoudizhuMatchupSummaryBuilder(run_config = run_config)
        self._wandb_sink = DoudizhuWandbSink(
            logdir = self._logdir,
            run_config = run_config,
            planned_matchups = planned_matchups,
        )
        self._event_formatter = _DoudizhuEventFormatter()
        self._pending_render: str | None = None

    def record_event(self, event: QueueEvent) -> None:
        self._wandb_sink.record_event(event)
        self._pending_render = self._event_formatter.format(event)

    def format_event(self, event: QueueEvent) -> str | None:
        del event
        rendered = self._pending_render
        self._pending_render = None
        return rendered

    def write_job(self, result: JobResult) -> Path:
        out_path = self._logdir / "jobs" / result.job_id / "summary.json"
        out_path.parent.mkdir(parents = True, exist_ok = True)
        out_path.write_text(json.dumps(result.payload, indent = 2) + "\n", encoding = "utf-8")
        return out_path

    def write_run(self, results: Sequence[JobResult]) -> Path:
        summary_path = self._logdir / "summary.json"
        logical_results = self._merger.merge(results)
        payload = self._summary_builder.build_payload(logical_results)
        summary_path.write_text(json.dumps(payload, indent = 2) + "\n", encoding = "utf-8")
        self._wandb_sink.write_matchup_summaries(matchups = list(payload["matchups"]), summary_path = summary_path)
        return summary_path

    def render_report(self, results: Sequence[JobResult]) -> str | None:
        logical_results = self._merger.merge(results)
        return self._summary_builder.render_report(logical_results)

    def close(self) -> None:
        self._wandb_sink.close()


__all__ = [
    "DoudizhuAggregator",
    "DoudizhuLogicalResultMerger",
    "DoudizhuMatchupSummaryBuilder",
    "DoudizhuResultBuilder",
    "DoudizhuSummaryWriter",
    "DoudizhuWandbSink",
]
