from __future__ import annotations

import collections
import math
import time
from typing import Sequence

from evaluations.core.dispatch import SharedUnitAllocator
from evaluations.core.progress import build_normalized_progress, build_progress_metric
from evaluations.core.socket_worker import PolicyWorkerRequest
from evaluations.core.types import (
    DecisionBatch,
    DecisionRequest,
    EpisodeUpdate,
    Finished,
    HostMessage,
    ProgressUpdate,
    StatusUpdate,
)
from evaluations.doudizhu.runtime.config import DoudizhuHostConfig
from evaluations.doudizhu.runtime.game import DoudizhuGame
from evaluations.doudizhu.specs import POSITIONS, SeatingPlan
from marvel.runtime.intervals import IntervalTimer


class DoudizhuGameBatch:
    def __init__(
        self,
        *,
        host_cfg: DoudizhuHostConfig,
        card_play_data: Sequence[dict[str, object]],
        work_allocator: SharedUnitAllocator,
        game_factory = None,
    ) -> None:
        self._host_cfg = host_cfg
        self._card_play_data = [dict(item) for item in card_play_data]
        self._work_allocator = work_allocator
        self._game_factory = DoudizhuGame if game_factory is None else game_factory
        self._active: dict[int, DoudizhuGame] = {}
        self._total_games = 0

    def open(self, *, num_games: int) -> None:
        total_available_games = min(int(num_games), len(self._card_play_data))
        if total_available_games <= 0:
            raise ValueError("No games to evaluate.")
        self._total_games = int(total_available_games)
        self._active.clear()
        self._fill_active_games()

    @property
    def total_games(self) -> int:
        return int(self._total_games)

    def has_active_games(self) -> bool:
        return bool(self._active)

    def snapshot_decisions(self) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
        seat_env_ids: dict[str, list[int]] = {seat: [] for seat in POSITIONS}
        seat_states: dict[str, list[str]] = {seat: [] for seat in POSITIONS}
        for env_id, game in self._active.items():
            seat = game.current_player()
            seat_env_ids[seat].append(int(env_id))
            seat_states[seat].append(game.current_infoset(strict_turn = True).to_string())
        return seat_env_ids, seat_states

    def apply_action(self, *, env_id: int, action: str) -> tuple[bool, int | None, float | None]:
        game = self._active.get(int(env_id))
        if game is None:
            raise RuntimeError(f"Missing active env_id={env_id} while applying actions.")
        game.step(str(action))
        if not game.is_terminal():
            return False, None, None

        score = float(game.landlord_score())
        landlord_win = int(game.env.player_utility_dict["landlord"] > 0)
        self._active.pop(int(env_id), None)
        self._fill_active_games()
        return True, int(landlord_win), float(score)

    def close(self) -> None:
        self._active.clear()
        self._total_games = 0

    def _fill_active_games(self) -> None:
        max_envs = max(1, min(int(self._host_cfg.max_envs), int(self._total_games)))
        while len(self._active) < max_envs:
            claimed_game_ids = self._work_allocator.claim_unit_ids(max_count = int(max_envs - len(self._active)))
            if not claimed_game_ids:
                return
            for game_idx in claimed_game_ids:
                if int(game_idx) < 1 or int(game_idx) > int(self._total_games):
                    raise RuntimeError(
                        f"Claimed Dou Dizhu game id {game_idx} outside available range 1..{self._total_games}."
                    )
                data_idx = int(game_idx) - 1
                self._active[int(game_idx)] = self._game_factory(
                    {k: list(v) if isinstance(v, list) else v for k, v in self._card_play_data[data_idx].items()}
                )


class DoudizhuProgressTracker:
    def __init__(
        self,
        *,
        job_id: str,
        logical_job_id: str,
        plan: SeatingPlan,
        host_cfg: DoudizhuHostConfig,
        total_games: int,
    ) -> None:
        self._job_id = str(job_id)
        self._logical_job_id = str(logical_job_id)
        self._plan = plan
        self._host_cfg = host_cfg
        self._total_games = int(total_games)
        self._total_processed = 0
        self._num_landlord_wins = 0
        self._sum_landlord_scores = 0.0
        self._sum_sq_landlord_scores = 0.0
        self._last_progress_completed = 0
        self._start_time = 0.0
        self._progress_timer = IntervalTimer(host_cfg.progress_interval, total_steps = total_games)

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def record_completed_game(self, *, landlord_win: int, score: float) -> None:
        self._total_processed += 1
        self._num_landlord_wins += int(landlord_win)
        self._sum_landlord_scores += float(score)
        self._sum_sq_landlord_scores += float(score) * float(score)

    def build_progress_update(self) -> dict[str, object] | None:
        if not self._progress_timer.triggered(
            self._total_processed,
            previous_step = self._last_progress_completed,
        ):
            return None
        self._last_progress_completed = int(self._total_processed)
        return self._build_progress_payload()

    def plan_metadata(self) -> dict[str, object]:
        return {
            "matchup_id": self._plan.matchup_id,
            "matchup_name": self._plan.matchup_name,
            "mode": self._plan.mode,
            "role_key": self._plan.role_key(),
            "role_label": self._plan.role_label(),
            "left_agent": self._plan.left_agent.display_name(),
            "right_agent": self._plan.right_agent.display_name(),
            "left_view_role": self._plan.left_view_role(),
        }

    def opened_message(self) -> str:
        return f"Opened Dou Dizhu role pass with {self._plan.role_label(compact = True)}."

    def _build_progress_payload(self) -> dict[str, object]:
        elapsed = time.perf_counter() - self._start_time
        hands_per_s = self._total_processed / max(1e-6, elapsed)

        wp_landlord = self._num_landlord_wins / float(self._total_processed)
        wp_stderr = math.sqrt(wp_landlord * (1.0 - wp_landlord) / float(self._total_processed))

        mean_landlord = self._sum_landlord_scores / float(self._total_processed)
        var_landlord = (self._sum_sq_landlord_scores / float(self._total_processed)) - mean_landlord**2
        var_landlord = max(var_landlord, 0.0)
        stderr_landlord = math.sqrt(var_landlord / float(self._total_processed))

        return build_normalized_progress(
            scope_kind = "physical_job",
            scope_label = self._plan.progress_scope_label(),
            completed = int(self._total_processed),
            total = int(self._total_games),
            unit = "games",
            metrics = (
                build_progress_metric(
                    key = "landlord_win_rate",
                    label = "Landlord Win Rate",
                    value = float(wp_landlord),
                    ci = float(float(self._host_cfg.ci_multiplier) * wp_stderr),
                    precision = 3,
                ),
                build_progress_metric(
                    key = "landlord_avg_gain",
                    label = "Landlord Avg Gain",
                    value = float(mean_landlord),
                    ci = float(float(self._host_cfg.ci_multiplier) * stderr_landlord),
                    precision = 3,
                ),
            ),
            throughput_label = "Hands/s",
            throughput_value = float(hands_per_s),
            throughput_precision = 2,
            job_id = self._job_id,
            logical_job_id = self._logical_job_id,
            extra = {
                "landlord_win_rate": float(wp_landlord),
                "landlord_win_rate_ci": float(float(self._host_cfg.ci_multiplier) * wp_stderr),
                "landlord_avg_gain": float(mean_landlord),
                "landlord_avg_gain_ci": float(float(self._host_cfg.ci_multiplier) * stderr_landlord),
                "hands_per_s": float(hands_per_s),
                "final": bool(self._total_processed >= self._total_games),
                **self.plan_metadata(),
            },
        )


class DoudizhuHostSession:
    def __init__(
        self,
        *,
        job_id: str,
        logical_job_id: str,
        plan: SeatingPlan,
        card_play_data: Sequence[dict[str, object]],
        num_games: int,
        host_cfg: DoudizhuHostConfig,
        work_allocator: SharedUnitAllocator,
    ) -> None:
        self._job_id = str(job_id)
        self._logical_job_id = str(logical_job_id)
        self._plan = plan
        self._card_play_data = [dict(item) for item in card_play_data]
        self._num_games = int(num_games)
        self._host_cfg = host_cfg
        self._work_allocator = work_allocator

        self._messages: collections.deque[HostMessage] = collections.deque()
        self._batch = DoudizhuGameBatch(
            host_cfg = host_cfg,
            card_play_data = self._card_play_data,
            work_allocator = self._work_allocator,
        )
        self._progress: DoudizhuProgressTracker | None = None
        self._batch_counter = 0
        self._inflight_batch_id: str | None = None
        self._inflight_env_ids: tuple[int, ...] = ()
        self._inflight_controller_id: str | None = None
        self._done = False

    def open(self) -> None:
        self._batch.open(num_games = self._num_games)
        self._progress = DoudizhuProgressTracker(
            job_id = self._job_id,
            logical_job_id = self._logical_job_id,
            plan = self._plan,
            host_cfg = self._host_cfg,
            total_games = self._batch.total_games,
        )
        self._progress.start()
        self._messages.append(
            StatusUpdate(
                {
                    "stage": "host_opened",
                    "message": self._progress.opened_message(),
                    "logical_job_id": self._logical_job_id,
                    "scope_label": self._plan.progress_scope_label(),
                    **self._progress.plan_metadata(),
                }
            )
        )

    def next_message(self) -> HostMessage:
        if self._messages:
            return self._messages.popleft()
        if self._done:
            return Finished()
        if self._inflight_batch_id is not None:
            raise RuntimeError("HostSession cannot issue a new decision batch before submit_actions().")

        seat_env_ids, seat_states = self._batch.snapshot_decisions()
        for seat in POSITIONS:
            env_ids = seat_env_ids[seat]
            if not env_ids:
                continue
            requests = tuple(
                DecisionRequest(
                    controller_id = seat,
                    acting_entity_id = seat,
                    episode_id = str(env_id),
                    decision_index = 0,
                    payload = PolicyWorkerRequest(
                        request_id = int(env_id),
                        infoset_str = infoset_str,
                    ),
                )
                for env_id, infoset_str in zip(env_ids, seat_states[seat], strict = False)
            )
            self._batch_counter += 1
            self._inflight_batch_id = f"{self._job_id}:{seat}:{self._batch_counter}"
            self._inflight_env_ids = tuple(int(env_id) for env_id in env_ids)
            self._inflight_controller_id = seat
            return DecisionBatch(
                batch_id = self._inflight_batch_id,
                controller_id = seat,
                requests = requests,
            )

        if self._batch.has_active_games():
            raise RuntimeError("Active games remain but no acting player batch could be formed.")
        self._done = True
        return Finished()

    def submit_actions(self, batch_id: str, actions) -> None:
        if self._inflight_batch_id != str(batch_id):
            raise RuntimeError(
                f"submit_actions batch_id mismatch: expected {self._inflight_batch_id!r}, got {batch_id!r}."
            )
        if len(actions) != len(self._inflight_env_ids):
            raise RuntimeError(
                "submit_actions count mismatch: "
                f"expected {len(self._inflight_env_ids)} got {len(actions)}."
            )
        if self._progress is None:
            raise RuntimeError("HostSession must be opened before submit_actions().")

        for env_id, action in zip(self._inflight_env_ids, actions, strict = False):
            finished, landlord_win, score = self._batch.apply_action(env_id = int(env_id), action = str(action))
            if not finished:
                continue
            if landlord_win is None or score is None:
                raise RuntimeError("Finished Dou Dizhu game must provide landlord outcome and score.")
            self._progress.record_completed_game(landlord_win = int(landlord_win), score = float(score))
            self._messages.append(
                EpisodeUpdate(
                    {
                        "episode_id": str(env_id),
                        "landlord_win": int(landlord_win),
                        "landlord_score": float(score),
                        "logical_job_id": self._logical_job_id,
                        "scope_label": self._plan.progress_scope_label(),
                        **self._progress.plan_metadata(),
                    }
                )
            )

        self._inflight_batch_id = None
        self._inflight_env_ids = ()
        self._inflight_controller_id = None

        progress_payload = self._progress.build_progress_update()
        if progress_payload is not None:
            self._messages.append(ProgressUpdate(progress_payload))
        if not self._batch.has_active_games():
            self._done = True

    def close(self) -> None:
        self._batch.close()
        self._messages.clear()
        self._progress = None
        self._inflight_batch_id = None
        self._inflight_env_ids = ()
        self._inflight_controller_id = None


__all__ = ["DoudizhuHostSession"]
