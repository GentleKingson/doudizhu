from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Literal, Sequence

import nanopokerlib as npl
import torch
from kagami.core.common import validate_temperature
from kagami.games import DoudizhuInferenceEngine, DoudizhuInferenceEngineConfig
from kagami.search.routines import (
    SEARCH_ROUTINE_LOCAL_SUBGAME,
    default_local_search_floor,
    normalize_search_routine,
    validate_search_floor,
)
from marvel.config import deserialize_train_config
from marvel.runtime.checkpoints import CheckpointStore

from evaluations.core.socket_worker import PolicyWorkerRequest, run_policy_worker_server
from evaluations.doudizhu.runtime.config import DEFAULT_MARVEL_AGENT, DEFAULT_NETWORK_SERVER

_SEARCH_ROUTINE_CHOICES = (
    "update-equiv",
    DEFAULT_MARVEL_AGENT.search.routine,
)


def _normalize_checkpoint_locator(locator: str) -> str:
    text = str(locator).strip()
    if not text:
        raise ValueError("checkpoint locator must be non-empty.")
    if text.startswith("wandb:"):
        return text
    path = Path(text)
    if not path.is_absolute():
        path = path.resolve()
    return str(path)


def _validate_non_negative_float(value: float, *, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _validate_search_budget(value: int, *, name: str = "search_budget") -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _load_checkpoint_game_module(*, checkpoint: str) -> object | None:
    checkpoint_path = CheckpointStore.resolve_checkpoint_file_path(checkpoint)
    if not checkpoint_path.exists():
        return None

    payload = CheckpointStore.load_checkpoint_payload_file(
        checkpoint_path,
        device = torch.device("cpu"),
    )
    train_config = deserialize_train_config(payload.get("train_config"))
    if train_config is None:
        return None
    game_name = str(train_config.game_name).strip()
    if game_name.startswith("nanopokerlib."):
        game_name = game_name[len("nanopokerlib.") :]
    parts = [part for part in game_name.split(".") if part]
    if not parts:
        raise ValueError("Checkpoint train_config.game_name must be non-empty.")
    resolved_game: object = npl
    for part in parts:
        next_game = getattr(resolved_game, part, None)
        if next_game is None:
            raise ValueError(f"Unknown checkpoint game '{train_config.game_name}' in nanopokerlib.")
        resolved_game = next_game
    return resolved_game


def _policy_factory(
    *,
    checkpoint: str,
    mode: Literal["direct", "search"],
    belief_checkpoint: str | None = None,
    device: str = DEFAULT_MARVEL_AGENT.device,
    action_temp: float = DEFAULT_MARVEL_AGENT.action_temp,
    belief_temp: float = DEFAULT_MARVEL_AGENT.belief_temp,
    search_temp: float = DEFAULT_MARVEL_AGENT.search_temp,
    routine: str = DEFAULT_MARVEL_AGENT.search.routine,
    search_beta: float | None = None,
    search_budget: int | None = None,
    search_floor: int | None = None,
):
    resolved_device = str(device).strip() or DEFAULT_MARVEL_AGENT.device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"

    device_obj = torch.device(resolved_device)
    validate_temperature(action_temp, name = "action_temp")
    validate_temperature(belief_temp, name = "belief_temp")
    validate_temperature(search_temp, name = "search_temp")

    engine_cfg = DoudizhuInferenceEngineConfig(
        belief_temp = float(belief_temp),
        search_temp = float(search_temp),
        action_temp = float(action_temp),
    )
    if mode == "search":
        if belief_checkpoint is None:
            raise ValueError("belief_checkpoint is required when mode='search'.")
        resolved_routine = normalize_search_routine(routine)
        resolved_search_beta = (
            float(DEFAULT_MARVEL_AGENT.search.search_beta)
            if search_beta is None
            else float(search_beta)
        )
        _validate_non_negative_float(resolved_search_beta, name = "search_beta")
        rollout_budget = (
            int(DEFAULT_MARVEL_AGENT.search.rollout_budget)
            if search_budget is None
            else int(search_budget)
        )
        _validate_search_budget(rollout_budget)
        if search_floor is not None:
            resolved_search_floor = int(search_floor)
        elif resolved_routine == SEARCH_ROUTINE_LOCAL_SUBGAME:
            resolved_search_floor = default_local_search_floor(rollout_budget)
        else:
            resolved_search_floor = None
        validate_search_floor(resolved_search_floor, search_budget = int(rollout_budget))
        if resolved_search_floor is not None and resolved_routine != SEARCH_ROUTINE_LOCAL_SUBGAME:
            raise ValueError("search_floor is only valid when routine='local-subgame'.")
        engine_cfg = DoudizhuInferenceEngineConfig(
            belief_temp = float(belief_temp),
            search_temp = float(search_temp),
            action_temp = float(action_temp),
            update_equivalent = DoudizhuInferenceEngineConfig.UpdateEquivalent(
                enabled = True,
                routine = resolved_routine,
                rollout_budget = int(rollout_budget),
                search_floor = resolved_search_floor,
                search_beta = float(resolved_search_beta),
                seed = int(DEFAULT_MARVEL_AGENT.search.seed),
            ),
        )
    elif search_beta is not None:
        raise ValueError("search_beta is only valid when mode='search'.")
    elif search_budget is not None:
        raise ValueError("search_budget is only valid when mode='search'.")
    elif search_floor is not None:
        raise ValueError("search_floor is only valid when mode='search'.")
    elif normalize_search_routine(routine) != str(DEFAULT_MARVEL_AGENT.search.routine):
        raise ValueError("routine is only valid when mode='search'.")
    checkpoint_game = _load_checkpoint_game_module(checkpoint = checkpoint)
    if checkpoint_game is None:
        engine = DoudizhuInferenceEngine(cfg = engine_cfg)
    else:
        engine = DoudizhuInferenceEngine(cfg = engine_cfg, game = checkpoint_game)
    engine.bind_marvel_model(
        checkpoint = checkpoint,
        device = device_obj,
    )
    if mode == "search" and normalize_search_routine(routine) == SEARCH_ROUTINE_LOCAL_SUBGAME:
        engine.bind_marvel_critic(
            checkpoint = checkpoint,
            device = device_obj,
        )
    if belief_checkpoint is not None:
        engine.bind_yomi_model(
            checkpoint = belief_checkpoint,
            device = device_obj,
        )

    def policy(requests: Sequence[PolicyWorkerRequest]) -> List[str]:
        infoset_strs = [req.infoset_str for req in requests]
        return [str(action) for action in engine.act_batch(infoset_strs = infoset_strs)]

    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description = "Run a Kagami-backed MARVEL DouDiZhu socket agent.")
    parser.add_argument("socket_path", type = Path)
    parser.add_argument("checkpoint", type = str, help = "Policy checkpoint path or wandb: locator.")
    parser.add_argument("--mode", choices = ("direct", "search"), default = "direct")
    parser.add_argument(
        "--belief-checkpoint",
        dest = "belief_checkpoint",
        type = str,
        default = None,
        help = "Required for --mode search. Accepts a local path or wandb: locator.",
    )
    parser.add_argument("--device", type = str, default = DEFAULT_MARVEL_AGENT.device)
    parser.add_argument("--action-temp", type = float, default = float(DEFAULT_MARVEL_AGENT.action_temp))
    parser.add_argument("--belief-temp", type = float, default = float(DEFAULT_MARVEL_AGENT.belief_temp))
    parser.add_argument("--search-temp", type = float, default = float(DEFAULT_MARVEL_AGENT.search_temp))
    parser.add_argument("--routine", choices = _SEARCH_ROUTINE_CHOICES, default = DEFAULT_MARVEL_AGENT.search.routine)
    parser.add_argument("--search-beta", type = float, default = None)
    parser.add_argument("--search-budget", type = int, default = None)
    parser.add_argument("--search-floor", type = int, default = None)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))

    try:
        validate_temperature(float(args.action_temp), name = "action_temp")
        validate_temperature(float(args.belief_temp), name = "belief_temp")
        validate_temperature(float(args.search_temp), name = "search_temp")
        if args.search_beta is not None:
            _validate_non_negative_float(float(args.search_beta), name = "search_beta")
        if args.search_budget is not None:
            _validate_search_budget(int(args.search_budget))
        if args.search_floor is not None:
            budget = int(args.search_budget) if args.search_budget is not None else int(DEFAULT_MARVEL_AGENT.search.rollout_budget)
            validate_search_floor(int(args.search_floor), search_budget = budget)
    except ValueError as exc:
        parser.error(str(exc))

    socket_path = Path(args.socket_path)
    checkpoint = _normalize_checkpoint_locator(args.checkpoint)
    if not checkpoint.startswith("wandb:") and not Path(checkpoint).exists():
        print(f"Checkpoint not found: {checkpoint}", file = sys.stderr)
        return 2

    belief_checkpoint = None
    if args.belief_checkpoint is not None:
        belief_checkpoint = _normalize_checkpoint_locator(args.belief_checkpoint)
        if not belief_checkpoint.startswith("wandb:") and not Path(belief_checkpoint).exists():
            print(f"Belief checkpoint not found: {belief_checkpoint}", file = sys.stderr)
            return 2
    if str(args.mode) == "search" and belief_checkpoint is None:
        parser.error("--belief-checkpoint is required when --mode search is selected.")
    if str(args.mode) != "search" and args.search_beta is not None:
        parser.error("--search-beta is only valid when --mode search is selected.")
    if str(args.mode) != "search" and args.search_budget is not None:
        parser.error("--search-budget is only valid when --mode search is selected.")
    if str(args.mode) != "search" and args.search_floor is not None:
        parser.error("--search-floor is only valid when --mode search is selected.")
    if (
        str(args.mode) == "search"
        and args.search_floor is not None
        and normalize_search_routine(str(args.routine)) != SEARCH_ROUTINE_LOCAL_SUBGAME
    ):
        parser.error("--search-floor is only valid with --routine local-subgame.")
    if (
        str(args.mode) != "search"
        and normalize_search_routine(str(args.routine)) != str(DEFAULT_MARVEL_AGENT.search.routine)
    ):
        parser.error("--routine is only valid when --mode search is selected.")

    policy = _policy_factory(
        checkpoint = checkpoint,
        mode = str(args.mode),
        belief_checkpoint = belief_checkpoint,
        device = str(args.device),
        action_temp = float(args.action_temp),
        belief_temp = float(args.belief_temp),
        search_temp = float(args.search_temp),
        routine = str(args.routine),
        search_beta = float(args.search_beta) if args.search_beta is not None else None,
        search_budget = int(args.search_budget) if args.search_budget is not None else None,
        search_floor = int(args.search_floor) if args.search_floor is not None else None,
    )
    run_policy_worker_server(
        socket_path,
        policy = policy,
        cfg = DEFAULT_NETWORK_SERVER,
        server_label = "Dou Dizhu policy worker",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
