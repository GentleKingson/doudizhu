from __future__ import annotations

import argparse
import sys
from typing import Sequence

from evaluations.core import EvaluationCoordinator
from evaluations.doudizhu.scenario import DoudizhuScenario


def _registered_scenarios() -> tuple[object, ...]:
    return (DoudizhuScenario(),)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Run evaluation scenarios.")
    subparsers = parser.add_subparsers(dest = "scenario", required = True)
    for scenario in _registered_scenarios():
        scenario_parser = subparsers.add_parser(
            scenario.name,
            help = f"Run {scenario.name} evaluations.",
        )
        scenario.configure_parser(scenario_parser)
        scenario_parser.set_defaults(_scenario_impl = scenario)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    scenario = getattr(args, "_scenario_impl", None)
    if scenario is None:
        parser.error("Missing scenario implementation.")

    plan = scenario.plan_run(args)
    if len(plan.logical_jobs) > 1 or len(plan.slots) > 1:
        print(
            f"[Info] Planned {len(plan.logical_jobs)} logical jobs across {len(plan.slots)} worker slots.",
            file = sys.stderr,
        )
    coordinator = EvaluationCoordinator(plan = plan)
    try:
        _ = coordinator.run()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
