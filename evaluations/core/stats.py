"""Shared statistics primitives for evaluation scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen = True)
class MetricStats:
    n: int
    mean: float
    std: float
    sem: float
    ci: float
    unit: str = ""


class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return float("nan")
        return self.m2 / float(self.n - 1)

    @property
    def std(self) -> float:
        variance = self.variance
        return math.sqrt(variance) if math.isfinite(variance) else float("nan")

    @property
    def sem(self) -> float:
        if self.n < 2:
            return float("nan")
        return self.std / math.sqrt(float(self.n))


def _validate_ci_multiplier(ci_multiplier: float) -> None:
    if float(ci_multiplier) != 1.0:
        raise ValueError("ci_multiplier must be exactly 1.0.")


def metric_from_mean_ci(
    *,
    n: int,
    mean: float,
    ci: float,
    ci_multiplier: float,
    unit: str = "",
) -> MetricStats:
    n_value = int(n)
    if n_value <= 0:
        raise ValueError("n must be positive.")
    _validate_ci_multiplier(ci_multiplier)
    ci_mult = float(ci_multiplier)
    sem = float(ci) / ci_mult
    std = sem * math.sqrt(float(n_value))
    return MetricStats(
        n = n_value,
        mean = float(mean),
        std = float(std),
        sem = float(sem),
        ci = float(ci),
        unit = str(unit),
    )


def metric_from_running_stats(
    stats: RunningStats,
    *,
    ci_multiplier: float,
    unit: str = "",
) -> MetricStats:
    _validate_ci_multiplier(ci_multiplier)
    ci_mult = float(ci_multiplier)
    sem = float(stats.sem)
    ci = float(sem * ci_mult) if math.isfinite(sem) else float("nan")
    return MetricStats(
        n = int(stats.n),
        mean = float(stats.mean) if int(stats.n) > 0 else float("nan"),
        std = float(stats.std),
        sem = sem,
        ci = ci,
        unit = str(unit),
    )


__all__ = ["MetricStats", "RunningStats", "metric_from_mean_ci", "metric_from_running_stats"]
