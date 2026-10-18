from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from bisect import bisect_left
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from marvel.buffer import trajectory_field_specs
from marvel.config import TrainConfig


_NO_SCOPE = nullcontext()
_REQUIRED_SCOPES = {
    "vrpo.iteration", "vrpo.rollout.collect", "vrpo.rollout.metrics",
    "vrpo.bootstrap.critic_forward", "vrpo.replay.add",
    "vrpo.policy.batch_prepare", "vrpo.policy.forward", "vrpo.policy.gae_loss",
    "vrpo.policy.backward", "vrpo.policy.grad_clip", "vrpo.policy.optimizer",
    "vrpo.policy.metrics",
    "vrpo.critic.batch_prepare", "vrpo.critic.agent_forward", "vrpo.critic.forward",
    "vrpo.critic.gae_loss", "vrpo.critic.backward", "vrpo.critic.grad_clip",
    "vrpo.critic.optimizer", "vrpo.critic.metrics",
    "vrpo.critic.replay_sample", "vrpo.ema", "vrpo.log_and_save",
    "sampler.cache_rebuild", "sampler.agent_forward", "sampler.action_select",
    "sampler.actions_d2h", "sampler.env_step", "sampler.obs_h2d",
    "sampler.buffer_commit", "sampler.collect_clone_consume",
}
_IPPO_REQUIRED_SCOPES = {
    "ippo.iteration", "ippo.rollout.collect", "ippo.rollout.metrics",
    "ippo.bootstrap.agent_forward", "ippo.bootstrap.gae",
    "ippo.policy.batch_prepare", "ippo.policy.forward", "ippo.policy.loss",
    "ippo.policy.backward", "ippo.policy.grad_clip", "ippo.policy.optimizer",
    "ippo.policy.metrics", "ippo.ema", "ippo.log_and_save",
} | {name for name in _REQUIRED_SCOPES if name.startswith("sampler.")}


def diagnostic_scope(enabled: bool, name: str):
    return torch.profiler.record_function(name) if enabled else _NO_SCOPE


def validate_trace(path: Path, algo_name: str = "vrpo") -> dict:
    return _validate_events(json.loads(path.read_text())["traceEvents"], algo_name)


def _validate_events(events: list[dict], algo_name: str = "vrpo") -> dict:
    if algo_name not in {"ippo", "vrpo"}:
        raise ValueError(f"Unsupported diagnostic algorithm: {algo_name}")
    scopes = {e.get("name") for e in events if e.get("cat") == "user_annotation"}
    cpu_events = sum(e.get("cat") == "cpu_op" for e in events)
    cuda_kernels = sum(e.get("cat") == "kernel" for e in events)
    compilation_events = sum(
        any(marker in e.get("name", "") for marker in (
            "_compile.compile_inner", "compile_fx", "GraphLowering.compile",
        ))
        for e in events
    )
    required = _IPPO_REQUIRED_SCOPES if algo_name == "ippo" else _REQUIRED_SCOPES
    missing = sorted(required - scopes)
    return {
        "status": "PROFILE_VALID" if cpu_events and cuda_kernels and not missing and not compilation_events else "PROFILE_INVALID",
        "cpu_events": cpu_events,
        "cuda_kernel_events": cuda_kernels,
        "missing_scopes": missing,
        "compilation_events": compilation_events,
    }


def trace_phase_summary(events: list[dict]) -> tuple[list[dict], int]:
    launches = {
        e["args"]["correlation"]: e["ts"] for e in events
        if e.get("cat") in {"cuda_runtime", "cuda_driver"} and "correlation" in e.get("args", {})
    }
    kernels = [e for e in events if e.get("cat") == "kernel"]
    correlated = sorted(
        (launches[e["args"]["correlation"]], e["dur"]) for e in kernels
        if e.get("args", {}).get("correlation") in launches
    )
    launch_times = [t for t, _ in correlated]
    cumulative_us = [0.0]
    for _, duration in correlated:
        cumulative_us.append(cumulative_us[-1] + duration)
    phases: dict[str, dict] = {}
    for event in events:
        name = event.get("name", "")
        if event.get("cat") != "user_annotation" or not name.startswith(("ippo.", "vrpo.", "sampler.")):
            continue
        phase = phases.setdefault(name, dict(phase = name, cpu_inclusive_ms = 0.0, cuda_kernel_ms = 0.0, count = 0))
        start = bisect_left(launch_times, event["ts"])
        end = bisect_left(launch_times, event["ts"] + event["dur"])
        # Autograd workers launch outside the main thread's scope tree; join by CUDA correlation and launch time.
        phase["cuda_kernel_ms"] += (cumulative_us[end] - cumulative_us[start]) / 1000
        phase["cpu_inclusive_ms"] += event["dur"] / 1000
        phase["count"] += 1
    rows = [dict(**p, avg_cpu_per_call_ms = p["cpu_inclusive_ms"] / p["count"],
                 avg_cuda_per_call_ms = p["cuda_kernel_ms"] / p["count"],
                 iteration_wall_share = "", evidence = "trace.json (CUDA launch correlation)") for p in phases.values()]
    return rows, len(kernels) - len(correlated)


class VRPODiagnostics:
    """Opt-in IPPO/VRPO wall measurements and one CPU/CUDA diagnostic iteration."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.enabled = cfg.benchmark or cfg.profile
        self.profiler = None
        self.sampler = None
        self.rows: list[dict] = []
        self.validation: dict | None = None
        self.iterations_seen = 0
        self.block_start = None

    def __enter__(self):
        if self.cfg.benchmark_contiguous and (not self.cfg.benchmark or self.cfg.profile):
            raise ValueError("benchmark_contiguous requires benchmark=True and profile=False.")
        if not self.enabled:
            return self
        if self.cfg.algo_name not in {"ippo", "vrpo"}:
            raise ValueError("Training diagnostics require IPPO or VRPO.")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Training diagnostics currently require a single process/GPU.")
        if self.cfg.profile_warmup_iterations < 0:
            raise ValueError("profile_warmup_iterations must be non-negative.")
        if self.cfg.total_iterations < self.cfg.profile_warmup_iterations + 3:
            raise ValueError("Diagnostics require warm-up iterations plus at least 3 full iterations.")
        if self.cfg.benchmark_contiguous:
            self.run_start = time.perf_counter()
        self.device = torch.device(self.cfg.device)
        self.directory = Path(self.cfg.profile_trace_dir or str(Path(self.cfg.logging_dir) / "profiles"))
        self.directory.mkdir(parents = True, exist_ok = False)
        self._write_snapshot()
        if self.cfg.profile and (
            self.device.type != "cuda" or not torch.cuda.is_available()
            or not {torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA}
            <= torch.profiler.supported_activities()
        ):
            (self.directory / "findings.md").write_text("INCONCLUSIVE: profiling requires CUDA hardware and both CPU/CUDA activities.\n")
            raise RuntimeError("Profiling requires CUDA hardware and both CPU/CUDA activities.")
        filename = "wall_contiguous.csv" if self.cfg.benchmark_contiguous else (
            "profile_wall.csv" if self.cfg.profile else "wall_baseline.csv"
        )
        self.wall_file = (self.directory / filename).open("x", newline = "")
        self.wall_writer = None
        return self

    def _write_snapshot(self) -> None:
        root = Path(__file__).resolve().parents[2]
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd = root, capture_output = True, text = True)
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output = True, text = True,
        ) if torch.cuda.is_available() else None
        specs = trajectory_field_specs(self.cfg.game)
        tensors = [factory(1, torch.device("meta")) for factory in specs.values()]
        bytes_per_game = 8 + sum(t.numel() * t.element_size() for t in tensors)
        snapshot = {
            "config": asdict(self.cfg), "git_sha": git_sha.stdout.strip(),
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "marvel").rglob("*.py"))},
            "python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "hostname": platform.node(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "driver_gpus": driver.stdout.strip() if driver else None,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" and torch.cuda.is_available() else None,
            "profiler_activities": sorted(a.name for a in torch.profiler.supported_activities()),
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "autocast_cuda": torch.is_autocast_enabled("cuda"),
            "rank": int(os.environ.get("RANK", "0")), "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "environment": {k: os.environ.get(k) for k in (
                "WANDB_DISABLED", "MARVEL_METRICS_CSV", "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING",
                "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
                "TORCH_LOGS", "TORCH_TRACE", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
            )},
            "buffer_bytes": {
                "per_game": bytes_per_game,
                "rollout": bytes_per_game * self.cfg.rollout_batch_size,
                "commit": bytes_per_game * (self.cfg.rollout_batch_size + self.cfg.collect_batch_size),
                "replay": bytes_per_game * self.cfg.collect_batch_size * self.cfg.replay_buffer_scale,
            },
        }
        (self.directory / "config_snapshot.json").write_text(json.dumps(snapshot, indent = 2) + "\n")

    def scope(self, name: str):
        return diagnostic_scope(self.profiler is not None, name)

    @contextmanager
    def iteration(self, iteration: int, sampler):
        if not self.enabled:
            yield
            return
        self.sampler = sampler
        if self.cfg.benchmark_contiguous:
            with self._contiguous_iteration(iteration, sampler):
                yield
            return
        position = len(self.rows)
        if self.cfg.profile and position == self.cfg.profile_warmup_iterations:
            self.profiler = torch.profiler.profile(
                activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule = torch.profiler.schedule(wait = 1, warmup = 1, active = 1, repeat = 1),
                on_trace_ready = self._export_trace,
                record_shapes = False, profile_memory = False, with_stack = False,
            )
            self.profiler.start()
        sampler.profile = self.profiler is not None
        action = self.profiler.current_action.name if self.profiler else "OFF"
        env_before, steps_before, games_before, pending_before = (
            sampler.total_env_steps, sampler.total_game_steps,
            sampler.total_completed_games, sampler.commit_buffer_size,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        with self.scope(f"{self.cfg.algo_name}.iteration"):
            yield
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        games = sampler.total_completed_games - games_before
        row = {
            "iteration": iteration,
            "phase": "warmup" if position < self.cfg.profile_warmup_iterations else "measurement",
            "profiler_action": action, "iteration_wall_ms": elapsed * 1000,
            "collect_batch_size": games + pending_before - sampler.commit_buffer_size,
            "completed_games_delta": games, "env_steps_delta": sampler.total_env_steps - env_before,
            "game_steps_delta": sampler.total_game_steps - steps_before,
            "games_per_second": games / elapsed,
            "env_steps_per_second": (sampler.total_env_steps - env_before) / elapsed,
            "game_steps_per_second": (sampler.total_game_steps - steps_before) / elapsed,
            "max_memory_allocated": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
            "compiled_unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"],
        }
        if self.wall_writer is None:
            self.wall_writer = csv.DictWriter(self.wall_file, fieldnames = list(row))
            self.wall_writer.writeheader()
        self.wall_writer.writerow(row)
        self.wall_file.flush()
        self.rows.append(row)
        if self.profiler is not None:
            self.profiler.step()

    @contextmanager
    def _contiguous_iteration(self, iteration: int, sampler):
        sampler.profile = False
        if self.iterations_seen == self.cfg.profile_warmup_iterations:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            self.block_before = (
                sampler.total_env_steps, sampler.total_game_steps,
                sampler.total_completed_games, sampler.commit_buffer_size,
            )
            self.block_first_iteration = iteration
            self.block_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            self.block_start = time.perf_counter()
        yield
        self.iterations_seen += 1
        if iteration == self.cfg.total_iterations and self.block_start is not None:
            # Include all queued device work, without fencing or flushing inside the block.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - self.block_start
            env_before, steps_before, games_before, pending_before = self.block_before
            games = sampler.total_completed_games - games_before
            trajectories = games + pending_before - sampler.commit_buffer_size
            count = self.iterations_seen - self.cfg.profile_warmup_iterations
            graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            row = {
                "first_iteration": self.block_first_iteration, "last_iteration": iteration,
                "iterations": count, "block_wall_ms": elapsed * 1000,
                "startup_and_warmup_ms": (self.block_start - self.run_start) * 1000,
                "amortized_iteration_wall_ms": elapsed * 1000 / count,
                "collect_batch_size": trajectories, "completed_games_delta": games,
                "env_steps_delta": sampler.total_env_steps - env_before,
                "game_steps_delta": sampler.total_game_steps - steps_before,
                "trajectories_per_second": trajectories / elapsed,
                "games_per_second": games / elapsed,
                "env_steps_per_second": (sampler.total_env_steps - env_before) / elapsed,
                "game_steps_per_second": (sampler.total_game_steps - steps_before) / elapsed,
                "max_memory_allocated": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
                "max_memory_reserved": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None,
                "compiled_unique_graphs_before": self.block_graphs,
                "compiled_unique_graphs_after": graphs,
                "status": "MEASURED" if graphs == self.block_graphs and count >= 3 else "INCONCLUSIVE",
            }
            writer = csv.DictWriter(self.wall_file, fieldnames = list(row))
            writer.writeheader()
            writer.writerow(row)
            self.wall_file.flush()
            self.rows.append(row)

    def _export_trace(self, profiler) -> None:
        trace = self.directory / "trace.json"
        profiler.export_chrome_trace(str(trace))
        events = profiler.key_averages()
        (self.directory / "cpu_top.txt").write_text(events.table(sort_by = "self_cpu_time_total", row_limit = 20))
        (self.directory / "cuda_top.txt").write_text(events.table(sort_by = "self_device_time_total", row_limit = 20))
        trace_events = json.loads(trace.read_text())["traceEvents"]
        phases, unmatched = trace_phase_summary(trace_events)
        with (self.directory / "phase_summary.csv").open("x", newline = "") as output:
            writer = csv.DictWriter(output, fieldnames = ["phase", "cpu_inclusive_ms", "cuda_kernel_ms", "count", "avg_cpu_per_call_ms", "avg_cuda_per_call_ms", "iteration_wall_share", "evidence"])
            writer.writeheader()
            writer.writerows(phases)
        self.validation = _validate_events(trace_events, self.cfg.algo_name)
        self.validation["uncorrelated_cuda_kernels"] = unmatched
        if unmatched:
            self.validation["status"] = "PROFILE_INVALID"
        (self.directory / "trace_validation.json").write_text(json.dumps(self.validation, indent = 2) + "\n")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self.enabled:
            return
        try:
            if self.profiler is not None:
                self.profiler.stop()
        finally:
            if self.sampler is not None:
                self.sampler.profile = False
            self.wall_file.close()
        lines = ["INCONCLUSIVE: bottleneck attribution requires review of wall measurements and the CPU/CUDA timeline."]
        if exc_value is not None:
            lines.append(f"Run failed: {type(exc_value).__name__}: {exc_value}")
        samples = [r["iteration_wall_ms"] for r in self.rows if r.get("phase") == "measurement"]
        if self.cfg.benchmark_contiguous:
            if self.rows:
                row = self.rows[0]
                lines.append(f"{row['status']}: {row['iterations']} contiguous iterations, {row['block_wall_ms']:.3f} ms total, {row['amortized_iteration_wall_ms']:.3f} ms per iteration. This is a measurement, not a performance gate verdict.")
            else:
                lines.append("INCONCLUSIVE: no complete post-warm-up block; partial work is not a timing sample.")
        if samples:
            p90 = statistics.quantiles(samples, n = 10, method = "inclusive")[8] if len(samples) > 1 else samples[0]
            lines.append(f"{len(samples)} post-warm-up iterations: median {statistics.median(samples):.3f} ms, P90 {p90:.3f} ms, min/max {min(samples):.3f}/{max(samples):.3f} ms.")
        lines.append("Warm-up is configurable; verify compilation has settled in the run log. Profiler CPU/CUDA sums and nested phases are not additive wall time.")
        if self.cfg.profile:
            lines.append(f"Trace validation: {json.dumps(self.validation or {'status': 'PROFILE_INVALID', 'reason': 'No complete active iteration exported'})}")
            lines.append("profile_wall.csv is diagnostic timing and must not replace wall_baseline.csv from a separate profiler-off run.")
        (self.directory / "findings.md").write_text("\n\n".join(lines) + "\n")
        if self.cfg.profile and exc_value is None and (self.validation is None or self.validation["status"] != "PROFILE_VALID"):
            raise RuntimeError(f"PROFILE_INVALID: see {self.directory / 'findings.md'}")
