from __future__ import annotations

import csv
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from marvel.config import ModelConfig, TrainConfig, VRPOConfig
from marvel.runtime.profiling import VRPODiagnostics, _IPPO_REQUIRED_SCOPES, _REQUIRED_SCOPES, trace_phase_summary, validate_trace
from marvel.training.ippo import IPPOTrainer
from marvel.training.vrpo import VRPOTrainer
from marvel.training.workflow import TrainingWorkflow


@pytest.fixture
def smoke_config(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield TrainConfig(
        device = "cpu", use_compile = False, total_iterations = 4,
        checkpoint_interval = -1, evaluation_interval = -1, param_stats_interval = -1,
        rollout_batch_size = 4, collect_batch_size = 4, mini_batches = 2,
        replay_buffer_scale = 2, profile_warmup_iterations = 1,
        logging_dir = str(tmp_path / "training"), profile_trace_dir = str(tmp_path / "diagnostics"),
        model = ModelConfig(history_dim = 8, feature_dim = 16, decoder_layers = 1, decoder_heads = 2,
                            state_layers = 1, action_layers = 1, fuse_layers = 1),
        vrpo = VRPOConfig(replay_buffer_device = "cpu", aux_epoch = 1),
    )
    torch.set_num_threads(threads)


def train(cfg):
    runtimes = []

    def factory(config):
        runtime = TrainingWorkflow(config, game = config.game)
        runtimes.append(runtime)
        return runtime

    trainer = IPPOTrainer if cfg.algo_name == "ippo" else VRPOTrainer
    trainer(cfg, runtime_factory = factory).run()
    return runtimes[0]


def assert_same_state(before, after):
    if isinstance(before, torch.Tensor):
        torch.testing.assert_close(before, after, rtol = 0, atol = 0)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            assert_same_state(before[key], after[key])
    elif isinstance(before, (tuple, list)):
        assert len(before) == len(after)
        for left, right in zip(before, after):
            assert_same_state(left, right)
    else:
        assert before == after


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("algo_name", ["ippo", "vrpo"])
def test_benchmark_preserves_cpu_training_and_real_counts(smoke_config, monkeypatch, algo_name, contiguous):
    smoke_config = replace(smoke_config, algo_name = algo_name)
    def unexpected(*args, **kwargs):
        pytest.fail("Profiler scopes and CUDA synchronizations must be disabled here")

    monkeypatch.setattr(torch.profiler, "record_function", unexpected)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    plain = train(smoke_config)
    rng_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert not VRPODiagnostics(smoke_config).enabled
    assert not Path(smoke_config.profile_trace_dir).exists()
    measured = train(replace(smoke_config, benchmark = True, benchmark_contiguous = contiguous))
    assert rng_before[0] == random.getstate()
    rng_after = np.random.get_state()
    assert rng_before[1][0] == rng_after[0] and rng_before[1][2:] == rng_after[2:]
    np.testing.assert_array_equal(rng_before[1][1], rng_after[1])
    assert torch.equal(rng_before[2], torch.get_rng_state())
    keys = ["agent_resume_state_dict", "agent_state_dict"]
    if algo_name == "vrpo":
        keys.append("critic_state_dict")
    for key in keys:
        before = plain._state.entries[key].state_dict()
        after = measured._state.entries[key].state_dict()
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol = 0, atol = 0)
            assert torch.isfinite(after[name]).all()
    for key in ("agent_optimizer_state_dict", "critic_optimizer_state_dict"):
        if key in plain._state.entries:
            assert_same_state(plain._state.entries[key].state_dict(), measured._state.entries[key].state_dict())
    assert plain.total_env_steps == measured.total_env_steps
    assert plain.total_game_steps == measured.total_game_steps
    assert plain.sampler.total_completed_games == measured.sampler.total_completed_games
    filename = "wall_contiguous.csv" if contiguous else "wall_baseline.csv"
    with open(smoke_config.profile_trace_dir + "/" + filename) as output:
        rows = list(csv.DictReader(output))
    if contiguous:
        assert len(rows) == 1 and int(rows[0]["iterations"]) == 3
        assert int(rows[0]["collect_batch_size"]) == 12
        assert float(rows[0]["block_wall_ms"]) > 0 and rows[0]["status"] == "MEASURED"
        assert not Path(smoke_config.profile_trace_dir, "wall_baseline.csv").exists()
    else:
        assert len(rows) == 4 and [r["phase"] for r in rows] == ["warmup"] + ["measurement"] * 3
        assert sum(int(r["completed_games_delta"]) for r in rows) == measured.sampler.total_completed_games
        assert sum(int(r["env_steps_delta"]) for r in rows) == measured.total_env_steps
        assert sum(int(r["game_steps_delta"]) for r in rows) == measured.total_game_steps
        assert all(int(r["collect_batch_size"]) == 4 and float(r["iteration_wall_ms"]) > 0 for r in rows)
    snapshot = json.loads(Path(smoke_config.profile_trace_dir, "config_snapshot.json").read_text())
    assert snapshot["torch_num_threads"] == torch.get_num_threads() == 2
    assert not Path(smoke_config.profile_trace_dir, "trace.json").exists()
    assert TrainConfig().use_compile and not TrainConfig().profile and not TrainConfig().benchmark


@pytest.mark.parametrize("compile_during_block", [False, True])
def test_contiguous_block_fences_only_boundaries_and_counts_actual_work(smoke_config, monkeypatch, compile_during_block):
    events = []
    clock = iter((5., 10., 13.))
    monkeypatch.setattr("marvel.runtime.profiling.time.perf_counter", lambda: next(clock))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: events.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 200)
    monkeypatch.setattr(torch.profiler, "record_function", lambda *args: pytest.fail("No profiling in block"))
    graphs = torch._dynamo.utils.counters["stats"]
    monkeypatch.setitem(graphs, "unique_graphs", 5)
    sampler = SimpleNamespace(profile = False, total_env_steps = 0, total_game_steps = 0,
                              total_completed_games = 0, commit_buffer_size = 0)
    cfg = replace(smoke_config, benchmark = True, benchmark_contiguous = True)
    with VRPODiagnostics(cfg) as diagnostics:
        # Exercise the CUDA fence protocol without allocating or running on a GPU.
        diagnostics.device = torch.device("cuda")
        for iteration in range(1, 5):
            with diagnostics.iteration(iteration, sampler):
                assert events == ([] if iteration == 1 else ["sync", "reset"])
                assert diagnostics.wall_file.tell() == 0
                sampler.total_env_steps += iteration
                sampler.total_game_steps += 10 * iteration
                sampler.total_completed_games += 4
                sampler.commit_buffer_size = iteration
                if compile_during_block and iteration == 3:
                    graphs["unique_graphs"] += 1
    assert events == ["sync", "reset", "sync"]
    row, = diagnostics.rows
    assert row["block_wall_ms"] == 3000 and row["amortized_iteration_wall_ms"] == 1000
    assert row["startup_and_warmup_ms"] == 5000
    assert (row["first_iteration"], row["last_iteration"], row["iterations"]) == (2, 4, 3)
    assert row["env_steps_delta"] == 9 and row["game_steps_delta"] == 90
    assert row["completed_games_delta"] == 12 and row["collect_batch_size"] == 9
    assert row["max_memory_allocated"] == 100 and row["max_memory_reserved"] == 200
    assert row["status"] == ("INCONCLUSIVE" if compile_during_block else "MEASURED")


@pytest.mark.parametrize("benchmark,profile", [(False, False), (True, True)])
def test_contiguous_requires_profiler_off_benchmark(smoke_config, benchmark, profile):
    cfg = replace(smoke_config, benchmark = benchmark, profile = profile, benchmark_contiguous = True)
    with pytest.raises(ValueError, match = "requires benchmark=True and profile=False"):
        with VRPODiagnostics(cfg):
            pytest.fail("Invalid mode must fail before training")


def test_contiguous_failure_does_not_publish_partial_timing(smoke_config):
    sampler = SimpleNamespace(profile = False, total_env_steps = 0, total_game_steps = 0,
                              total_completed_games = 0, commit_buffer_size = 0)
    cfg = replace(smoke_config, benchmark = True, benchmark_contiguous = True, profile_warmup_iterations = 0)
    with pytest.raises(RuntimeError, match = "interrupted"):
        with VRPODiagnostics(cfg) as diagnostics:
            with diagnostics.iteration(1, sampler):
                pass
            with diagnostics.iteration(2, sampler):
                raise RuntimeError("interrupted")
    assert not diagnostics.rows and not sampler.profile
    assert not (diagnostics.directory / "wall_contiguous.csv").read_text()
    assert "no complete" in (diagnostics.directory / "findings.md").read_text()


@pytest.mark.parametrize("start,count,status", [(3, 3, "MEASURED"), (5, 1, "INCONCLUSIVE"), (6, 0, None)])
def test_contiguous_resume_counts_only_iterations_actually_run(smoke_config, start, count, status):
    sampler = SimpleNamespace(profile = False, total_env_steps = 0, total_game_steps = 0,
                              total_completed_games = 0, commit_buffer_size = 0)
    cfg = replace(smoke_config, benchmark = True, benchmark_contiguous = True, total_iterations = 6)
    with VRPODiagnostics(cfg) as diagnostics:
        for iteration in range(start, 7):
            with diagnostics.iteration(iteration, sampler):
                sampler.total_completed_games += 1
    if status is None:
        assert not diagnostics.rows
    else:
        row, = diagnostics.rows
        assert row["iterations"] == count and row["collect_batch_size"] == count
        assert row["status"] == status


@pytest.mark.parametrize("algo_name,required", [("ippo", _IPPO_REQUIRED_SCOPES), ("vrpo", _REQUIRED_SCOPES)])
def test_trace_validation_rejects_cpu_only_missing_phases_and_compilation(tmp_path, algo_name, required):
    path = tmp_path / "trace.json"
    scopes = [{"cat": "user_annotation", "name": name} for name in required]
    events = scopes + [{"cat": "cpu_op", "name": "aten::mm"}]
    path.write_text(json.dumps({"traceEvents": events}))
    assert validate_trace(path, algo_name)["status"] == "PROFILE_INVALID"
    events.append({"cat": "kernel", "name": "sgemm"})
    path.write_text(json.dumps({"traceEvents": events}))
    assert validate_trace(path, algo_name)["status"] == "PROFILE_VALID"
    events.append({"cat": "cpu_op", "name": "_compile.compile_inner"})
    path.write_text(json.dumps({"traceEvents": events}))
    assert validate_trace(path, algo_name)["compilation_events"] == 1
    assert validate_trace(path, algo_name)["status"] == "PROFILE_INVALID"
    path.write_text(json.dumps({"traceEvents": [{"cat": "cpu_op"}, {"cat": "kernel"}]}))
    assert f"{algo_name}.policy.backward" in validate_trace(path, algo_name)["missing_scopes"]


def test_profile_requires_cuda(smoke_config):
    with pytest.raises(RuntimeError, match = "Profiling requires CUDA"):
        with VRPODiagnostics(replace(smoke_config, profile = True)):
            pytest.fail("CPU-only profiling must fail before training")


def test_phase_summary_correlates_worker_thread_kernels_and_excludes_gpu_annotations():
    events = [
        {"cat": "user_annotation", "name": "vrpo.policy.backward", "ts": 10, "dur": 100, "tid": 1},
        {"cat": "gpu_user_annotation", "name": "vrpo.policy.backward", "ts": 100, "dur": 200},
        {"cat": "cuda_driver", "name": "cuLaunchKernel", "ts": 40, "dur": 3, "tid": 2, "args": {"correlation": 7}},
        {"cat": "kernel", "name": "backward_kernel", "ts": 120, "dur": 150, "args": {"correlation": 7}},
        {"cat": "kernel", "name": "unmatched_kernel", "ts": 280, "dur": 10, "args": {"correlation": 8}},
    ]
    rows, unmatched = trace_phase_summary(events)
    assert len(rows) == 1 and rows[0]["count"] == 1 and unmatched == 1
    assert rows[0]["cpu_inclusive_ms"] == 0.1 and rows[0]["cuda_kernel_ms"] == 0.15
    assert rows[0]["iteration_wall_share"] == ""


def test_profiler_stops_and_sampler_scopes_reset_on_error(smoke_config):
    stopped = []
    diagnostics = VRPODiagnostics(replace(smoke_config, benchmark = True))
    sampler = SimpleNamespace(profile = True)
    with pytest.raises(ValueError, match = "training failed"):
        with diagnostics:
            diagnostics.profiler = SimpleNamespace(stop = lambda: stopped.append(True))
            diagnostics.sampler = sampler
            raise ValueError("training failed")
    assert stopped == [True] and sampler.profile is False
    assert "Run failed: ValueError" in (diagnostics.directory / "findings.md").read_text()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "Requires CUDA/CUPTI")
@pytest.mark.parametrize("algo_name", ["ippo", "vrpo"])
def test_real_cuda_trace_contains_all_training_phases(smoke_config, algo_name):
    cfg = replace(smoke_config, algo_name = algo_name, device = "cuda", profile = True,
                  vrpo = replace(smoke_config.vrpo, replay_buffer_device = "cuda"))
    train(cfg)
    directory = Path(cfg.profile_trace_dir)
    validation = validate_trace(directory / "trace.json", algo_name)
    assert validation["status"] == "PROFILE_VALID", validation
    assert validation["cpu_events"] > 0 and validation["cuda_kernel_events"] > 0
    assert (directory / "cpu_top.txt").stat().st_size > 0
    assert (directory / "cuda_top.txt").stat().st_size > 0
    with (directory / "phase_summary.csv").open() as output:
        rows = list(csv.DictReader(output))
    phases = {r["phase"]: r for r in rows}
    assert len(phases) == len(rows)
    assert int(phases[f"{algo_name}.policy.forward"]["count"]) == 8
    assert float(phases[f"{algo_name}.policy.backward"]["cuda_kernel_ms"]) > 0
    if algo_name == "vrpo":
        assert int(phases["vrpo.critic.forward"]["count"]) == 2
        assert float(phases["vrpo.critic.backward"]["cuda_kernel_ms"]) > 0
    assert not (directory / "wall_baseline.csv").exists()
