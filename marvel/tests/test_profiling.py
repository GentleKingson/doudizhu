from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from marvel.config import ModelConfig, TrainConfig, VRPOConfig
from marvel.runtime.profiling import VRPODiagnostics, _REQUIRED_SCOPES, trace_phase_summary, validate_trace
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

    VRPOTrainer(cfg, runtime_factory = factory).run()
    return runtimes[0]


def test_benchmark_preserves_cpu_training_and_real_counts(smoke_config, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Profiler scopes and CUDA synchronizations must be disabled here")

    monkeypatch.setattr(torch.profiler, "record_function", unexpected)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    plain = train(smoke_config)
    assert not VRPODiagnostics(smoke_config).enabled
    assert not Path(smoke_config.profile_trace_dir).exists()
    measured = train(replace(smoke_config, benchmark = True))
    for key in ("agent_resume_state_dict", "critic_state_dict", "agent_state_dict"):
        before = plain._state.entries[key].state_dict()
        after = measured._state.entries[key].state_dict()
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol = 0, atol = 0)
            assert torch.isfinite(after[name]).all()
    assert plain.total_env_steps == measured.total_env_steps
    assert plain.total_game_steps == measured.total_game_steps
    assert plain.sampler.total_completed_games == measured.sampler.total_completed_games
    with open(smoke_config.profile_trace_dir + "/wall_baseline.csv") as output:
        rows = list(csv.DictReader(output))
    assert len(rows) == 4 and [r["phase"] for r in rows] == ["warmup"] + ["measurement"] * 3
    assert sum(int(r["completed_games_delta"]) for r in rows) == measured.sampler.total_completed_games
    assert sum(int(r["env_steps_delta"]) for r in rows) == measured.total_env_steps
    assert sum(int(r["game_steps_delta"]) for r in rows) == measured.total_game_steps
    assert all(int(r["collect_batch_size"]) == 4 and float(r["iteration_wall_ms"]) > 0 for r in rows)
    snapshot = json.loads(Path(smoke_config.profile_trace_dir, "config_snapshot.json").read_text())
    assert snapshot["torch_num_threads"] == torch.get_num_threads() == 2
    assert not Path(smoke_config.profile_trace_dir, "trace.json").exists()
    assert TrainConfig().use_compile and not TrainConfig().profile and not TrainConfig().benchmark


def test_trace_validation_rejects_cpu_only_missing_phases_and_compilation(tmp_path):
    path = tmp_path / "trace.json"
    scopes = [{"cat": "user_annotation", "name": name} for name in _REQUIRED_SCOPES]
    events = scopes + [{"cat": "cpu_op", "name": "aten::mm"}]
    path.write_text(json.dumps({"traceEvents": events}))
    assert validate_trace(path)["status"] == "PROFILE_INVALID"
    events.append({"cat": "kernel", "name": "sgemm"})
    path.write_text(json.dumps({"traceEvents": events}))
    assert validate_trace(path)["status"] == "PROFILE_VALID"
    events.append({"cat": "cpu_op", "name": "_compile.compile_inner"})
    path.write_text(json.dumps({"traceEvents": events}))
    assert validate_trace(path)["compilation_events"] == 1
    assert validate_trace(path)["status"] == "PROFILE_INVALID"
    path.write_text(json.dumps({"traceEvents": [{"cat": "cpu_op"}, {"cat": "kernel"}]}))
    assert "vrpo.policy.backward" in validate_trace(path)["missing_scopes"]


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
def test_real_cuda_trace_contains_all_training_phases(smoke_config):
    cfg = replace(smoke_config, device = "cuda", profile = True,
                  vrpo = replace(smoke_config.vrpo, replay_buffer_device = "cuda"))
    train(cfg)
    directory = Path(cfg.profile_trace_dir)
    validation = validate_trace(directory / "trace.json")
    assert validation["status"] == "PROFILE_VALID", validation
    assert validation["cpu_events"] > 0 and validation["cuda_kernel_events"] > 0
    assert (directory / "cpu_top.txt").stat().st_size > 0
    assert (directory / "cuda_top.txt").stat().st_size > 0
    with (directory / "phase_summary.csv").open() as output:
        rows = list(csv.DictReader(output))
    phases = {r["phase"]: r for r in rows}
    assert len(phases) == len(rows)
    assert int(phases["vrpo.policy.forward"]["count"]) == 8
    assert int(phases["vrpo.critic.forward"]["count"]) == 2
    assert float(phases["vrpo.policy.backward"]["cuda_kernel_ms"]) > 0
    assert float(phases["vrpo.critic.backward"]["cuda_kernel_ms"]) > 0
    assert not (directory / "wall_baseline.csv").exists()
