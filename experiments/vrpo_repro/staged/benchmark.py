"""One normal-mode continuous block; identical lightweight observers on both sides."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch

from compile_mode import compile_mode, acceleration_stats
from experiments.vrpo_repro.repro_runtime import runtime_settings
from marvel.config import TrainConfig
from marvel.optim.common import MultiOptimizer
from marvel.training.vrpo import VRPOTrainer
from marvel.training.workflow import TrainingWorkflow
from marvel.utils import configure_torch


def run(directory, mode):
    cfg = TrainConfig(
        device="cuda", seed=0, rollout_batch_size=256, collect_batch_size=128,
        mini_batches=2, replay_buffer_scale=16, total_iterations=50,
        param_stats_interval=-1, checkpoint_interval=-1, evaluation_interval=-1,
        benchmark=True, benchmark_contiguous=True, profile_warmup_iterations=20,
        profile_trace_dir=str(directory / "diagnostics"), logging_dir=str(directory / "training"),
        run_name=directory.name,
    )
    configure_torch(cfg)
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.is_autocast_enabled("cuda") and torch.get_num_threads() == 6
    runtimes, samples = [], []
    counts = {phase: dict(agent=0, critic=0) for phase in ("warmup", "measurement")}
    original_collect, original_step = TrainingWorkflow.collect_trajectories, MultiOptimizer.step

    def factory(config):
        runtime = TrainingWorkflow(config, game=config.game)
        runtimes.append(runtime)
        return runtime

    def collect(runtime, *args, **kwargs):
        rollout = original_collect(runtime, *args, **kwargs)
        if runtime.current_iteration > cfg.profile_warmup_iterations:
            # Retain device copies for post-block histograms; no extra host fence in the block.
            samples.append((rollout["num_steps"].clone(), rollout["avail_weights"].clone()))
        return rollout

    def step(optimizer, *args, **kwargs):
        runtime = runtimes[0]
        role = "agent" if optimizer is runtime._state.entries["agent_optimizer_state_dict"] else "critic"
        phase = "measurement" if runtime.current_iteration > cfg.profile_warmup_iterations else "warmup"
        result = original_step(optimizer, *args, **kwargs)
        counts[phase][role] += 1
        return result

    with compile_mode(mode) as compile_calls, pytest.MonkeyPatch.context() as patch:
        patch.setattr(TrainingWorkflow, "collect_trajectories", collect)
        patch.setattr(MultiOptimizer, "step", step)
        VRPOTrainer(cfg, runtime_factory=factory).run()
    assert counts == {"warmup": dict(agent=160, critic=160), "measurement": dict(agent=240, critic=240)}
    assert len(samples) == 30
    lengths_hist = torch.zeros(178, dtype=torch.int64)
    actions_hist = torch.zeros(36, dtype=torch.int64)
    observer_bytes = sum(n.numel()*n.element_size() + w.numel()*w.element_size() for n, w in samples)
    for lengths, weights in samples:
        lengths, weights = lengths.cpu(), weights.cpu()
        valid = torch.arange(weights.shape[1])[None, :] < lengths[:, None]
        lengths_hist += torch.bincount(lengths.long(), minlength=178)
        actions_hist += torch.bincount((weights > 0).sum(-1)[valid], minlength=36)
    record = dict(
        optimizer_steps=counts, length_histogram=lengths_hist.tolist(),
        legal_action_count_histogram=actions_hist.tolist(), observer_retained_bytes=observer_bytes,
        observer="Same GPU clones and Python step counters in every mode; histogram transfers occur after timed block.",
        runtime=runtime_settings(), config=asdict(cfg), compile_calls=compile_calls,
        acceleration=acceleration_stats(),
    )
    assert sum(record["length_histogram"]) == 3840
    (directory / "workload.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve(), sys.argv[2])
