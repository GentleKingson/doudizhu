import json
from types import SimpleNamespace

import pytest
import torch

from experiments.vrpo_repro.staged.live_numeric import LiveTrace, digest


def test_digest_checks_shape_dtype_and_raw_bytes():
    values = torch.tensor([0., -0., float("nan")])
    assert digest(values) == digest(values.clone())
    assert digest(values) != digest(values.reshape(1, 3))
    assert digest(values[:1]) != digest(values[1:2])
    assert digest(torch.ones(2, dtype=torch.long)) != digest(torch.ones(2))
    assert digest(torch.tensor(1.))["shape"] == []


def test_live_trace_rejects_first_different_value(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    event = dict(ordinal=0, collect=0, kind="probabilities", payload=digest(torch.tensor([.25, .75])))
    (reference / "live-trace.jsonl").write_text(json.dumps(event) + "\n")
    trace = LiveTrace(tmp_path / "candidate", reference)
    with pytest.raises(AssertionError, match="Live trace diverged at event 0"):
        with trace.observe():
            trace.emit("probabilities", torch.tensor([.3, .7]))
    assert json.loads((trace.directory / "live-coverage.json").read_text())["events"] == 1
    assert json.loads((trace.directory / "live-trace.jsonl").read_text())["payload"] == digest(torch.tensor([.3, .7]))


def test_live_trace_restores_production_methods_after_failure(tmp_path):
    from marvel.env_batch import EnvStepBatch
    from marvel.model.cache import KVCache
    from marvel.runtime.sampler import Sampler

    convert, reset, collect = EnvStepBatch.from_env_step, KVCache.reset, Sampler.collect
    with pytest.raises(RuntimeError, match="interrupted"):
        with LiveTrace(tmp_path / "failed").observe():
            raise RuntimeError("interrupted")
    assert EnvStepBatch.from_env_step == convert
    assert KVCache.reset is reset and Sampler.collect is collect


def test_real_cpu_sampler_and_replay_default_repeat(tmp_path):
    npl = pytest.importorskip("nanopokerlib._nanopokerlib")
    from marvel.buffer import BufferConfig, TrajectoryBuffer, trajectory_field_specs
    from marvel.config import ModelConfig
    from marvel.model.agent import Agent
    from marvel.runtime.sampler import Sampler, SamplerConfig

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(31)
            game = npl.doudizhu
            agent = Agent(SimpleNamespace(game=game, model=ModelConfig(
                history_dim=8, feature_dim=8, decoder_layers=1, decoder_heads=2,
                state_layers=1, action_layers=1, fuse_layers=1,
            ))).eval()
            rng = torch.get_rng_state()
            for variant in ("default", "repeat"):
                torch.set_rng_state(rng)
                trace = LiveTrace(tmp_path / variant, None if variant == "default" else tmp_path / "default")
                with trace.observe():
                    sampler = Sampler(config=SamplerConfig(game=game, device="cpu", rollout_batch_size=2,
                                                          collect_batch_size=2, env_seed=73))
                    replay = TrajectoryBuffer(8, BufferConfig(game=game, device="cpu", field_specs=trajectory_field_specs(game)))
                    for _ in range(2):
                        batch = sampler.collect(agent_model=agent)
                        replay.add(batch)
                        for _ in range(4):
                            replay.sample(2)
                coverage = json.loads((tmp_path / variant / "live-coverage.json").read_text())
                assert coverage["calls"]["collected_rollout"] == 2
                assert coverage["calls"]["replay_sample"] == 8
                assert coverage["reset_selected_rows"] > 0
                assert set(coverage["observed_player_ids"]) == {"-2", "0", "1", "2"}
            assert (tmp_path / "default" / "live-trace.jsonl").read_bytes() == (tmp_path / "repeat" / "live-trace.jsonl").read_bytes()
    finally:
        torch.set_num_threads(threads)


def test_wrapper_uses_live_numeric_and_independent_variant_paths(tmp_path, monkeypatch):
    from experiments.vrpo_repro.staged import frozen_numeric, live_numeric

    def numeric(root, variant, *, live, observation_context):
        trace = observation_context.__self__
        assert root == tmp_path and live
        assert trace.directory == tmp_path / f"live-{variant}"
        assert trace.reference == (None if variant == "default" else tmp_path / "live-default")
        return variant

    monkeypatch.setattr(frozen_numeric, "run", numeric)
    for variant in ("default", "repeat", "k1", "e1"):
        assert live_numeric.run(tmp_path, variant) == variant
