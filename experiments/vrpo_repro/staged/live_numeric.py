"""Strict real-Sampler numerical evidence; this is deliberately not a timer."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from marvel.buffer import TrajectoryBuffer
from marvel.env_batch import EnvStepBatch
from marvel.model.cache import KVCache
from marvel.runtime.sampler import Sampler
from experiments.vrpo_repro.staged.numeric_checks import tensor_digest


def digest(value):
    """Hash exact tensor bytes and metadata without retaining GPU tensors."""
    if isinstance(value, torch.Tensor):
        return tensor_digest(value)
    if hasattr(value, "items"):
        return {str(key): digest(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [digest(child) for child in value]
    return value


def cache_metadata(cache):
    return {key: getattr(cache, key) for key in ("_seq_lens", "_position_ids", "_step_mask")}


class LiveTrace:
    def __init__(self, directory: Path, reference: Path | None = None):
        self.directory = Path(directory)
        self.reference = Path(reference) if reference is not None else None
        self.counts = Counter()
        self.player_ids = Counter()
        self.ordinal = 0
        self.collect_number = 0
        self.reset_selected_rows = 0
        self.replay_generation = 0

    def emit(self, kind, value):
        event = dict(ordinal=self.ordinal, collect=self.collect_number, kind=kind, payload=digest(value))
        self.output.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        self.ordinal += 1
        self.counts[kind] += 1
        if self.reference_file is not None:
            line = self.reference_file.readline()
            assert line, f"Reference trace ended before event {event['ordinal']}: {kind}"
            expected = json.loads(line)
            assert event == expected, f"Live trace diverged at event {event['ordinal']}: {kind}"

    def full_cache(self, kind, cache):
        self.emit(kind, dict(keys=cache.keys, values=cache.values, **cache_metadata(cache)))

    @contextmanager
    def observe(self):
        original_convert = EnvStepBatch.from_env_step
        original_record = EnvStepBatch.record_agent_outputs
        original_rollout = Sampler.rollout_step
        original_collect = Sampler.collect
        original_rebuild = Sampler._rebuild_agent_cache
        original_reset = KVCache.reset
        original_add = TrajectoryBuffer.add
        original_sample = TrajectoryBuffer.sample

        def convert(cls, step, cfg):
            parsed = original_convert(step, cfg)
            self.emit("env_conversion", parsed.build_transition())
            self.player_ids.update(parsed["player_ids"].detach().cpu().tolist())
            return parsed

        def record(parsed, **kwargs):
            result = original_record(parsed, **kwargs)
            probs, actions = kwargs["avail_probs"], kwargs["action_taken"]
            weights = parsed["avail_weights"]
            assert bool(torch.isfinite(probs).all()) and bool((probs >= 0).all())
            assert not bool(probs[weights <= 0].count_nonzero()), "Illegal action probability"
            assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), rtol=0, atol=1e-5)
            assert bool((weights.gather(1, actions[:, None]) > 0).all()), "Illegal sampled action"
            self.emit("sampled_action", dict(kwargs, player_ids=parsed["player_ids"]))
            return result

        def reset(cache, mask):
            selected = mask.to(device=cache.device, dtype=torch.bool).reshape(-1)
            indices = selected.nonzero(as_tuple=True)[0]
            is_full = int(indices.numel()) == cache.batch_size
            before = cache_metadata(cache)
            # Metadata is tiny; clone because reset changes it in place.
            before = {key: value.clone() for key, value in before.items()}
            original_reset(cache, mask)
            payload = dict(mask=mask, before=before, after=cache_metadata(cache), full=is_full)
            if is_full:
                assert not bool(cache.keys.count_nonzero()) and not bool(cache.values.count_nonzero())
            elif indices.numel():
                keys, values = cache.keys.index_select(1, indices), cache.values.index_select(1, indices)
                assert not bool(keys.count_nonzero()) and not bool(values.count_nonzero())
                payload.update(selected_keys=keys, selected_values=values)
                self.reset_selected_rows += int(indices.numel())
            self.emit("cache_reset", payload)

        def rebuild(sampler, agent_model):
            self.emit("rebuild_input", sampler.rollout_buffer.storage)
            result = original_rebuild(sampler, agent_model)
            self.full_cache("rebuild_cache", sampler.agent_kv)
            return result

        def rollout(sampler, **kwargs):
            terminal = original_rollout(sampler, **kwargs)
            self.emit("sampler_step", dict(terminal=terminal, cache=cache_metadata(sampler.agent_kv),
                                           num_steps=sampler.rollout_buffer["num_steps"],
                                           completed_games=sampler.total_completed_games,
                                           env_steps=sampler.total_env_steps, game_steps=sampler.total_game_steps,
                                           commit_buffer_size=sampler.commit_buffer_size))
            return terminal

        def collect(sampler, **kwargs):
            self.collect_number += 1
            result = original_collect(sampler, **kwargs)
            self.emit("collected_rollout", result)
            self.emit("pending_rollout", sampler.rollout_buffer.storage)
            self.full_cache("collected_cache", sampler.agent_kv)
            return result

        def add(buffer, batch):
            self.emit("replay_add_input", dict(generation=self.replay_generation, ptr=buffer._ptr, batch=batch))
            result = original_add(buffer, batch)
            self.replay_generation += 1
            self.emit("replay_storage", dict(generation=self.replay_generation, ptr=buffer._ptr,
                                             size=len(buffer), content=buffer.storage[:len(buffer)]))
            return result

        def sample(buffer, *args, **kwargs):
            result = original_sample(buffer, *args, **kwargs)
            self.emit("replay_sample", dict(generation=self.replay_generation, batch=result))
            return result

        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "live-trace.jsonl").open("x") as output:
            self.output = output
            self.reference_file = (self.reference / "live-trace.jsonl").open() if self.reference is not None else None
            try:
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(EnvStepBatch, "from_env_step", classmethod(convert))
                    patch.setattr(EnvStepBatch, "record_agent_outputs", record)
                    patch.setattr(Sampler, "rollout_step", rollout)
                    patch.setattr(Sampler, "collect", collect)
                    patch.setattr(Sampler, "_rebuild_agent_cache", rebuild)
                    patch.setattr(KVCache, "reset", reset)
                    patch.setattr(TrajectoryBuffer, "add", add)
                    patch.setattr(TrajectoryBuffer, "sample", sample)
                    yield self
                if self.reference_file is not None:
                    assert not self.reference_file.read(1), "Candidate trace ended before reference"
                assert self.collect_number == 2, "Exactly two real collect calls are required"
                assert self.counts["env_conversion"] > 1 and self.counts["sampled_action"] > 0
                assert self.counts["cache_reset"] > 2 and self.counts["rebuild_cache"] == 2
                assert self.reset_selected_rows > 0, "No partial terminal reset was exercised"
                assert self.counts["replay_add_input"] == 2 and self.counts["replay_sample"] == 8
            finally:
                if self.reference_file is not None:
                    self.reference_file.close()
                coverage = dict(events=self.ordinal, calls=dict(self.counts),
                    reset_selected_rows=self.reset_selected_rows, observed_player_ids=dict(self.player_ids),
                    production_reset_sha256=hashlib.sha256(inspect.getsource(original_reset).encode()).hexdigest(),
                    production_conversion_sha256=hashlib.sha256(inspect.getsource(original_convert).encode()).hexdigest(),
                    cache_coverage="Every step: length/position/step mask; terminal reset: all selected K/V rows; "
                                   "full reset: zero assertion; rebuild/collect boundaries: full K/V digest. "
                                   "Unselected K/V rows are not copied at every step; CPU/CUDA cache unit tests cover that contract.",
                    environment_rng="Fresh native environments use the declared env seed; exact per-step results are compared. "
                                    "No C++ RNG serialization or checkpoint-resume claim.",
                    timing="Synchronous CPU hashing is numerical instrumentation, never performance evidence.")
                (self.directory / "live-coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")


def run(root, variant):
    from experiments.vrpo_repro.staged.frozen_numeric import run as numeric_run

    root = Path(root)
    directory = root / f"live-{variant}"
    trace = LiveTrace(directory, None if variant == "default" else root / "live-default")
    return numeric_run(root, variant, live=True, observation_context=trace.observe)


def main():
    from experiments.vrpo_repro.repro_runtime import deterministic_runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("variant", choices=("default", "repeat", "k1", "e1"))
    args = parser.parse_args()
    with deterministic_runtime():
        run(args.root, args.variant)


if __name__ == "__main__":
    main()
