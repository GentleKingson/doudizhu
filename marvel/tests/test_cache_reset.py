from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from marvel.config import ModelConfig
from marvel.model.agent import Agent
from marvel.model.cache import KVCache
from marvel.model.tensordict import TensorDict
from marvel.runtime.sampler import Sampler, SamplerConfig


class LegacyKVCache(KVCache):
    """Frozen pre-K1 reset, used as the behavioral oracle."""

    def reset(self, reset_mask):
        reset_mask = reset_mask.to(device = self.device)
        if reset_mask.dtype != torch.bool:
            reset_mask = reset_mask.to(torch.bool)
        if reset_mask.dim() > 1:
            reset_mask = reset_mask.squeeze(-1)
        if reset_mask.size(0) != self.batch:
            raise ValueError("reset_mask batch dimension does not match the cache")
        if not reset_mask.any():
            return
        self._seq_lens.masked_fill_(reset_mask, 0)
        self._position_ids.masked_fill_(reset_mask, 0)
        reset_kv = reset_mask.view(1, self.batch, 1, 1, 1)
        self.keys.masked_fill_(reset_kv, 0)
        self.values.masked_fill_(reset_kv, 0)


def make_cache(cls = KVCache):
    cache = cls(2, 9, max_ctx_len = 4, num_heads = 2, head_dim = 4,
                value_heads = 1, value_head_dim = 3, device = "cpu")
    for tensor in (cache.keys, cache.values):
        tensor.copy_(torch.arange(tensor.numel()).reshape(tensor.shape))
        tensor.flatten()[0:3] = torch.tensor([float("nan"), -0.0, float("inf")])
    cache._seq_lens.copy_(torch.arange(9) % 5)
    cache._position_ids.copy_(torch.arange(9) % 4)
    cache._step_mask.copy_(torch.arange(9) % 2)
    return cache


def assert_bits_equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if actual.numel() == 0:
        return
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_bitwise_oracle_accepts_empty_zero_stride_views(dtype):
    actual = torch.empty(0, dtype=dtype).as_strided((0, 3, 4), (0, 0, 0))
    assert_bits_equal(actual, torch.empty((0, 3, 4), dtype=dtype))
    with pytest.raises(AssertionError):
        assert_bits_equal(actual, torch.empty((0, 3, 5), dtype=dtype))


def assert_cache_equal(actual, expected):
    for field in ("keys", "values", "_seq_lens", "_position_ids", "_step_mask"):
        assert_bits_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize("rows", [[], [3], [0, 4, 8], list(range(9))])
@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
def test_reset_matches_legacy_and_preserves_storage(rows, column, dtype):
    actual, expected = make_cache(), make_cache(LegacyKVCache)
    fields = ("keys", "values", "_seq_lens", "_position_ids", "_step_mask")
    aliases = {field: getattr(actual, field) for field in fields}
    layers = list(zip(actual.key_cache, actual.value_cache))
    untouched_keys, untouched_values = actual.keys.clone(), actual.values.clone()
    mask = torch.zeros(18, dtype = dtype)[::2]  # Non-contiguous masks remain supported.
    mask[rows] = True if dtype == torch.bool else -2
    if rows and dtype == torch.float32:
        mask[rows[0]] = float("nan")
    selected = mask.bool().clone()
    mask = mask.unsqueeze(-1) if column else mask
    for _ in range(2):
        actual.reset(mask)
        expected.reset(mask)
        assert_cache_equal(actual, expected)
        for field, alias in aliases.items():
            assert getattr(actual, field) is alias
        for index, (keys, values) in enumerate(layers):
            assert actual.key_cache[index] is keys and actual.value_cache[index] is values
            assert keys.untyped_storage().data_ptr() == actual.keys.untyped_storage().data_ptr()
            assert values.untyped_storage().data_ptr() == actual.values.untyped_storage().data_ptr()
            assert_bits_equal(keys, actual.keys[index])
            assert_bits_equal(values, actual.values[index])
        assert_bits_equal(actual.keys[:, ~selected], untouched_keys[:, ~selected])
        assert_bits_equal(actual.values[:, ~selected], untouched_values[:, ~selected])
        assert not actual.keys[:, selected].count_nonzero()
        assert not actual.values[:, selected].count_nonzero()


@pytest.mark.parametrize("shape,fill", [
    ((), True), ((8,), True), ((8,), False), ((9, 2), True), ((9, 1, 1), True),
    ((9, 0), False), ((9, 2), False), ((9, 1, 1), False),
])
def test_invalid_and_empty_mask_behavior_matches_legacy(shape, fill):
    actual, expected = make_cache(), make_cache(LegacyKVCache)
    errors = []
    for cache in (expected, actual):
        try:
            cache.reset(torch.full(shape, fill, dtype = torch.bool))
        except (IndexError, ValueError, RuntimeError) as exc:
            errors.append((type(exc), str(exc)))
        else:
            errors.append(None)
    assert errors[0] == errors[1]
    assert_cache_equal(actual, expected)


@pytest.fixture
def agent_and_game():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    game = SimpleNamespace(
        MAX_GAME_HORIZON = 4, NUM_PLAYERS = 3, NUM_TOKENS = 16, PAD_TOKEN = 0,
        NUM_CHANNELS = 2, MAX_OBSERVATION_TOKENS = 2, MAX_ACTION_TOKENS = 2,
        MAX_AVAILABLE_ACTIONS = 3, MAX_BELIEF_ROWS = 1, MAX_BELIEF_TOKENS = 2,
        MAX_BELIEF_CANDIDATES = 2,
    )
    with torch.random.fork_rng(devices = []):
        torch.manual_seed(31)
        agent = Agent(SimpleNamespace(game = game, model = ModelConfig(
            history_dim = 8, feature_dim = 8, decoder_layers = 2, decoder_heads = 2,
            state_layers = 1, action_layers = 1, fuse_layers = 1,
        )), enable_value_head = True).eval()
        yield agent, game
    torch.set_num_threads(threads)


def rollout(time_dim):
    return TensorDict({
        "num_steps": torch.full((3,), time_dim, dtype = torch.long),
        "player_ids": torch.tensor([[-2, 0, 1, 2], [-1, 1, 2, 0], [2, 0, 1, 2]])[:, :time_dim],
        "obs_tokens": (torch.arange(3 * time_dim * 3 * 2).reshape(3, time_dim, 3, 2) % 15) + 1,
        "obs_channels": torch.ones((3, time_dim, 3, 2), dtype = torch.uint8),
        "avail_tokens": (torch.arange(3 * time_dim * 3 * 2).reshape(3, time_dim, 3, 2) % 15) + 1,
        "avail_weights": torch.tensor([1.0, 2.0, 0.0]).expand(3, time_dim, 3).clone(),
    }, batch_size = [3], device = "cpu")


def assert_cached_output_equal(agent, actual, expected, step):
    outputs = [agent(step, use_cache = True, past_key_values = cache) for cache in (actual, expected)]
    for field in ("avail_probs", "mask", "state_values", "player_state_value"):
        assert_bits_equal(getattr(outputs[0], field), getattr(outputs[1], field))
    assert_cache_equal(actual, expected)


def new_agent_cache(cls):
    return cls(2, 9, max_ctx_len = 4, num_heads = 2, head_dim = 4, device = "cpu")


@torch.inference_mode()
@pytest.mark.parametrize("prefix_len", [0, 3, 4])
def test_cache_only_rebuild_reset_append_and_full_window(agent_and_game, prefix_len):
    agent, _ = agent_and_game
    actual, expected = new_agent_cache(KVCache), new_agent_cache(LegacyKVCache)
    prefix = rollout(4)
    prefix["num_steps"] = torch.tensor([prefix_len, max(0, prefix_len - 1), 0])
    for cache in (actual, expected):
        cache.keys.fill_(7)
        cache.values.fill_(9)
        cache.reset(torch.ones(9, dtype = torch.bool))
        assert agent(prefix, use_cache = True, past_key_values = cache, cache_only = True) is None
    assert_cache_equal(actual, expected)
    # Synthetic terminal sequence: sparse, empty, dense and repeated environment resets.
    for terminal in ([True, False, False], [False, False, False], [False, True, True],
                     [False, True, True], [True, True, True]):
        mask = torch.tensor(terminal).repeat_interleave(3)
        actual.reset(mask)
        expected.reset(mask)
        assert_cache_equal(actual, expected)
        assert_cached_output_equal(agent, actual, expected, rollout(1))
    for _ in range(5):
        assert_cached_output_equal(agent, actual, expected, rollout(1))
    assert actual.seq_lens.tolist() == [4] * 9
    assert actual._position_ids.tolist() == [3] * 9


def raw_step():
    return SimpleNamespace(
        rewards = np.zeros((3, 3), dtype = np.float32), is_terminal = np.zeros(3, dtype = bool),
        player_ids = np.array([0, 1, 2]),
        observations = SimpleNamespace(tokens = np.ones((3, 3, 2), dtype = np.int64),
                                       channels = np.ones((3, 3, 2), dtype = np.uint8)),
        available_actions = SimpleNamespace(tokens = np.ones((3, 3, 2), dtype = np.int64),
                                            weights = np.ones((3, 3), dtype = np.float32),
                                            is_commit = np.ones((3, 3), dtype = bool)),
    )


@torch.inference_mode()
@pytest.mark.parametrize("caller", ["marvel", "kagami", "yomi"])
def test_shared_sampler_rebuild_and_terminal_reset(agent_and_game, monkeypatch, caller):
    from kagami.search import _sampler_bootstrap
    from marvel.env_batch import EnvStepBatch
    from yomi.runtime import sampler as yomi_sampler

    agent, game = agent_and_game
    agent._device = torch.device("cpu")
    prefix = rollout(4)
    num_steps = np.array([4, 3, 1])
    samplers = []
    for cls in (KVCache, LegacyKVCache):
        envs = SimpleNamespace(belief_supervision = lambda: None)
        if caller == "kagami":
            sampler = _sampler_bootstrap.build_sampler(
                game = game, policy_model = agent, rollout_temp = 1.0, envs = envs, envs_step = raw_step(),
            )
        else:
            sampler = Sampler(
                config = SamplerConfig(game = game, device = "cpu", rollout_batch_size = 3, collect_batch_size = 0),
                envs = envs, parsed_envs_step = EnvStepBatch.from_env_step(raw_step(), SimpleNamespace(device = "cpu")),
            )
        sampler.agent_kv = new_agent_cache(cls)
        sampler._reset_mask_full = torch.ones(9, dtype = torch.bool)
        sampler.agent_kv.keys.fill_(7)
        sampler.agent_kv.values.fill_(9)
        if caller == "kagami":
            _sampler_bootstrap._write_rollout_prefix_excluding_current_state(
                sampler = sampler, rollout_prefix = prefix, num_steps_np = num_steps,
            )
        elif caller == "yomi":
            collector = yomi_sampler.CompletedEpisodeRootCollector(
                sampler = sampler, policy = agent, game = game, pad_token = 0,
            )
            collector.reset()
            game.parse_history_strs = lambda rows: rows
            monkeypatch.setattr(yomi_sampler, "history_batch_to_rollout", lambda **kwargs: (prefix, num_steps, None, None, None))
            collector._prefill_rollout_buffer_from_history_strs(history_strs = ["a", "b", "c"])
            collector._prepare_agent_cache()
        else:
            for field in prefix:
                sampler.rollout_buffer[field].copy_(prefix[field])
            sampler.rollout_buffer["num_steps"].copy_(torch.from_numpy(num_steps - 1))
        sampler.prepare_agent_cache(agent_model = agent)
        samplers.append(sampler)
    actual, expected = samplers
    assert_cache_equal(actual.agent_kv, expected.agent_kv)
    assert actual.agent_kv.seq_lens.tolist() == [3, 3, 3, 2, 2, 2, 0, 0, 0]
    for terminal in ([False, True, False], [True, False, True], [True, True, True]):
        untouched = actual.agent_kv.keys.clone()
        for sampler in samplers:
            sampler._commit_done(torch.tensor(terminal))
        mask = torch.tensor(terminal).repeat_interleave(3)
        assert not actual.agent_kv.keys[:, mask].count_nonzero()
        assert_bits_equal(actual.agent_kv.keys[:, ~mask], untouched[:, ~mask])
        assert_cache_equal(actual.agent_kv, expected.agent_kv)
        assert_cached_output_equal(agent, actual.agent_kv, expected.agent_kv, rollout(1))
    for sampler in samplers:
        sampler.rollout_buffer.reset()
        sampler.prepare_agent_cache(agent_model = agent)
        assert not sampler.agent_kv.keys.count_nonzero()
        assert sampler.agent_kv.seq_lens.tolist() == [0] * 9
    assert_cache_equal(actual.agent_kv, expected.agent_kv)


@torch.inference_mode()
def test_real_doudizhu_terminal_sequences_collect_and_rebuild_match(agent_and_game):
    npl = pytest.importorskip("nanopokerlib._nanopokerlib")
    game = npl.doudizhu
    agent = Agent(SimpleNamespace(game = game, model = ModelConfig(
        history_dim = 8, feature_dim = 8, decoder_layers = 2, decoder_heads = 2,
        state_layers = 1, action_layers = 1, fuse_layers = 1,
    )), enable_value_head = True).eval()
    samplers, reset_sequences = [], []
    for cls in (KVCache, LegacyKVCache):
        sampler = Sampler(config = SamplerConfig(
            game = game, device = "cpu", rollout_batch_size = 2, collect_batch_size = 2, env_seed = 73,
        ))
        sampler._ensure_agent_cache(agent)
        cache = cls(2, 6, max_ctx_len = sampler.T, num_heads = 2, head_dim = 4, device = "cpu")
        reset_masks = []
        original_reset = cache.reset

        def recorded_reset(mask, *, original = original_reset, recorded = reset_masks):
            recorded.append(mask.clone())
            original(mask)

        cache.reset = recorded_reset
        sampler.agent_kv = cache
        samplers.append(sampler)
        reset_sequences.append(reset_masks)

    for _ in range(2):
        rng_before = torch.get_rng_state()
        actual = samplers[0].collect(agent_model = agent)
        rng_after = torch.get_rng_state()
        torch.set_rng_state(rng_before)
        expected = samplers[1].collect(agent_model = agent)
        assert_bits_equal(torch.get_rng_state(), rng_after)
        for field in actual:
            assert_bits_equal(actual[field], expected[field])
        for field in samplers[0].rollout_buffer.storage:
            assert_bits_equal(samplers[0].rollout_buffer[field], samplers[1].rollout_buffer[field])
        assert_cache_equal(samplers[0].agent_kv, samplers[1].agent_kv)
        for field in ("total_completed_games", "total_game_steps", "total_env_steps", "commit_buffer_size"):
            assert getattr(samplers[0], field) == getattr(samplers[1], field)
        for field in samplers[0].parsed_envs_step.build_transition():
            assert_bits_equal(samplers[0].parsed_envs_step.build_transition()[field],
                              samplers[1].parsed_envs_step.build_transition()[field])

    assert len(reset_sequences[0]) == len(reset_sequences[1])
    for actual_mask, expected_mask in zip(*reset_sequences):
        assert_bits_equal(actual_mask, expected_mask)
        assert torch.equal(actual_mask.view(2, 3), actual_mask.view(2, 3)[:, :1].expand(2, 3))
    assert any(0 < int(mask.sum()) < 6 for mask in reset_sequences[0])
    assert samplers[0].total_completed_games >= 4
