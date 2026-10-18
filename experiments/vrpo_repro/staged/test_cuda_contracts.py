"""Small CUDA correctness gates; the existing GAE Graph test is run alongside these."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from marvel.env_batch import EnvStepBatch
from marvel.model.cache import KVCache
from marvel.tests import test_cache_reset as cache_contract
from marvel.tests.test_env_batch import _Step, _legacy_from_env_step


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
agent_and_game = cache_contract.agent_and_game


def _cuda_cache(cls):
    source = cache_contract.make_cache(cls)
    result = cls(2, 9, max_ctx_len=4, num_heads=2, head_dim=4,
                 value_heads=1, value_head_dim=3, device="cuda")
    for name in ("keys", "values", "_seq_lens", "_position_ids", "_step_mask"):
        getattr(result, name).copy_(getattr(source, name))
    return result


@pytest.mark.parametrize("rows", [[], [3], [0, 4, 8], list(range(9))])
@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
@pytest.mark.parametrize("mask_device", ["cpu", "cuda"])
def test_cuda_reset_matches_legacy_and_preserves_aliases(rows, column, dtype, mask_device):
    actual = _cuda_cache(KVCache)
    expected = _cuda_cache(cache_contract.LegacyKVCache)
    aliases = {name: getattr(actual, name) for name in
               ("keys", "values", "_seq_lens", "_position_ids", "_step_mask")}
    layer_aliases = list(zip(actual.key_cache, actual.value_cache))
    before_keys, before_values = actual.keys.clone(), actual.values.clone()
    mask = torch.zeros(18, dtype=dtype, device=mask_device)[::2]
    mask[rows] = True if dtype == torch.bool else -2
    if rows and dtype == torch.float32:
        mask[rows[0]] = float("nan")
    selected = mask.to(device="cuda", dtype=torch.bool)
    mask = mask.unsqueeze(-1) if column else mask
    for _ in range(2):
        actual.reset(mask)
        expected.reset(mask)
        cache_contract.assert_cache_equal(actual, expected)
        for name, alias in aliases.items():
            assert getattr(actual, name) is alias
        for layer, (keys, values) in enumerate(layer_aliases):
            assert actual.key_cache[layer] is keys and actual.value_cache[layer] is values
            assert keys.untyped_storage().data_ptr() == actual.keys.untyped_storage().data_ptr()
            assert values.untyped_storage().data_ptr() == actual.values.untyped_storage().data_ptr()
            cache_contract.assert_bits_equal(keys, actual.keys[layer])
            cache_contract.assert_bits_equal(values, actual.values[layer])
        cache_contract.assert_bits_equal(actual.keys[:, ~selected], before_keys[:, ~selected])
        cache_contract.assert_bits_equal(actual.values[:, ~selected], before_values[:, ~selected])
        assert not actual.keys[:, selected].count_nonzero()
        assert not actual.values[:, selected].count_nonzero()


@torch.inference_mode()
@pytest.mark.parametrize("prefix_len", [0, 3, 4])
def test_cuda_reset_populate_cached_append_and_full_window(agent_and_game, prefix_len):
    agent, _ = agent_and_game
    agent = agent.to("cuda")
    actual, expected = [cls(2, 9, max_ctx_len=4, num_heads=2, head_dim=4, device="cuda")
                        for cls in (KVCache, cache_contract.LegacyKVCache)]
    prefix = cache_contract.rollout(4).to("cuda")
    prefix["num_steps"] = torch.tensor([prefix_len, max(0, prefix_len - 1), 0], device="cuda")
    for cache in (actual, expected):
        cache.keys.fill_(7)
        cache.values.fill_(9)
        cache.reset(torch.ones(9, dtype=torch.bool, device="cuda"))
        assert agent(prefix, use_cache=True, past_key_values=cache, cache_only=True) is None
    cache_contract.assert_cache_equal(actual, expected)
    step = cache_contract.rollout(1).to("cuda")
    for terminal in ([True, False, False], [False, False, False], [False, True, True],
                     [False, True, True], [True, True, True]):
        mask = torch.tensor(terminal, device="cuda").repeat_interleave(3)
        actual.reset(mask)
        expected.reset(mask)
        cache_contract.assert_cache_equal(actual, expected)
        cache_contract.assert_cached_output_equal(agent, actual, expected, step)
    for _ in range(5):
        cache_contract.assert_cached_output_equal(agent, actual, expected, step)
    assert actual.seq_lens.tolist() == [4] * 9
    assert actual._position_ids.tolist() == [3] * 9


def _assert_cuda_step_equal(actual, expected):
    assert actual.batch_size == expected.batch_size
    actual_fields, expected_fields = actual.build_transition(), expected.build_transition()
    assert list(actual_fields.keys()) == list(expected_fields.keys())
    for name, value in actual_fields.items():
        assert value.device.type == expected_fields[name].device.type == "cuda"
        cache_contract.assert_bits_equal(value, expected_fields[name])


@pytest.mark.parametrize("batch", [0, 1, 5])
def test_cuda_env_conversion_counts_fields_and_numpy_lifetime(batch):
    cfg = SimpleNamespace(device="cuda")
    step = _Step(batch)
    expected = _legacy_from_env_step(step, cfg)
    step.calls.clear()
    actual = EnvStepBatch.from_env_step(step, cfg)
    assert step.calls == {"rewards": 1, "is_terminal": 1, "observations": 1,
                          "player_ids": 1, "available_actions": 1}
    _assert_cuda_step_equal(actual, expected)
    # Every CUDA field must own device storage even where CPU keeps a NumPy alias.
    for value in list(step.source.values()) + step.returned:
        for array in vars(value).values() if isinstance(value, SimpleNamespace) else [value]:
            array.fill(23)
    del step
    _assert_cuda_step_equal(actual, expected)


def test_cuda_real_env_steps_and_reset_preserve_retained_tensors():
    import nanopokerlib as npl

    game = npl.doudizhu
    envs = game.EnvBatch(2)
    step = envs.reset(19)
    cfg = SimpleNamespace(device="cuda")
    retained = []
    for _ in range(8):
        actual = EnvStepBatch.from_env_step(step, cfg)
        expected = _legacy_from_env_step(step, cfg)
        _assert_cuda_step_equal(actual, expected)
        retained.append((actual, expected))
        actions = step.available_actions
        indices = actions.weights.argmax(axis=1)
        step = envs.step(game.actions_from_np(actions.tokens[np.arange(2), indices]))
    envs.reset(20)
    del step, envs
    for actual, expected in retained:
        _assert_cuda_step_equal(actual, expected)
