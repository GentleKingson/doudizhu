from __future__ import annotations

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from marvel.env_batch import EnvOutcome, EnvStepBatch
from marvel.model.tensordict import TensorDict


def _legacy_from_env_step(step, cfg):
    """Frozen pre-E1 conversion, including dtype, copy and evaluation order."""
    batch = int(step.rewards.shape[0])
    device = torch.device(cfg.device)
    outcome = EnvOutcome(TensorDict({
        "rewards": step.rewards,
        "is_terminal": step.is_terminal.astype(bool),
    }, batch_size = [batch], device = device))
    observation = TensorDict({
        "obs_tokens": step.observations.tokens.astype(np.int64),
        "obs_channels": step.observations.channels.astype(np.uint8),
        "player_ids": step.player_ids.astype(np.int64),
        "avail_tokens": step.available_actions.tokens.astype(np.int64),
        "avail_weights": step.available_actions.weights,
        "avail_is_commit": step.available_actions.is_commit.astype(bool),
    }, batch_size = [batch], device = device)
    return EnvStepBatch(observation, outcome)


class _Step:
    """Like C++ StepBatch, each getter returns independently owned arrays."""

    def __init__(self, batch = 5):
        self.source = {
            "rewards": np.arange(batch * 3, dtype = np.float32).reshape(batch, 3),
            "is_terminal": np.resize(np.array([0, 1], dtype = np.int32), batch),
            "player_ids": np.resize(np.array([0, 1, 2, -1, -2], dtype = np.int32), batch),
            "observations": SimpleNamespace(
                tokens = np.arange(batch * 12, dtype = np.int32).reshape(batch, 3, 4),
                channels = np.resize(np.array([0, 1], dtype = np.float32), (batch, 3, 2)),
            ),
            "available_actions": SimpleNamespace(
                tokens = np.arange(batch * 8, dtype = np.int32).reshape(batch, 4, 2),
                weights = np.resize(np.array([0, 0.25, 0.75, 0], dtype = np.float32), (batch, 4)),
                is_commit = np.resize(np.array([0, 1], dtype = np.uint8), (batch, 4)),
            ),
        }
        self.calls = Counter()
        self.returned = []
        self.fail_at = None

    def __getattr__(self, name):
        self.calls[name] += 1
        if self.fail_at == name:
            raise RuntimeError(f"{name} getter failed")
        value = deepcopy(self.source[name])
        self.returned.append(value)
        return value


def _assert_equal(actual, expected):
    assert actual.batch_size == expected.batch_size
    assert actual.device == expected.device == torch.device("cpu")
    actual_fields = actual.build_transition()
    expected_fields = expected.build_transition()
    assert list(actual_fields.keys()) == list(expected_fields.keys())
    for key in actual_fields:
        torch.testing.assert_close(actual_fields[key], expected_fields[key], rtol = 0, atol = 0)


@pytest.mark.parametrize("batch", [0, 1, 5])
def test_getter_counts_and_all_fields_match_legacy(batch):
    step = _Step(batch)
    cfg = SimpleNamespace(device = "cpu")
    expected = _legacy_from_env_step(step, cfg)
    assert step.calls == {"rewards": 2, "is_terminal": 1, "observations": 2,
                          "player_ids": 1, "available_actions": 3}
    step.calls.clear()
    actual = EnvStepBatch.from_env_step(step, cfg)
    assert step.calls == {"rewards": 1, "is_terminal": 1, "observations": 1,
                          "player_ids": 1, "available_actions": 1}
    _assert_equal(actual, expected)
    if batch == 5:
        assert actual["player_ids"].tolist() == [0, 1, 2, -1, -2]
        assert actual.outcome["is_terminal"].tolist() == [False, True, False, True, False]


@pytest.mark.parametrize("convert", [_legacy_from_env_step, EnvStepBatch.from_env_step])
def test_numpy_copy_and_ownership_contract(convert):
    step = _Step()
    parsed = convert(step, SimpleNamespace(device = "cpu"))
    before = parsed.build_transition().clone()
    # A later environment step/getter must not overwrite any submitted field.
    for value in step.source.values():
        for array in vars(value).values() if isinstance(value, SimpleNamespace) else [value]:
            array.fill(17)
    convert(step, SimpleNamespace(device = "cpu"))
    for key, value in before.items():
        torch.testing.assert_close(parsed.build_transition()[key], value, rtol = 0, atol = 0)

    fresh = _Step()
    parsed = convert(fresh, SimpleNamespace(device = "cpu"))
    before = parsed.build_transition().clone()
    # CPU rewards/weights intentionally retain the returned NumPy storage.
    for value in fresh.returned:
        for array in vars(value).values() if isinstance(value, SimpleNamespace) else [value]:
            array.fill(23)
    for key, value in parsed.build_transition().items():
        if key in {"rewards", "avail_weights"}:
            assert torch.all(value == 23)
        else:
            torch.testing.assert_close(value, before[key], rtol = 0, atol = 0)


@pytest.mark.parametrize("name", ["rewards", "is_terminal", "observations", "player_ids", "available_actions"])
def test_getter_errors_propagate_like_legacy(name):
    for convert in [_legacy_from_env_step, EnvStepBatch.from_env_step]:
        step = _Step()
        step.fail_at = name
        with pytest.raises(RuntimeError, match = f"^{name} getter failed$"):
            convert(step, SimpleNamespace(device = "cpu"))


@pytest.mark.parametrize("fault", ["scalar_rewards", "negative_stride", "invalid_tokens", "missing_weights", "device"])
def test_invalid_inputs_preserve_exception_type_message_and_precedence(fault):
    errors = []
    for convert in [_legacy_from_env_step, EnvStepBatch.from_env_step]:
        step = _Step()
        cfg = SimpleNamespace(device = "cpu")
        if fault == "scalar_rewards":
            step.source["rewards"] = np.array(1, dtype = np.float32)
        elif fault == "negative_stride":
            # deepcopy normalizes strides, so use a plain step for this case.
            step = SimpleNamespace(**step.source)
            step.rewards = step.rewards[::-1]
        elif fault == "invalid_tokens":
            step.source["observations"].tokens = np.array(["invalid"])
            step.fail_at = "available_actions"
        elif fault == "missing_weights":
            del step.source["available_actions"].weights
        else:
            cfg.device = "invalid-device"
            step.fail_at = "observations"
        with pytest.raises(Exception) as error:
            convert(step, cfg)
        errors.append((type(error.value), str(error.value)))
    assert errors[0] == errors[1]


def test_real_environment_steps_match_and_do_not_overwrite_previous_tensors():
    npl = pytest.importorskip("nanopokerlib._nanopokerlib")
    game = npl.doudizhu
    envs = game.EnvBatch(2)
    step = envs.reset(19)
    cfg = SimpleNamespace(device = "cpu")
    saved = []
    for _ in range(8):
        parsed = EnvStepBatch.from_env_step(step, cfg)
        _assert_equal(parsed, _legacy_from_env_step(step, cfg))
        saved.append((parsed, parsed.build_transition().clone()))
        actions = step.available_actions
        indices = actions.weights.argmax(axis = 1)
        step = envs.step(game.actions_from_np(actions.tokens[np.arange(2), indices]))
    envs.reset(20)
    del step, envs
    for parsed, expected in saved:
        for key, value in expected.items():
            torch.testing.assert_close(parsed.build_transition()[key], value, rtol = 0, atol = 0)
