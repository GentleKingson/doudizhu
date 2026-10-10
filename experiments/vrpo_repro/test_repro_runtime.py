import pytest
import torch

from experiments.vrpo_repro.repro_runtime import deterministic_runtime, runtime_settings


@pytest.mark.parametrize("fail", [False, True])
def test_runtime_is_strict_and_restores_after_exit(monkeypatch, fail):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    before = runtime_settings()
    try:
        with deterministic_runtime():
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert not torch.backends.cudnn.benchmark
            assert torch.backends.cudnn.deterministic
            assert torch.get_float32_matmul_precision() == before["float32_matmul_precision"]
            assert torch.backends.cuda.matmul.allow_tf32 == before["matmul_allow_tf32"]
            if fail:
                raise ValueError("intentional failure")
    except ValueError:
        assert fail
    assert runtime_settings() == before


def test_runtime_requires_frozen_environment(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = runtime_settings()
    with pytest.raises(AssertionError):
        with deterministic_runtime():
            pytest.fail("Missing workspace setting must reject entry")
    assert runtime_settings() == before
