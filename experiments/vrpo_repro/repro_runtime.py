from contextlib import contextmanager
import os

import torch


@contextmanager
def deterministic_runtime():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert os.environ.get("PYTHONHASHSEED") == "0"
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
            yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def runtime_settings():
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                pythonhashseed=os.environ.get("PYTHONHASHSEED"))
