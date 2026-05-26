from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


def _socket_tmp_base() -> Path:
    preferred = Path("/tmp")
    if preferred.is_dir():
        return preferred
    return Path(tempfile.gettempdir())


@pytest.fixture
def socket_tmp_path() -> Path:
    path = Path(tempfile.mkdtemp(prefix = "dse_sock_", dir = str(_socket_tmp_base())))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors = True)
