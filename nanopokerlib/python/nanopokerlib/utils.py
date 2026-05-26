from __future__ import annotations

from collections.abc import Sequence

from ._nanopokerlib import utils as _utils


def append_prefix(histories: Sequence[str], prefix: str) -> list[str]:
    return _utils.append_prefix(histories, prefix)


__all__ = ["append_prefix"]
