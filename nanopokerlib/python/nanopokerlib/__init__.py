# nanopokerlib/__init__.py

import importlib as _importlib
from types import ModuleType as _ModuleType
from typing import Dict as _Dict
from typing import Optional as _Optional

try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _pkg_version
except ModuleNotFoundError:  # pragma: no cover - Python < 3.8
    _PackageNotFoundError = None  # type: ignore[assignment]
    _pkg_version = None  # type: ignore[assignment]

_pybind = _importlib.import_module("._nanopokerlib", __name__)  # note the dot + underscore

from ._nanopokerlib import *  # noqa: F403,F401
from . import utils as utils

def _attach_optional_submodule(name: str) -> _Optional[_ModuleType]:
    module = getattr(_pybind, name, None)
    if isinstance(module, _ModuleType):
        globals()[name] = module
        return module
    return None


for _name in (
    "hulhe",
    "hunl200",
    "nlhe6m100",
    "huplo200",
    "ext",
    "bridge",
    "bridge_simple",
    "darkhex2",
    "darkhex3",
    "darkhex4",
    "darkhex11",
    "pttt",
    "doudizhu",
    "doudizhu_small",
    "doudizhu_tiny",
    "guandan",
    "hanabi2",
    "hanabi3",
    "hanabi4",
    "hanabi5",
    "liarsdice2d3f",
    "liarsdice2d4f",
    "liarsdice2d5f",
    "riichi4",
):
    _attach_optional_submodule(_name)

if isinstance(globals().get("riichi4"), _ModuleType) and hasattr(_pybind, "HistoryBoundary"):
    setattr(riichi4, "HistoryBoundary", _pybind.HistoryBoundary)
__version__ = getattr(_pybind, "__version__", None)
if __version__ is None:
    if _pkg_version is not None and _PackageNotFoundError is not None:
        try:
            __version__ = _pkg_version("nanopokerlib")
        except _PackageNotFoundError:  # pragma: no cover
            __version__ = None


def _attach_eas_traversers() -> None:
    """Expose exp-a-spiel traversers when the optional dependency is present."""
    try:
        eas = _importlib.import_module("eas")
    except ModuleNotFoundError:
        return

    registry: _Dict[str, str] = {
        "darkhex3": "AbruptDhTraverser",
        "pttt": "AbruptPtttTraverser",
    }
    for game_name, attr_name in registry.items():
        game_module = globals().get(game_name)
        traverser_cls = getattr(eas, attr_name, None)
        if not isinstance(game_module, _ModuleType) or traverser_cls is None:
            continue
        setattr(game_module, "traverser", traverser_cls)
        if not hasattr(game_module, "Traverser"):
            setattr(game_module, "Traverser", traverser_cls)


_attach_eas_traversers()
__all__ = [n for n in dir() if not n.startswith("_")]
