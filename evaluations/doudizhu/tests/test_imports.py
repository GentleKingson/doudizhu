import importlib
import sys


def test_doudizhu_package_import_does_not_import_scenario() -> None:
    sys.modules.pop("evaluations.doudizhu.scenario", None)
    sys.modules.pop("evaluations.doudizhu", None)

    importlib.import_module("evaluations.doudizhu")

    assert "evaluations.doudizhu.scenario" not in sys.modules


def test_doudizhu_submodule_import_still_works() -> None:
    module = importlib.import_module("evaluations.doudizhu.specs")

    assert module is not None
