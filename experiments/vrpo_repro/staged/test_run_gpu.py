import os
import subprocess
import sys
import time

import pytest

from experiments.vrpo_repro.staged.run_gpu import run_budget, run_child


def test_continuation_cannot_reset_cumulative_budget():
    protocol = dict(approved_gpu_seconds=1800, prior_gpu_seconds=207.261949,
                    run_budget_seconds=1582)
    assert run_budget(protocol) == (207.261949, 1582)
    for changed in (dict(run_budget_seconds=1800), dict(prior_gpu_seconds=-1),
                    dict(prior_gpu_seconds=1800), dict(run_budget_seconds=1590)):
        with pytest.raises(AssertionError):
            run_budget(dict(protocol, **changed))


def test_child_records_success_and_stops_on_budget_or_resource_failure(tmp_path):
    result = run_child([sys.executable, "-c", "print('ok')"], cwd=tmp_path, env=os.environ,
                       output=tmp_path/"success.log", deadline=time.monotonic()+30,
                       check_resources=lambda pid: None)
    assert result["returncode"] == 0 and (tmp_path/"success.log").read_text().strip() == "ok"
    for mode in ("budget", "resource"):
        observed = []
        def check(pid):
            observed.append(pid)
            if mode == "resource":
                raise RuntimeError("foreign GPU process")
        with pytest.raises(TimeoutError if mode == "budget" else RuntimeError):
            run_child([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path,
                      env=os.environ, output=tmp_path/(mode+".log"),
                      deadline=time.monotonic()+(20.25 if mode == "budget" else 30), check_resources=check)
        assert observed
        with pytest.raises(ProcessLookupError):
            os.kill(observed[-1], 0)


def test_no_process_launch_when_budget_is_already_exhausted(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(TimeoutError):
        run_child([], cwd=tmp_path, env=os.environ, output=tmp_path/"never.log",
                  deadline=time.monotonic()+1, check_resources=lambda pid: None)
    assert not (tmp_path/"never.log").exists()
