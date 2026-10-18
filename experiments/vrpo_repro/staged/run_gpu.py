"""Budgeted experiment, optionally continuing an explicitly authorized prior run."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_budget(protocol):
    assert protocol['approved_gpu_seconds'] == 1800
    consumed = protocol.get('prior_gpu_seconds', 0)
    limit = protocol.get('run_budget_seconds', 1620)
    assert 0 <= consumed < 1800 and 0 < limit
    assert consumed + limit + 5 <= 1800, 'Cumulative budget must include termination allowance'
    return consumed, limit


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def run_child(command, *, cwd, env, output, deadline, check_resources):
    """The deadline includes process cleanup; callers never extend it after failure."""
    if time.monotonic() >= deadline - 20:
        raise TimeoutError("Stage budget has no remaining launch/cleanup allowance")
    started = time.monotonic()
    with output.open("x") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline - 20:
                    raise TimeoutError("Stage budget exhausted")
                check_resources(process.pid)
                try:
                    process.wait(timeout=min(5, max(.01, deadline - 20 - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f"Child failed with exit status {process.returncode}: {output.name}")
        finally:
            stop_process(process)
    return dict(process_wall_seconds=time.monotonic()-started, returncode=process.returncode)


def main(root):
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    assert not (root / "job-status.json").exists(), "Never restart or overwrite an experiment"
    protocol = json.loads((root / "protocol.json").read_text())
    consumed, run_limit = run_budget(protocol)
    continuation = 'previous_root' in protocol
    device = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    assert device == protocol["gpu_uuid"], "Unexpected GPU"
    for relative, expected in protocol["package_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    if continuation:
        previous = Path(protocol['previous_root'])
        for relative, expected in protocol['previous_evidence_sha256'].items():
            assert hashlib.sha256((previous / relative).read_bytes()).hexdigest() == expected, relative
        for name in ('repeat', 'live-repeat', 'live-k1', 'live-e1'):
            assert json.loads((previous/name/'summary.json').read_text())['status'] == 'PASS'
    state = dict(phase="starting", started_at=started_at,
                 approved_gpu_seconds=1800, prior_gpu_seconds=consumed, run_budget_seconds=run_limit,
                 completed=[], current=None, candidate_results={})
    def update(**values):
        state.update(values, elapsed_seconds=time.monotonic()-started,
                     cumulative_elapsed_seconds=consumed+time.monotonic()-started,
                     updated_at=datetime.now(timezone.utc).isoformat())
        write_json(root / "job-status.json", state)

    common = dict(os.environ, OMP_NUM_THREADS="6", WANDB_DISABLED="true", PYTHONUNBUFFERED="1",
                  PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
                  TORCH_LOGS="recompiles", CUDA_VISIBLE_DEVICES="0")
    samples = []
    def resources(pid):
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5)
        memory, utilization, temperature, power = map(float, result.stdout.strip().split(","))
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True, timeout=5)
        foreign = [p.strip() for p in apps.stdout.splitlines() if p.strip() and int(p.strip()) != pid]
        samples.append(dict(elapsed=time.monotonic()-started, run=state["current"],
                            memory_mib=memory, utilization=utilization, temperature=temperature, power_w=power))
        if memory > 10240 or foreign:
            raise RuntimeError(f"Resource gate: memory={memory} MiB; other compute PIDs={foreign}")

    def child(name, variant, command, phase_deadline, numeric=True):
        source = root / "sources" / variant
        env = dict(common, PYTHONPATH=os.pathsep.join([str(source), str(root)]),
                   TORCHINDUCTOR_CACHE_DIR=str(root / "cache" / variant))
        if numeric:
            env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0")
        else:
            for key in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "CUDA_LAUNCH_BLOCKING"):
                env.pop(key, None)
            env["MARVEL_METRICS_CSV"] = str(root / name / "metrics.csv")
        update(current=name)
        timing = run_child(command, cwd=source, env=env, output=root / (name + ".log"),
                           deadline=min(started+phase_deadline, started+run_limit), check_resources=resources)
        if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > 10*1024**3:
            raise RuntimeError("New experiment exceeded 10 GiB disk budget")
        write_json(root / (name + "-timing.json"), timing)
        state["completed"].append(name)
        update(last_completed=name)

    python = sys.executable
    frozen = [python, "-m", "experiments.vrpo_repro.staged.frozen_numeric", str(root)]
    live = [python, "-m", "experiments.vrpo_repro.staged.live_numeric", str(root)]
    try:
        update(phase="numeric")
        if continuation:
            child('e1-cuda-contracts', 'e1', [python, '-m', 'pytest', '-q', '-x',
                  'experiments/vrpo_repro/staged/test_cuda_contracts.py::test_cuda_env_conversion_counts_fields_and_numpy_lifetime',
                  'experiments/vrpo_repro/staged/test_cuda_contracts.py::test_cuda_real_env_steps_and_reset_preserve_retained_tensors',
                  '--junitxml='+str(root/'e1-cuda-contracts.xml')], run_limit)
            child('g1-cuda-contracts', 'g1', [python, '-m', 'pytest', '-q', '-x',
                  'experiments/vrpo_repro/test_gae_cuda_graph.py',
                  '--junitxml='+str(root/'g1-cuda-contracts.xml')], run_limit)
        else:
            for name in ("default", "repeat"):
                child("frozen-"+name, "baseline", frozen+[name], 480)
                assert json.loads((root/name/"summary.json").read_text())["status"] == "PASS"
            for name, variant in (("default", "baseline"), ("repeat", "baseline"), ("k1", "k1"), ("e1", "e1")):
                child("live-"+name, variant, live+[name], 480)
                assert json.loads((root/("live-"+name)/"summary.json").read_text())["status"] == "PASS"
            child("cuda-contracts", "contracts", [python, "-m", "pytest", "-q", "-x",
                  "experiments/vrpo_repro/staged/test_cuda_contracts.py",
                  "experiments/vrpo_repro/test_gae_cuda_graph.py", "--junitxml="+str(root/"cuda-contracts.xml")], 480)
        child("frozen-candidate", "g1", frozen+["candidate"], run_limit if continuation else 480)
        assert json.loads((root/"candidate"/"summary.json").read_text())["status"] == "PASS"
        state["numeric_status"] = "PASS"

        def benchmark(name, variant, deadline):
            directory = root/name
            directory.mkdir()
            mode = "gae-cudagraph" if variant == "g1" else "default"
            child(name, variant, [python, "-m", "experiments.vrpo_repro.staged.benchmark", str(directory), mode], deadline, numeric=False)
            with (directory/"diagnostics/wall_contiguous.csv").open() as csvfile:
                rows = list(csv.DictReader(csvfile))
            assert len(rows) == 1
            row = rows[0]
            assert row["status"] == "MEASURED" and int(row["iterations"]) == 30
            assert int(row["collect_batch_size"]) == 3840
            assert max(int(row["max_memory_allocated"]), int(row["max_memory_reserved"])) <= 10*1024**3
            workload = json.loads((directory/"workload.json").read_text())
            assert workload["optimizer_steps"]["measurement"] == dict(agent=240,critic=240)
            assert not workload["runtime"]["deterministic_algorithms"]
            return row

        update(phase="normal_aa")
        first, second = [benchmark(name, "baseline", run_limit if continuation else 780) for name in ("aa-1", "aa-2")]
        state["normal_aa"] = dict(first=first, second=second,
            wall_change_percent=100*(float(second["block_wall_ms"])/float(first["block_wall_ms"])-1))
        update(phase="independent_ab")
        for candidate in ("k1", "e1", "g1"):
            # Alternate pair order before observing results; each candidate has its own fresh A.
            order = [("a", "baseline"), ("b", candidate)] if candidate != "e1" else [("b", candidate), ("a", "baseline")]
            pair = {arm:benchmark(candidate+"-"+arm, variant, run_limit) for arm,variant in order}
            reduction = 1-float(pair["b"]["block_wall_ms"])/float(pair["a"]["block_wall_ms"])
            state["candidate_results"][candidate] = dict(status="INCONCLUSIVE_SINGLE_PAIR", order=[arm for arm,_ in order],
                wall_reduction_percent=100*reduction, pair=pair, ci=None, quality="NOT_VERIFIED")
            update(last_pair=candidate)
        update(phase="completed", current=None)
    except BaseException as error:
        update(phase="stopped", failed_run=state["current"], stop_reason=f"{type(error).__name__}: {error}", current=None)
        raise
    finally:
        write_json(root/"gpu-samples.json", samples)
        update(gpu_work_finished=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
