"""Bounded serial search on one G4. Each trial is a fresh process."""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/content/g4-vrpo')
OUT = ROOT / 'results'
REPO = ROOT / 'repo'
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
OUT.mkdir(parents=True, exist_ok=True)
START = time.monotonic()
DEADLINE = START + 55 * 60
ENV = dict(os.environ, WANDB_DISABLED='true', PYTHONUNBUFFERED='1',
    PYTORCH_ALLOC_CONF='expandable_segments:True',
    TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'inductor-cache'),
    TORCHINDUCTOR_COMPILE_THREADS='2', NANOPOKERLIB_BUILD_JOBS='4')
results = []


def run(command, **kw):
    subprocess.run(command, check=True, env=ENV, timeout=min(600, max(1, DEADLINE-time.monotonic())), **kw)


def trial(config, label, warmup=16, measure=8, seed=0):
    if time.monotonic() > DEADLINE - 180:
        raise TimeoutError('Search budget exhausted')
    folder = OUT / label
    folder.mkdir()
    settings = dict(config, warmup=warmup, measure=measure, block_size=8, seed=seed, output=str(folder))
    (folder / 'settings.json').write_text(json.dumps(settings))
    env = dict(ENV, OMP_NUM_THREADS=str(config['threads']), MKL_NUM_THREADS=str(config['threads']), PYTHONPATH=str(REPO))
    print('START ' + label + ' ' + json.dumps(settings), flush=True)
    started = time.monotonic()
    with (folder / 'gpu.csv').open('w') as gpu, (folder / 'train.log').open('w') as log:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits', '--loop-ms=200'], stdout=gpu, stderr=subprocess.DEVNULL)
        process = subprocess.Popen([sys.executable, str(ROOT / 'benchmark.py'), str(folder / 'settings.json')],
            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=min(600, DEADLINE-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    samples = list(csv.reader((folder / 'gpu.csv').read_text().splitlines()))
    peak = max(float(r[0]) for r in samples if len(r) == 3)
    total = float(samples[-1][1])
    result = dict(label=label, config=config, seed=seed, exit_code=process.returncode,
        timed_out=timed_out, gpu_peak_mib=peak, gpu_total_mib=total, wall_seconds=time.monotonic()-started)
    measurement = folder / 'measurement.json'
    if measurement.exists() and process.returncode == 0:
        result.update(json.loads(measurement.read_text()))
        result['conservative_peak_mib'] = max(peak, result['peak_reserved_mib'] + 512)
        result['headroom_mib'] = total - result['conservative_peak_mib']
        result['eligible'] = result['headroom_mib'] >= max(3072, total * .15) and result['new_graphs'] == 0
    else:
        result['eligible'] = False
        result['error_tail'] = (folder / 'train.log').read_text()[-6000:]
    assert not list(folder.rglob('*.pt'))
    results.append(result)
    (folder / 'result.json').write_text(json.dumps(result, indent=2))
    (OUT / 'results.json').write_text(json.dumps(results, indent=2))
    print('RESULT ' + json.dumps({k:v for k,v in result.items() if k not in ('blocks', 'error_tail', 'settings')}), flush=True)
    if not measurement.exists():
        print(result.get('error_tail', ''), flush=True)
    if result['exit_code'] != 0 and 'out of memory' not in result.get('error_tail', '').lower():
        raise RuntimeError('Non-memory failure; stop for diagnosis')
    return result


def screen(config):
    if any(r['config'] == config for r in results):
        return
    if len([r for r in results if r['label'].startswith('screen')]) >= 12:
        return
    return trial(config, 'screen-' + str(len(results)).zfill(2))


def fastest():
    valid = [r for r in results if r['eligible']]
    if not valid:
        raise RuntimeError('No eligible candidate; inspect errors before consuming more credits')
    return max(valid, key=lambda r:r['trajectories_per_second'])['config'].copy()


try:
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    assert 'RTX PRO 6000' in gpu and 'Blackwell' in gpu, gpu
    (OUT / 'hardware.json').write_text(json.dumps(dict(gpu=gpu, cpu_count=os.cpu_count(), affinity=len(os.sched_getaffinity(0)), python=sys.version)))
    if not REPO.exists():
        run(['git', 'clone', '--depth', '1', PROTOCOL['repository'], str(REPO)])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == PROTOCOL['commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
    run([sys.executable, '-m', 'pip', 'install', '-q', 'cmake', 'ninja', 'pybind11', 'tyro', 'numpy>=2', 'wheel', 'setuptools'])
    run([sys.executable, '-m', 'pip', 'install', '-q', '--no-build-isolation', '--no-deps', '-e', str(REPO / 'nanopokerlib')])
    run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', str(REPO)])
    (OUT / 'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    (OUT / 'protocol.json').write_text(json.dumps(PROTOCOL, indent=2))
    print('SETUP_SECONDS ' + str(time.monotonic()-START), flush=True)
    for config in PROTOCOL['candidates']:
        result = screen(config)
        if result and result['exit_code'] != 0 and 'out of memory' not in result.get('error_tail', '').lower():
            raise RuntimeError('Non-memory failure; stop search for diagnosis')
    anchor = fastest()
    for threads in (2, 4, 8):
        screen(dict(anchor, threads=threads))
    anchor = fastest()
    for mini in sorted({2, 4, 8} - {anchor['mini_batches']}):
        screen(dict(anchor, mini_batches=mini))
    anchor = fastest()
    for rollout in sorted({max(128, anchor['rollout']//2), anchor['rollout']*2}):
        screen(dict(anchor, rollout=rollout))
    anchor = fastest()
    for collect in (anchor['collect']//2, anchor['collect']*2):
        if 64 <= collect <= 2048:
            screen(dict(anchor, collect=collect))
    finalists = sorted((r for r in results if r['eligible']), key=lambda r:r['trajectories_per_second'], reverse=True)[:2]
    verify = []
    for seed in (0, 1):
        for index in ([0, 1] if seed == 0 else [1, 0]):
            if index < len(finalists):
                verify.append(trial(finalists[index]['config'], f'verify-{index}-seed{seed}', 16, 16, seed))
    summary = []
    for finalist in finalists:
        runs = [r for r in verify if r['config'] == finalist['config']]
        if len(runs) == 2 and all(r['eligible'] for r in runs):
            seconds = sum(r['seconds'] for r in runs)
            summary.append(dict(config=finalist['config'], trajectories_per_second=sum(sum(b['trajectories'] for b in r['blocks']) for r in runs)/seconds,
                headroom_mib=min(r['headroom_mib'] for r in runs), gpu_peak_mib=max(r['gpu_peak_mib'] for r in runs),
                trials=[r['label'] for r in runs]))
    assert summary, 'No finalist passed verification'
    best = max(summary, key=lambda r:r['trajectories_per_second'])
    (OUT / 'summary.json').write_text(json.dumps(dict(best=best, finalists=summary, total_seconds=time.monotonic()-START), indent=2))
    print('BEST ' + json.dumps(best), flush=True)
except BaseException:
    (OUT / 'failure.txt').write_text(traceback.format_exc())
    traceback.print_exc()
finally:
    import shutil
    shutil.make_archive(str(ROOT / 'results'), 'zip', OUT)
    (ROOT / 'done.json').write_text(json.dumps(dict(seconds=time.monotonic()-START, success=(OUT / 'summary.json').exists())))
