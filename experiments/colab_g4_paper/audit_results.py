"""Recompute candidate and finalist metrics from downloaded raw records."""
import csv
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
results = json.loads((root/'results.json').read_text())
summary = json.loads((root/'summary.json').read_text())
assert not list(root.rglob('*.pt'))
manifest = json.loads((root/'source-manifest.json').read_text())
assert manifest['tracked_changes'] == ''
assert manifest['commit'] == 'acdcf199cf422e2b683cb22142b8e7cb33b14460'
checked = []
for result in results:
    if not result['eligible']:
        continue
    folder = root/result['label']
    measurement = json.loads((folder/'measurement.json').read_text())
    cfg = json.loads((folder/'train-config.json').read_text())
    settings = json.loads((folder/'settings.json').read_text())
    assert cfg['algo_name'] == 'vrpo' and cfg['replay_buffer_scale'] == 64
    assert cfg['ppo']['agent_epoch'] == cfg['vrpo']['aux_epoch'] == 4
    assert cfg['model']['architecture'] == 'v1.3'
    assert cfg['checkpoint_interval'] == -1
    assert cfg['rollout_batch_size'] == 512 and cfg['collect_batch_size'] == 2048
    assert cfg['mini_batches'] == 4 and cfg['agent_merging_beta'] == .999
    assert cfg['lr_stable_iterations'] == 5000 and cfg['reg_stable_iterations'] == 500
    assert cfg['vrpo']['replay_buffer_device'] == 'cpu'
    assert cfg['ppo']['base_lr'] == cfg['vrpo']['base_lr'] == .0004
    assert measurement['host_min_available_mib'] >= measurement['host_total_mib'] * .2
    if result['label'].startswith('verify'):
        assert settings['warmup'] >= cfg['replay_buffer_scale']
    assert cfg['total_iterations'] == settings['warmup'] + settings['measure']
    assert measurement['new_graphs'] == 0
    assert measurement['effective_torch_threads'] == settings['threads']
    blocks = measurement['blocks']
    seconds = sum(b['seconds'] for b in blocks)
    trajectories = sum(b['trajectories'] for b in blocks)
    assert trajectories == settings['measure'] * cfg['collect_batch_size']
    assert math.isclose(result['trajectories_per_second'], trajectories/seconds, rel_tol=1e-12)
    with (folder/'training/metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert {int(row['step']) for row in rows} == set(range(1, cfg['total_iterations']+1))
    fields = {row['key'] for row in rows if int(row['step']) == 1}
    for step in range(1, cfg['total_iterations']+1):
        step_rows = [row for row in rows if int(row['step']) == step]
        assert len(step_rows) == len(fields)
        assert {row['key'] for row in step_rows} == fields
    for row in rows:
        assert math.isfinite(float(row['value'])), (folder, row)
        if int(row['step']) >= 64 and row['key'] == 'buffer/replay_buffer_size':
            assert float(row['value']) == 131072
    with (folder/'gpu.csv').open() as f:
        gpu = list(csv.reader(f))
    peak = max(float(r[0]) for r in gpu if len(r) == 3)
    reserved = measurement['peak_reserved_mib']
    headroom = result['gpu_total_mib'] - max(peak, reserved+512)
    assert headroom == result['headroom_mib']
    assert headroom >= max(3072, result['gpu_total_mib']*.15)
    checked.append(result['label'])
verified = []
for finalist in summary['finalists']:
    runs = [r for r in results if r['label'] in finalist['trials']]
    assert len(runs) == 2 and {r['seed'] for r in runs} == {0,1}
    assert all(r['eligible'] and r['config'] == finalist['config'] for r in runs)
    seconds = sum(r['seconds'] for r in runs)
    row = dict(finalist)
    for key, field in [('trajectories_per_second','trajectories'),('env_steps_per_second','env_steps'),('game_steps_per_second','game_steps')]:
        row[key] = sum(sum(b[field] for b in r['blocks']) for r in runs)/seconds
    assert math.isclose(row['trajectories_per_second'],finalist['trajectories_per_second'],rel_tol=1e-12)
    row['seed_rates'] = {r['seed']:r['trajectories_per_second'] for r in runs}
    verified.append(row)
best = max(verified,key=lambda r:r['trajectories_per_second'])
assert best['config'] == summary['best']['config']
audit = dict(passed=True,checked=checked,verified=verified,checkpoint_files=0,source_clean=True)
(root/'audit.json').write_text(json.dumps(audit,indent=2))
(root/'best-parameters.json').write_text(json.dumps(best,indent=2))
print(json.dumps(audit,indent=2))
