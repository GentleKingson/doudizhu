"""Audit the finite candidate search before choosing the overnight configuration."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

root=Path('/root/doudizhu/logs/localserver-5070-2026-10-12')
gpu=list(csv.reader((root/'gpu.csv').read_text().splitlines()))
times=[(datetime.strptime(r[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp(),float(r[1])) for r in gpu if len(r)==4]
runs=[]
for name in ('probe32','probe64','probe32t8'):
    folder=root/name
    result=json.loads((folder/'result.json').read_text())
    cfg=json.loads((folder/'config.json').read_text())
    settings=json.loads((folder/'settings.json').read_text())
    assert result['success'] and result['new_graphs']==0
    expected=dict(rollout_batch_size=512,collect_batch_size=2048,mini_batches=4,
        replay_buffer_scale=64,agent_merging_beta=.999,lr_stable_iterations=5000,reg_stable_iterations=500)
    assert all(cfg[k]==v for k,v in expected.items())
    assert cfg['ppo']['agent_epoch']==cfg['vrpo']['aux_epoch']==4
    assert cfg['ppo']['base_lr']==cfg['vrpo']['base_lr']==.0004
    assert cfg['vrpo']['replay_buffer_device']=='cpu'
    assert not list(folder.rglob('*.pt'))
    with (folder/'metrics/metrics.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert {int(r['step']) for r in rows}=={1,2,3,4,5}
    assert all(math.isfinite(float(r['value'])) for r in rows)
    end=datetime.fromisoformat(result['finished_at']).timestamp()
    peak=max(v for t,v in times if end-result['wall_seconds']-3<=t<=end)
    headroom=12227-max(peak,result['peak_reserved_mib']+512)
    assert headroom>=12227*.15 and result['host_min_available_gib']>=6.1
    runs.append(dict(name=name,chunk=settings['chunk'],threads=settings['threads'],
        trajectories_per_second=result['trajectories_per_second'],gpu_peak_mib=peak,
        conservative_headroom_mib=headroom,host_min_available_gib=result['host_min_available_gib']))
numeric=json.loads((root/'numeric-production.json').read_text())
assert numeric['replay_roundtrip_exact'] and numeric['full_capacity_touched']
assert all(numeric[k]['gradient_relative_l2']<2e-4 for k in ('agent','critic'))
best=max(runs,key=lambda r:r['trajectories_per_second'])
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='acdcf199cf422e2b683cb22142b8e7cb33b14460'
assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip()
manifest={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in subprocess.check_output(['git','ls-files'],text=True).splitlines() if Path(name).is_file()}
result=dict(passed=True,runs=runs,best=best,full_replay_training_verified=False,
    full_replay_allocation_and_touch_verified=True,source_hashes=manifest,
    limitation='Short candidate screening; full replay dynamics are verified during overnight training, not yet proven here.')
(root/'probe-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(best=best,runs=runs),indent=2))
