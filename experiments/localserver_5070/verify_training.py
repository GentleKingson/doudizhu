"""Verify the completed overnight run from raw records, checkpoint and live process state."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import torch
from experiments.colab_a100_paper.timed_training import checkpoints, validate_resume_payload

root=Path('/root/doudizhu/logs/localserver-5070-2026-10-12')
train=root/'train'
launch=json.loads((root/'launch.json').read_text())
result=json.loads((train/'result.json').read_text())
assert result['success']
deadline=datetime.fromisoformat(launch['deadline']).timestamp()
finished=datetime.fromisoformat(result['finished_at']).timestamp()
assert deadline-120 <= finished <= deadline+20, 'Training did not finish near the requested deadline'
unit=subprocess.run(['systemctl','show',launch['unit'],'-p','MainPID','-p','ActiveState','-p','SubState','-p','ExecMainStatus'],capture_output=True,text=True)
assert 'MainPID=0' in unit.stdout and 'SubState=running' not in unit.stdout, unit.stdout
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
assert not any('localserver_5070/run.py' in line and str(train) in line for line in processes.splitlines()), 'Training process remains'

iterations=[json.loads(line) for line in (train/'iterations.jsonl').read_text().splitlines()]
last=iterations[-1]['iteration']
assert last >= 80
assert [r['iteration'] for r in iterations] == list(range(1,last+1))
assert all(r['seconds']>0 and r['trajectories']==2048 for r in iterations)
with (train/'metrics/metrics.csv').open() as f:
    metrics=list(csv.DictReader(f))
assert {int(r['step']) for r in metrics}==set(range(1,last+1))
assert all(math.isfinite(float(r['value'])) for r in metrics)
replay={int(r['step']):int(float(r['value'])) for r in metrics if r['key']=='buffer/replay_buffer_size'}
assert replay=={i:min(i,64)*2048 for i in range(1,last+1)}

files=checkpoints(train/'checkpoints')
assert 1 <= len(files) <= 10
payload=torch.load(files[-1],map_location='cpu',weights_only=True)
validate_resume_payload(payload)
assert payload['iteration']==last==int(files[-1].stem.split('_')[1])
cfg=payload['train_config']
expected=dict(rollout_batch_size=512,collect_batch_size=2048,mini_batches=4,
    replay_buffer_scale=64,agent_merging_beta=.999,lr_stable_iterations=5000,reg_stable_iterations=500)
assert all(cfg[k]==v for k,v in expected.items())
assert cfg['ppo']['agent_epoch']==cfg['vrpo']['aux_epoch']==4
assert cfg['ppo']['base_lr']==cfg['vrpo']['base_lr']==.0004
assert cfg['vrpo']['replay_buffer_device']=='cpu'
def finite_tensors(value):
    if isinstance(value,torch.Tensor) and value.is_floating_point():
        assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for item in value.values():finite_tensors(item)
    elif isinstance(value,(list,tuple)):
        for item in value:finite_tensors(item)
finite_tensors(payload)
assert not list((train/'checkpoints').glob('*.partial'))

gpu=[]
with (root/'gpu.csv').open() as f:
    for row in csv.reader(f):
        if len(row)!=4:continue
        timestamp=datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
        if finished-result['wall_seconds']-3 <= timestamp <= finished:
            gpu.append(float(row[1]))
peak=max(gpu)
headroom=12227-max(peak,result['peak_reserved_mib']+512)
assert headroom>=12227*.15
assert result['host_min_available_gib']>=6.1
stable=[r for r in iterations if r['iteration']>=65]
assert all(r['new_graphs']==0 for r in stable)
seconds=sum(r['seconds'] for r in stable)
manifest=json.loads((root/'probe-audit.json').read_text())['source_hashes']
for name,digest in manifest.items():
    assert hashlib.sha256(Path('/root/doudizhu',name).read_bytes()).hexdigest()==digest,name
summary=dict(passed=True,finished_at=result['finished_at'],deadline=launch['deadline'],
    completed_iterations=last,total_trajectories=last*2048,full_replay_verified=True,
    trajectories_per_second_full_replay=sum(r['trajectories'] for r in stable)/seconds,
    env_steps_per_second_full_replay=sum(r['env_steps'] for r in stable)/seconds,
    gpu_peak_mib=peak,conservative_gpu_headroom_mib=headroom,
    host_min_available_gib=result['host_min_available_gib'],
    checkpoint=str(files[-1]),checkpoint_bytes=files[-1].stat().st_size,
    checkpoint_sha256=hashlib.sha256(files[-1].read_bytes()).hexdigest(),
    retained_checkpoints=[p.name for p in files],unit_state=unit.stdout,
    gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True),
    checkpoint_and_metrics_finite=True,official_source_unchanged=True)
(root/'training-audit.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
