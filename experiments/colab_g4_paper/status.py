from pathlib import Path
import json,subprocess
root=Path('/content/g4-paper')
for name in ('done.json','results/hardware.json','results/summary.json','results/failure.txt'):
 p=root/name
 if p.exists():
  if name.endswith('hardware.json'):
   h=json.loads(p.read_text()); print('HARDWARE',h['gpu'],h['cpu_count'])
  else: print(name,p.read_text())
p=root/'search.log'
if p.exists(): print('SEARCH_TAIL',p.read_text()[-150:])
r=root/'results/results.json'
if r.exists():
 results=json.loads(r.read_text())
 print('COMPLETED',len(results))
 for x in results[-1:]: print({k:x.get(k) for k in ('label','config','eligible','trajectories_per_second','headroom_mib','host_min_available_mib','wall_seconds')})
folders=sorted((root/'results').glob('screen-*'))+sorted((root/'results').glob('verify-*'))
for folder in folders:
 if not (folder/'result.json').exists() and (folder/'train.log').exists(): print('ACTIVE',folder.name,(folder/'train.log').read_text()[-140:])
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True))
