"""Reusable Linux/systemd launcher. Preview with --dry-run; no fixed calendar deadline."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import uuid


def build_plan(argv=None):
    repo=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-minutes',type=float,default=120)
    parser.add_argument('--save-every-minutes',type=float,default=30)
    parser.add_argument('--keep-checkpoints',type=int,default=10)
    parser.add_argument('--resume-from',type=Path,default=repo/'logs/localserver-5070-2026-10-12/train/checkpoints/ckpt_576.pt')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    for name in ('training_minutes','save_every_minutes'):
        value=getattr(args,name)
        if not math.isfinite(value) or value<=0:parser.error(name+' must be positive and finite')
    if args.keep_checkpoints<1:parser.error('--keep-checkpoints must be positive')
    resume=args.resume_from.resolve()
    if not resume.is_file():parser.error('Checkpoint not found: '+str(resume))
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
    out=(args.output or repo/'logs'/('rtx5070-'+stamp)).resolve()
    if out.exists():parser.error('Output path already exists; use a new directory to preserve prior runs')
    unit='vrpo5070-'+stamp
    env=dict(PYTHONPATH=str(repo),WANDB_DISABLED='true',PYTORCH_ALLOC_CONF='expandable_segments:True',
        TORCHINDUCTOR_CACHE_DIR=str(repo/'logs/localserver-5070-2026-10-12/inductor-cache'),
        TORCHINDUCTOR_COMPILE_THREADS='2',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    # Allow initialization/compilation and a final save beyond the iteration-loop duration.
    maximum=math.ceil(args.training_minutes*60)+900
    command=['systemd-run','--unit='+unit,'--property=WorkingDirectory='+str(repo),
        '--property=RuntimeMaxSec='+str(maximum),'--property=TimeoutStopSec=180',
        '--property=StandardOutput=append:'+str(out/'train.log'),'--property=StandardError=inherit']
    command+=['--setenv='+key+'='+value for key,value in env.items()]
    command+=['/usr/bin/flock','--nonblock',str(repo/'logs/rtx5070-training.lock'),
        str(repo/'.venv/bin/python'),'-u','-m','experiments.localserver_5070.run',
        '--output',str(out/'train'),'--chunk','32','--threads','8','--rollout','512',
        '--training-minutes',str(args.training_minutes),'--save-every-minutes',str(args.save_every_minutes),
        '--keep-checkpoints',str(args.keep_checkpoints),'--resume-from',str(resume)]
    return args,dict(unit=unit,output=str(out),resume_from=str(resume),
        training_minutes=args.training_minutes,save_every_minutes=args.save_every_minutes,
        keep_checkpoints=args.keep_checkpoints,runtime_max_seconds=maximum,command=command)


def main():
    args,plan=build_plan()
    if args.dry_run:
        print(json.dumps(plan,indent=2)); return
    out=Path(plan['output']); out.mkdir(parents=True,exist_ok=False)
    (out/'launch.json').write_text(json.dumps(plan,indent=2)+'\n')
    subprocess.run(plan['command'],check=True)
    plan['launched_at']=datetime.now(timezone.utc).isoformat()
    (out/'launch.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2))
    print('Status: systemctl status '+plan['unit'])
    print('Stop and save: systemctl stop '+plan['unit'])


if __name__=='__main__':main()
