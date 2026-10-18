"""Resolved configs and guarded single-runtime attempts for the L4 campaign."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
import shutil
import socket
from pathlib import Path
import subprocess
import sys
import time
import uuid

import torch

from marvel.config import TrainConfig
from marvel.runtime.full_state import (canonical_hash, full_config, scientific_config, sha256,
                                       verify_generation, write_json)
from marvel.training.vrpo import VRPOTrainer
from marvel.utils import configure_torch

PLAN = Path(__file__).parent / 'plan' / 'experiment_matrix.json'
REQUIRED_GATES = {'replay_wraparound', 'native_journal_replay', 'sampler_leftovers',
    'short_training_fork', 'negative_restore', 'full_capacity_save_readback_fixture',
    'drive_publish_restore', 'frozen_opponents_and_dealsets', 'resource_admission'}


def merge(left, right):
    result = dict(left)
    for key,value in right.items():
        result[key] = merge(result[key],value) if isinstance(value,dict) else value
    return result


def resolved(candidate, seed, output):
    matrix = json.loads(PLAN.read_text())
    choice = next(x for x in matrix['candidates'] if x['candidate_id']==candidate)
    if seed not in choice['training_seeds']:
        raise ValueError('Undeclared training seed')
    base = asdict(TrainConfig(run_name='resolved',logging_dir=str(output)))
    config = merge(merge(base,matrix['common_training_config']),choice['overrides'])
    config.update(seed=seed, run_name=f'{candidate}-s{seed}', logging_dir=str(output),
                  full_state_checkpoints=True, checkpoint_wallclock_seconds=900.)
    return full_config(config)


def check_ready(path, candidate):
    ready = json.loads(Path(path).read_text())
    current = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if not ready.get('training_ready') or ready.get('execution_sha') != current:
        raise ValueError('No verified READY execution SHA')
    if set(ready.get('gates',{})) != REQUIRED_GATES:
        raise ValueError('Missing readiness gate')
    for name,gate in ready['gates'].items():
        if gate['status'] != 'PASS' or sha256(gate['evidence_path']) != gate['evidence_sha256']:
            raise ValueError(f'Unverified gate {name}')
    if ready.get('candidates',{}).get(candidate) != 'READY':
        raise ValueError(f'{candidate} not admitted')
    if subprocess.check_output(['git','diff','--name-only'],text=True).strip():
        raise ValueError('Execution checkout contains tracked modifications')
    return ready


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','smoke','train','resume'])
    parser.add_argument('--candidate',choices=['A0','A1','A2','A3'],default='A0')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--ready',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--target',type=int,default=40000)
    parser.add_argument('--drive-root',type=Path,default=Path('/content/drive/MyDrive/doudizhu/l4-vrpo-500-20261018-v1'))
    parser.add_argument('--assignment',type=Path)
    args=parser.parse_args()
    torch.set_num_interop_threads(1)
    attempt=uuid.uuid4().hex
    assignment=None
    if args.mode in ('train','resume'):
        if args.assignment is None:
            parser.error('Formal execution requires an immutable per-attempt --assignment')
        assignment=json.loads(args.assignment.read_text())
        if (assignment['candidate_id'],assignment['seed'],assignment['mode'],assignment['runtime_hostname']) != (
                args.candidate,args.seed,args.mode,socket.gethostname()):
            raise ValueError('This job/attempt is assigned to another runtime or mode')
        attempt=uuid.UUID(assignment['attempt_id']).hex
    output=args.root/'runs'/args.candidate/f's{args.seed}'/attempt
    ready=None
    if args.mode in ('train','resume'):
        if args.ready is None:
            parser.error('Formal execution requires --ready with evidence')
        ready=check_ready(args.ready,args.candidate)
        if assignment['execution_sha']!=ready['execution_sha']:
            raise ValueError('Assignment source SHA mismatch')
        from experiments.colab_l4_500.drive import verify_mount
        verify_mount(args.drive_root)
    if args.mode=='resume':
        if args.resume is None:
            parser.error('--resume requires a specific verified generation')
        verify_generation(args.resume,require_remote=True)
        if assignment.get('parent_manifest_sha256')!=sha256(args.resume/'manifest.json'):
            raise ValueError('Assignment must name this exact verified resume parent')
        cfg=full_config(json.loads((args.resume/'config.json').read_text()))
        if cfg.seed != args.seed:
            raise ValueError('Resume seed mismatch')
        expected=resolved(args.candidate,args.seed,output)
        if scientific_config(asdict(cfg))!=scientific_config(asdict(expected)):
            raise ValueError('Parent does not match the declared campaign candidate')
        cfg.resume_from=str(args.resume)
        cfg.total_iterations=args.target
        cfg.logging_dir=str(output)
    else:
        if assignment is not None and assignment.get('parent_manifest_sha256') is not None:
            raise ValueError('Initial runs must start from initialization')
        cfg=resolved(args.candidate,args.seed,output)
        if args.mode=='smoke':
            cfg.total_iterations=3
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'resolved_config.json',asdict(cfg))
    write_json(output/'attempt.json',{'attempt_id':attempt,'candidate_id':args.candidate,'seed':args.seed,
        'mode':args.mode,'config_sha256':canonical_hash(asdict(cfg)),
        'scientific_config_sha256':canonical_hash(scientific_config(asdict(cfg))),
        'assignment':assignment,
        'training_ready':args.mode in ('train','resume')})
    print('ATTEMPT',str(output),flush=True)
    if args.mode=='prepare':
        return
    if os.environ.get('OMP_NUM_THREADS') != '2' or os.environ.get('MKL_NUM_THREADS') != '2':
        raise ValueError('Campaign requires explicit OMP_NUM_THREADS=2 MKL_NUM_THREADS=2')
    if os.environ.get('PYTHONHASHSEED') != str(args.seed):
        raise ValueError('PYTHONHASHSEED must be set before process start to the training seed')
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_default_dtype(torch.float32)
    configure_torch(cfg)
    os.environ['MARVEL_STOP_AFTER_BOUNDARY']=str(args.root/'STOP_AFTER_BOUNDARY')
    from marvel.training.workflow import TrainingWorkflow
    runtimes=[]

    def factory(config):
        runtime=TrainingWorkflow(config,game=config.game)
        runtimes.append(runtime)
        if args.mode in ('train','resume'):
            runtime.expected_initial_models=ready['initial_model_hashes'][str(args.seed)]
            from experiments.colab_l4_500.drive import stage_generation
            remote=args.drive_root/'runs'/args.candidate/f's{args.seed}'/attempt
            remote.mkdir(parents=True,exist_ok=False)
            save=runtime.checkpoint_policy.save

            def publish(state):
                save(state)
                generation=runtime.last_full_generation
                record=stage_generation(generation,remote/'checkpoints')
                for name in ('resolved_config.json','attempt.json','initial_models.json','metrics.jsonl','iterations.jsonl'):
                    if (output/name).exists():
                        shutil.copyfile(output/name,remote/name)
                marker=Path(record['remote_directory'])/'COMPLETE.json'
                print('WAITING_INDEPENDENT_DRIVE_VERIFICATION '+str(marker),flush=True)
                while not marker.exists():
                    time.sleep(5)
                complete=json.loads(marker.read_text())
                if (complete.get('remote_verified') is not True or complete['manifest_sha256']!=record['manifest_sha256']
                        or complete['archive_sha256']!=record['archive_sha256']):
                    raise ValueError('Invalid remote COMPLETE receipt')
                if state.iteration==cfg.replay_buffer_scale:
                    layout=json.loads((generation/'replay/layout.json').read_text())
                    for name,field in layout['fields'].items():
                        for part in field['parts']:
                            chunk=torch.load(generation/'replay'/part['file'],map_location='cpu',weights_only=True)
                            if not torch.equal(chunk,runtime.full_replay.storage[name][part['start']:part['end']]):
                                raise ValueError('Real filled replay readback mismatch')
                            del chunk
                    evidence={'status':'PASS','iteration':state.iteration,'real_data':True,
                              'manifest_sha256':record['manifest_sha256'],'drive_complete':complete}
                    write_json(output/'real_filled_replay_validation.json',evidence)
                    write_json(remote/'real_filled_replay_validation.json',evidence)

            runtime.checkpoint_policy.save=publish
        return runtime
    # flock is VM-local; assignments are static and workers never coordinate through Drive locks.
    with open('/tmp/doudizhu-l4-vrpo-500.worker.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['MARVEL_METRICS_JSONL']=str(output/'metrics.jsonl')
        started=time.monotonic()
        try:
            from experiments.colab_l4_500.resources import ResourceMonitor
            with ResourceMonitor(output/'resources.json'):
                VRPOTrainer(cfg,runtime_factory=factory).run()
        except BaseException as exc:
            write_json(output/'outcome.json',{'status':'FAILED','error':str(exc),'elapsed_seconds':time.monotonic()-started})
            raise
        completed=runtimes[0].current_iteration
        write_json(output/'outcome.json',{'status':'BUDGET_PAUSED' if completed<cfg.total_iterations else 'LOCAL_TRAINING_FINISHED_METADATA_UPLOAD_PENDING',
            'actual_completed_iteration':completed,
            'target_iteration':cfg.total_iterations, 'elapsed_seconds':time.monotonic()-started,
            'drive_complete':False})
        if args.mode in ('train','resume'):
            remote=args.drive_root/'runs'/args.candidate/f's{args.seed}'/attempt
            for name in ('resolved_config.json','attempt.json','initial_models.json','metrics.jsonl','iterations.jsonl',
                         'resources.json','outcome.json'):
                if (output/name).exists():
                    shutil.copyfile(output/name,remote/name)


if __name__=='__main__':
    main()
