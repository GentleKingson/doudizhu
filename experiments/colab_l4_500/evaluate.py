"""Run the existing evaluator with frozen assets, exact 2N billing and retained per-deal results."""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import uuid
from marvel.runtime.full_state import sha256, verify_generation, write_json

STAGES={'monitor':('monitor',1000,250,'primary'),
        'development':('development',5000,500,'primary'),
        'bridge':('development',5000,250,'primary'),
        'holdout_primary':('holdout_primary',10000,500,'primary'),
        'holdout_secondary':('holdout_secondary',5000,500,'secondary'),
        'pilot':('monitor',128,None,'primary')}


def verify_raw_counts(paths,n):
    roles={key:{} for key in ('as_landlord','as_peasants')}
    for path in paths:
        data=json.loads(path.read_text())
        role=data['left_view_role']
        if data['ci_multiplier']!=1.0 or role not in roles:
            raise ValueError('Unexpected role/CI definition')
        ids,wins,scores=(data[k] for k in ('game_ids','landlord_win_by_game','landlord_score_by_game'))
        if not len(ids)==len(wins)==len(scores):
            raise ValueError('Missing per-game fields')
        for game_id,win,score in zip(ids,wins,scores):
            if game_id in roles[role] or win not in (0,1) or not math.isfinite(score):
                raise ValueError('Duplicated deal or invalid win')
            roles[role][game_id]=(win,score)
    if any(set(values)!=set(range(1,n+1)) for values in roles.values()):
        raise ValueError('Evaluation must contain exactly N games in each of two roles')
    return 2*n


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--stage',choices=STAGES,required=True)
    parser.add_argument('--candidate',choices=['A0','A1','A2','A3'],required=True)
    parser.add_argument('--seed',type=int,choices=[17,29,43],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection-lock',type=Path)
    args=parser.parse_args()
    manifest=verify_generation(args.generation,require_remote=args.stage!='pilot')
    config=json.loads((args.generation/'config.json').read_text())
    if config['seed']!=args.seed:
        raise ValueError('Evaluation training-seed identity mismatch')
    name,n,iteration,opponent_slot=STAGES[args.stage]
    if iteration is not None and manifest['completed_iteration']!=iteration:
        raise ValueError('Wrong evaluation checkpoint iteration')
    if args.stage=='bridge' and args.candidate!='A3':
        raise ValueError('Only A3@250 belongs to the 64k bridge')
    if args.stage.startswith('holdout'):
        if args.selection_lock is None:
            raise ValueError('Development selection must be locked before holdout')
        locked=json.loads(args.selection_lock.read_text())
        if locked.get('stage')!='DEVELOPMENT_CHOICE_ONLY' or locked.get('execution_sha')!=manifest['provenance']['execution_sha']:
            raise ValueError('Holdout requires a conclusive locked development choice on this execution SHA')
        if args.candidate not in {'A0',locked['candidate_id']}:
            raise ValueError('Changing candidate after opening holdout is prohibited')
    assets=json.loads((args.assets/'opponents.json').read_text())
    opponent=assets[opponent_slot]
    if opponent['status']!='FROZEN_VALIDATED':
        raise ValueError('Opponent has not passed inference validation')
    repo=Path(__file__).resolve().parents[2]
    for relative,expected in opponent['files'].items():
        if sha256(repo/relative)!=expected:
            raise ValueError('Opponent code/model hash changed')
    datasets=json.loads((args.assets/'deals/manifest.json').read_text())
    dataset=args.assets/'deals'/f'{name}.pkl'
    if sha256(dataset)!=datasets[name]['sha256'] or dataset.read_bytes()[:2]!=b'\x80\x04':
        raise ValueError('Noncanonical evaluation dataset')
    with dataset.open('rb') as stream:
        if len(pickle.load(stream))<n:
            raise ValueError('Insufficient frozen deals')
    output=args.output/args.stage/args.candidate/f's{args.seed}'/uuid.uuid4().hex
    output.mkdir(parents=True,exist_ok=False)
    local_data=output/'eval_data.pkl'
    shutil.copyfile(dataset,local_data)
    model=args.generation/'train_state.pt'
    command=[sys.executable,'-m','evaluations.run','doudizhu',f'marvel={model}?action_temp=0',opponent['policy_spec'],
        '--num-games',str(n),'--batch-size','128','--workers-per-gpu','1',
        '--eval-data',str(local_data),'--logdir',str(output/'raw'),'--no-wandb']
    record={'stage':args.stage,'candidate_id':args.candidate,'training_seed':args.seed,
        'completed_iteration':manifest['completed_iteration'],'model_sha256':sha256(model),
        'opponent':opponent,'dealset_sha256':sha256(dataset),'num_games_per_role':n,
        'expected_actual_games':2*n,'source_sha':manifest['provenance']['execution_sha'],
        'original_ci_label':'1xSEM','command':command}
    write_json(output/'evaluation_identity.json',record)
    env={**os.environ,'PYTHON37':'/content/perfectdou37/bin/python','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    started=time.monotonic()
    with (output/'evaluator.log').open('w') as log:
        result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    if sha256(local_data)!=record['dealset_sha256']:
        raise ValueError('Evaluator mutated the canonical local dataset')
    actual_games=verify_raw_counts((output/'raw/jobs').glob('*/summary.json'),n) if result.returncode==0 else None
    write_json(output/'outcome.json',{'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-started,
                                     'expected_actual_games':2*n,'verified_actual_games':actual_games,
                                     'status':'PASS' if result.returncode==0 else 'FAILED'})
    print('EVALUATION_OUTPUT',str(output),flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
