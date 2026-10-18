"""Paired whole-deal bootstrap conditional on exactly three trained policies."""
import argparse
import json
from pathlib import Path
import numpy as np
from marvel.runtime.full_state import write_json

SEEDS = (17,29,43)
ANALYSIS_SEED = 20261018


def gate(lower, upper, margin):
    if not np.isfinite([lower, upper]).all() or lower > upper:
        return 'INCONCLUSIVE'
    return 'PASS' if lower > -margin else 'FAIL' if upper < -margin else 'INCONCLUSIVE'


def load_roles(paths):
    roles = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        role = payload['left_view_role']
        if role not in ('as_landlord','as_peasants') or role in roles or payload['ci_multiplier'] != 1.0:
            raise ValueError('Expected exactly two role results with original ci=1xSEM')
        ids = payload['game_ids']
        wins = payload['landlord_win_by_game']
        scores = payload['landlord_score_by_game']
        if len(ids) != len(set(ids)) or not len(ids) == len(wins) == len(scores):
            raise ValueError('Duplicate or missing per-deal records')
        if any(x not in (0,1) for x in wins) or not np.isfinite(scores).all():
            raise ValueError('Invalid wins/scores')
        roles[role] = {i:(w,s) for i,w,s in zip(ids,wins,scores)}
    if set(roles) != {'as_landlord','as_peasants'} or roles['as_landlord'].keys() != roles['as_peasants'].keys():
        raise ValueError('Missing paired roles')
    ids = sorted(roles['as_landlord'])
    values = np.array([[roles['as_landlord'][i][0],1-roles['as_peasants'][i][0],
                       roles['as_landlord'][i][1],-roles['as_peasants'][i][1]] for i in ids],dtype=float)
    return ids,values


def paired_bootstrap(candidate, baseline, *, replicates=10000, analysis_seed=ANALYSIS_SEED):
    candidate,baseline = np.asarray(candidate),np.asarray(baseline)
    if candidate.shape != baseline.shape or candidate.ndim != 3 or candidate.shape[0] != 3 or candidate.shape[2] != 4:
        raise ValueError('Expected [3 trained seeds, shared deals, 2 win roles + 2 score roles]')
    if candidate.shape[1] < 2 or not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError('Insufficient/invalid paired samples')
    delta=candidate-baseline
    per_deal=delta.mean(axis=0)
    # Average seeds first; retain both roles, all seeds, candidate and A0 inside each deal block.
    metrics=np.column_stack([per_deal[:,:2].mean(axis=1),per_deal[:,0],per_deal[:,1],
                             per_deal[:,2:].mean(axis=1)])
    rng=np.random.default_rng(analysis_seed)
    samples=np.empty((replicates,4))
    for start in range(0,replicates,100):
        end=min(replicates,start+100)
        indices=rng.integers(0,len(metrics),size=(end-start,len(metrics)))
        samples[start:end]=metrics[indices].mean(axis=1)
    intervals=np.quantile(samples,[.025,.975],axis=0)
    result={'analysis_seed':analysis_seed,'replicates':replicates,'confidence_level':.95,
        'interval_scope':'deal uncertainty conditional on these three trained models; not training-seed population CI',
        'original_ci_label':'1xSEM', 'num_deal_groups':len(metrics),
        'actual_games_per_model_opponent':2*len(metrics),'metrics':{}}
    for index,(name,margin) in enumerate([('role_average_win',.02),('landlord_win',.03),('peasant_team_win',.03),('role_average_score',None)]):
        per_seed=delta[:,:,:2].mean(axis=(1,2)) if index==0 else delta[:,:,index-1].mean(axis=1) if index in (1,2) else delta[:,:,2:].mean(axis=(1,2))
        low,high=intervals[:,index]
        result['metrics'][name]={'difference':float(metrics[:,index].mean()),'ci95':[float(low),float(high)],
            'seed_differences':dict(zip(map(str,SEEDS),map(float,per_seed))),
            'seed_range':[float(per_seed.min()),float(per_seed.max())],
            'gate':gate(low,high,margin) if margin else 'DESCRIPTIVE'}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path,help='Three seeds each with candidate/A0 role result paths and frozen dealset/opponent hashes')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.manifest.read_text())
    if set(data['seeds']) != set(map(str,SEEDS)) or not data['dealset_sha256'] or not data['opponent_sha256']:
        raise ValueError('Missing pairing identity')
    ids=None
    arrays={'candidate':[],'baseline':[]}
    for seed in SEEDS:
        for kind in arrays:
            entry=data['seeds'][str(seed)][kind]
            if entry['dealset_sha256'] != data['dealset_sha256'] or entry['opponent_sha256'] != data['opponent_sha256'] or not entry['model_sha256']:
                raise ValueError('Unpaired evaluation identity')
            current,values=load_roles(entry['role_results'])
            if ids is not None and current != ids:
                raise ValueError('Deal IDs differ across models/seeds')
            ids=current
            arrays[kind].append(values)
    result=paired_bootstrap(arrays['candidate'],arrays['baseline'])
    result['input_manifest']=data
    write_json(args.output,result)


if __name__=='__main__':
    main()
