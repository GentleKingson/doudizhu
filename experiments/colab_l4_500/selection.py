"""Development-only choice; persist one immutable candidate identity before opening holdout."""
import argparse
import json
import math
from pathlib import Path
from marvel.runtime.full_state import sha256
from experiments.colab_l4_500.paired import gate

METRICS=('role_average_win','landlord_win','peasant_team_win')


def quality_pass(result):
    return all(gate(*result['metrics'][key]['ci95'],margin)=='PASS'
               for key,margin in zip(METRICS,(.02,.03,.03)))


def choose(rows, *, floor_proven=False):
    baseline=rows['A0']
    if not baseline['valid_500'] or baseline['total_runtime_seconds_all_attempts']<=0:
        raise ValueError('No valid complete A0 baseline')
    eligible=[]
    diagnostics={}
    for name in ('A1','A2','A3'):
        row=rows[name]
        if (not row['valid_500'] or any(not isinstance(row.get(key),(int,float))
                or not math.isfinite(row[key]) or row[key]<=0
                for key in ('total_runtime_seconds_all_attempts','later_trajectories_per_second'))):
            diagnostics[name]={'valid_and_noninferior':False,'reason':'NOT_PROVEN'}
            continue
        valid=row['valid_500'] and quality_pass(row['development'])
        time_ok=row['total_runtime_seconds_all_attempts']<=.9*baseline['total_runtime_seconds_all_attempts']
        late_ok=row['later_trajectories_per_second']>baseline['later_trajectories_per_second']
        bridge_ok=name!='A3' or quality_pass(row['bridge'])
        a3_cu_ok=name!='A3' or (row.get('actual_cu') is not None and baseline.get('actual_cu') is not None
                                and row['actual_cu']<=baseline['actual_cu'])
        resource_route=valid and time_ok and late_ok and bridge_ok and a3_cu_ok
        quality_route=(valid and row['development']['metrics']['role_average_win']['ci95'][0]>0
                       and row['total_runtime_seconds_all_attempts']<=baseline['total_runtime_seconds_all_attempts']
                       and bridge_ok and a3_cu_ok)
        diagnostics[name]={'valid_and_noninferior':bool(valid),'resource_route':bool(resource_route),
            'quality_route':bool(quality_route),'bridge_pass':bool(bridge_ok),'A3_actual_CU_proven_nonworse':bool(a3_cu_ok)}
        if resource_route or quality_route:
            eligible.append(name)
    eligible.sort(key=lambda name:(rows[name]['total_runtime_seconds_all_attempts'],
                    -rows[name]['development']['metrics']['role_average_win']['difference'],name))
    return {'candidate_id':eligible[0] if eligible and floor_proven else 'A0','diagnostics':diagnostics,
            'conditional_candidates':eligible[:2] if not floor_proven else [],
            'stage':'DEVELOPMENT_CHOICE_ONLY' if floor_proven else 'EARLY_INCONCLUSIVE',
            'holdout_opened':False,
            'interval_scope':'deal uncertainty conditional on the three trained models'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('development',type=Path)
    parser.add_argument('--lock',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.development.read_text())
    floor=data.get('floor_assessment',{})
    proven=(floor.get('status')=='NOT_AT_FLOOR' and bool(floor.get('evidence_path'))
            and sha256(floor['evidence_path'])==floor.get('evidence_sha256'))
    result=choose(data['candidates'],floor_proven=proven)
    result.update(execution_sha=data['execution_sha'],development_manifest_sha256=sha256(args.development),
                  continuation_seed=29)
    # Refuse to replace the choice after any holdout information can have been seen.
    with args.lock.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')


if __name__=='__main__':
    main()
