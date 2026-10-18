from dataclasses import asdict
import numpy as np
import pytest
from experiments.colab_l4_500.campaign import resolved, check_ready
from experiments.colab_l4_500.paired import gate, paired_bootstrap


def test_matrix_single_differences(tmp_path):
    configs=[asdict(resolved(c,17,tmp_path)) for c in ('A0','A1','A2','A3')]
    base=configs[0]
    for index,path,value in [(1,'replay_buffer_scale',32),(2,'vrpo',2),(3,'collect_batch_size',256)]:
        candidate=configs[index]
        candidate['run_name']=base['run_name']
        if path=='vrpo':
            assert candidate[path]['aux_epoch']==value
            candidate[path]['aux_epoch']=base[path]['aux_epoch']
        else:
            assert candidate[path]==value
            candidate[path]=base[path]
        assert candidate==base
    assert base['agent_merging_beta']==.999 and base['lr_stable_iterations']==5000


def test_paired_and_strict_margins():
    baseline=np.zeros((3,20,4))
    candidate=baseline.copy()
    candidate[:,:,0]=1
    candidate[:,:,2]=2
    result=paired_bootstrap(candidate,baseline,replicates=200)
    assert result['metrics']['role_average_win']['ci95']==[.5,.5]
    assert result['metrics']['landlord_win']['ci95']==[1.,1.]
    assert result['metrics']['peasant_team_win']['ci95']==[0.,0.]
    assert result['metrics']['role_average_score']['ci95']==[1.,1.]
    assert gate(-.02,.1,.02)=='INCONCLUSIVE'
    assert gate(-.1,-.02,.02)=='INCONCLUSIVE'
    assert gate(-.1,-.03,.02)=='FAIL'
    with pytest.raises(ValueError):
        paired_bootstrap(candidate[:2],baseline[:2])


def test_plan_is_not_ready():
    with pytest.raises(ValueError):
        check_ready('experiments/colab_l4_500/plan/experiment_matrix.json','A0')


def test_static_queue_has_unique_jobs_and_rotates_actual_runtimes():
    import json
    from experiments.colab_l4_500.assignments import assign
    from experiments.colab_l4_500.campaign import PLAN
    hosts=[{'hostname':f'host{i}','gpu_name':'NVIDIA L4','resource_stratum':'same','name':str(i)} for i in range(4)]
    rows=assign(json.loads(PLAN.read_text()),hosts,'execution')
    assert len({r['attempt_id'] for r in rows})==12
    assert rows[0]['runtime_hostname']=='host0'
    assert rows[4]['runtime_hostname']=='host1'
    assert rows[8]['runtime_hostname']=='host2'
    with pytest.raises(ValueError):
        assign(json.loads(PLAN.read_text()),hosts+hosts[:1],'execution')


def test_development_choice_and_A3_bridge_CU_gate():
    import copy
    from experiments.colab_l4_500.selection import choose
    quality={'metrics':{key:{'gate':'PASS','ci95':[-.01,.02],'difference':0.0} for key in ('role_average_win','landlord_win','peasant_team_win')}}
    base={'valid_500':True,'total_runtime_seconds_all_attempts':1000.,'later_trajectories_per_second':10.,
          'actual_cu':1.,'development':quality,'bridge':quality}
    rows={name:copy.deepcopy(base) for name in ('A0','A1','A2','A3')}
    rows['A2']['total_runtime_seconds_all_attempts']=900.
    rows['A2']['later_trajectories_per_second']=11.
    assert choose(rows,floor_proven=True)['candidate_id']=='A2'
    assert choose(rows)['stage']=='EARLY_INCONCLUSIVE'
    rows['A3']['total_runtime_seconds_all_attempts']=800.
    rows['A3']['later_trajectories_per_second']=20.
    rows['A3']['actual_cu']=None
    assert choose(rows,floor_proven=True)['candidate_id']=='A2'
    rows['A3']['actual_cu']=.8
    assert choose(rows,floor_proven=True)['candidate_id']=='A3'
    rows['A3']['bridge']['metrics']['role_average_win']['ci95']=[-.05,.02]
    assert choose(rows,floor_proven=True)['candidate_id']=='A2'


def test_evaluation_counts_two_roles_and_rejects_missing_deals(tmp_path):
    import json
    from experiments.colab_l4_500.evaluate import verify_raw_counts
    paths=[]
    for role in ('as_landlord','as_peasants'):
        p=tmp_path/(role+'.json')
        p.write_text(json.dumps({'left_view_role':role,'ci_multiplier':1.,'game_ids':[1,2],
                                'landlord_win_by_game':[0,1],'landlord_score_by_game':[-2,2]}))
        paths.append(p)
    assert verify_raw_counts(paths,2)==4
    with pytest.raises(ValueError):
        verify_raw_counts(paths,3)
    with pytest.raises(ValueError):
        verify_raw_counts(paths+paths,2)
