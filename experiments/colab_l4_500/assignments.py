"""Coordinator-only static assignments; workers never claim jobs through Drive locks."""
import argparse
import json
from pathlib import Path
import uuid
from experiments.colab_l4_500.campaign import PLAN
from marvel.runtime.full_state import write_json


def assign(matrix,runtimes,execution_sha):
    if not runtimes or len({r['hostname'] for r in runtimes})!=len(runtimes):
        raise ValueError('Expected distinct actually observed runtime hostnames')
    if any(r['gpu_name']!='NVIDIA L4' or not r.get('resource_stratum') for r in runtimes):
        raise ValueError('Unknown hardware/resource stratum')
    rows=[]
    for job in matrix['queue']['jobs']:
        seed_index=[17,29,43].index(job['training_seed'])
        candidate_index=['A0','A1','A2','A3'].index(job['candidate_id'])
        runtime=runtimes[(candidate_index+seed_index)%len(runtimes)]
        rows.append({'logical_job_id':job['logical_job_id'],'candidate_id':job['candidate_id'],
            'seed':job['training_seed'],'attempt_id':uuid.uuid4().hex,'mode':'train',
            'runtime_hostname':runtime['hostname'],'runtime_name':runtime['name'],
            'resource_stratum':runtime['resource_stratum'],'execution_sha':execution_sha,
            'parent_manifest_sha256':None})
    assert len(rows)==12 and len({r['logical_job_id'] for r in rows})==12
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtimes',type=Path,required=True)
    parser.add_argument('--execution-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=assign(json.loads(PLAN.read_text()),json.loads(args.runtimes.read_text()),args.execution_sha)
    args.output.mkdir(parents=True,exist_ok=False)
    for row in rows:
        write_json(args.output/(row['logical_job_id']+'.json'),row)
    write_json(args.output/'queue.json',rows)


if __name__=='__main__':
    main()
