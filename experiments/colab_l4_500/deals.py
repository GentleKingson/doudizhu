"""Freeze four disjoint protocol-4 deal sets using the pinned PerfectDou generator."""
import argparse
import importlib.util
import pickle
from pathlib import Path
import numpy as np
from marvel.runtime.full_state import sha256, write_json

SETS=[('monitor',1000,100017),('development',5000,200029),
      ('holdout_primary',10000,300043),('holdout_secondary',5000,400059)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--perfectdou',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    generator=args.perfectdou/'generate_eval_data.py'
    spec=importlib.util.spec_from_file_location('frozen_perfectdou_deals',generator)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    global_state=np.random.get_state()
    seen=set()
    manifest={}
    try:
        for name,n,seed in SETS:
            np.random.seed(seed)
            rows=[module.generate() for _ in range(n)]
            for row in rows:
                identity=tuple(tuple(row[k]) for k in ['landlord','landlord_up','landlord_down','three_landlord_cards'])
                if identity in seen:
                    raise ValueError('Duplicate deal within or across independent sets')
                seen.add(identity)
                assert len(row['landlord'])==20 and len(row['landlord_up'])==len(row['landlord_down'])==17
            path=args.output/(name+'.pkl')
            with path.open('xb') as stream:
                pickle.dump(rows,stream,protocol=4)
            manifest[name]={'path':str(path),'num_games_per_role':n,'actual_games_per_model_opponent':2*n,
                'generation_seed':seed,'sha256':sha256(path),'generator_sha256':sha256(generator),
                'opponent_code_sha':'594404922ee3810e2d84b80bb2c2846cb20e5390'}
    finally:
        np.random.set_state(global_state)
    write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':
    main()
