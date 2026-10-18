"""Verify the actual initial Actor/Critic tensors for all 12 resolved configurations."""
import argparse
from pathlib import Path
import torch
from experiments.colab_l4_500.campaign import resolved
from marvel.model import Agent,CentralCritic
from marvel.runtime.full_state import model_hash,write_json
from marvel.utils import set_global_seed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=[17,29,43],required=True)
    args=parser.parse_args()
    torch.set_num_interop_threads(1)
    result={}
    for seed in (args.seed,):
        reference=None
        for candidate in ('A0','A1','A2','A3'):
            cfg=resolved(candidate,seed,'/tmp/initial-pairing')
            set_global_seed(seed)
            actor=Agent(cfg,enable_value_head=False)
            critic=CentralCritic(cfg,output_mode='qvalue')
            actual={'seed':seed,'actor_sha256':model_hash(actor),'critic_sha256':model_hash(critic)}
            if reference is None:
                reference=actual
            assert actual==reference, (candidate,seed)
            del actor,critic
        result[str(seed)]=reference
    write_json(args.output,{'status':'PASS','initial_model_hashes':result})


if __name__=='__main__':
    main()
