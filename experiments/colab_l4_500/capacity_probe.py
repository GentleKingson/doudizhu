"""Full CPU I/O fixture from repeated genuine trajectories; never a training/compression result."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from marvel.buffer import BufferConfig, TrajectoryBuffer, trajectory_field_specs
from marvel.config import TrainConfig
from marvel.runtime.full_state import load_buffer, save_buffer, write_json
from experiments.colab_l4_500.resources import ResourceMonitor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True,help='Verified genuine-trajectory generation')
    parser.add_argument('--capacity',type=int,choices=[4096,8192,16384],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=BufferConfig(game=TrainConfig(run_name='fixture').game,device='cpu',
                        field_specs=trajectory_field_specs(TrainConfig(run_name='fixture').game))
    source=json.loads((args.source/'replay/layout.json').read_text())
    genuine={name:torch.load(args.source/'replay'/field['parts'][0]['file'],weights_only=True)[:1].clone()
             for name,field in source['fields'].items()}
    assert genuine['num_steps'][0]>0
    with ResourceMonitor(args.output/'resources.json',cuda=False):
        buffer=TrajectoryBuffer(args.capacity,config)
        for name,tensor in buffer.storage.items():
            for start in range(0,args.capacity,32):
                tensor[start:start+32].copy_(genuine[name].expand_as(tensor[start:start+32]))
        buffer._size=args.capacity
        buffer._ptr=0
        started=time.monotonic()
        save_buffer(buffer,args.output/'replay',args.capacity)
        save_seconds=time.monotonic()-started
        del buffer
        gc.collect()
        started=time.monotonic()
        restored=TrajectoryBuffer(args.capacity,config)
        load_buffer(restored,args.output/'replay')
        load_seconds=time.monotonic()-started
        for name,tensor in restored.storage.items():
            for start in range(0,args.capacity,32):
                assert torch.equal(tensor[start:start+32],genuine[name].expand_as(tensor[start:start+32]))
        assert restored._size==args.capacity and restored._ptr==0
    write_json(args.output/'result.json',{'status':'PASS','capacity':args.capacity,
        'save_seconds':save_seconds,'load_seconds':load_seconds,
        'serialized_bytes':sum(p.stat().st_size for p in (args.output/'replay').rglob('*') if p.is_file()),
        'fixture':'Repeated genuine trajectories; not training, real replay compression or throughput evidence'})


if __name__=='__main__':
    main()
