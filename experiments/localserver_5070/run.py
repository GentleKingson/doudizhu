"""Run upstream VRPO with lossless memory adaptations for a configurable duration."""
import argparse
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
import torch
from torch._dynamo.utils import counters
from marvel.config import TrainConfig
from marvel.model import CentralCritic
from marvel.training.workflow import TrainingWorkflow
from marvel.training.vrpo import VRPOTrainer
from marvel.utils import configure_torch
from experiments.localserver_5070.memory_adapter import ChunkedModel, install_replay
from experiments.colab_a100_paper.timed_training import publish_checkpoint, validate_resume_payload

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--chunk',type=int,default=32)
p.add_argument('--rollout',type=int,default=512)
p.add_argument('--threads',type=int,default=8)
p.add_argument('--iterations',type=int,default=5)
p.add_argument('--warmup',type=int,default=2)
p.add_argument('--training-minutes',type=float,default=None)
p.add_argument('--save-every-minutes',type=float,default=30)
p.add_argument('--keep-checkpoints',type=int,default=10)
p.add_argument('--resume-from',default=None)
args=p.parse_args()
assert args.chunk > 0 and args.threads > 1
assert math.isfinite(args.save_every_minutes) and args.save_every_minutes>0
assert args.keep_checkpoints>0 and args.iterations>0 and args.warmup>=0
training=args.training_minutes is not None
if training:assert math.isfinite(args.training_minutes) and args.training_minutes>0
if args.resume_from and not training:p.error('--resume-from requires --training-minutes')
out=Path(args.output).resolve()
if out.exists() and any(out.iterdir()):p.error('Choose a new output directory; existing records are preserved')
torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
cfg=TrainConfig(algo_name='vrpo', device='cuda', rollout_batch_size=args.rollout,
    collect_batch_size=2048,mini_batches=4,replay_buffer_scale=64,
    agent_merging_beta=.999,lr_stable_iterations=5000,reg_stable_iterations=500,
    total_iterations=40000 if training else args.iterations,
    checkpoint_interval=-1,evaluation_interval=-1,param_stats_interval=-1,
    seed=0,logging_dir=str(out/'metrics'))
cfg.vrpo.replay_buffer_device='cpu'
if args.resume_from:
    args.resume_from=str(Path(args.resume_from).resolve())
    payload=torch.load(args.resume_from,map_location='cpu',weights_only=True)
    validate_resume_payload(payload)
    # The upstream deserializer restores inference metadata only, not learning schedules.
    saved=dict(payload['train_config'])
    for field in fields(cfg):
        template=getattr(cfg,field.name)
        if is_dataclass(template):saved[field.name]=type(template)(**saved[field.name])
    cfg=TrainConfig(**saved)
    assert asdict(cfg)==payload['train_config'], 'Training recipe did not round-trip exactly'
    assert cfg.collect_batch_size==2048 and cfg.mini_batches==4 and cfg.replay_buffer_scale==64
    assert cfg.rollout_batch_size==args.rollout and cfg.vrpo.replay_buffer_device=='cpu'
    assert 0<int(payload['iteration'])<cfg.total_iterations, 'Checkpoint has reached its configured iteration limit'
    cfg.resume_from=args.resume_from; cfg.resume_optimizer=True
    cfg.device='cuda'; cfg.logging_dir=str(out/'metrics')
    cfg.checkpoint_interval=cfg.evaluation_interval=cfg.param_stats_interval=-1
    del payload
configure_torch(cfg)
install_replay()
out.mkdir(parents=True,exist_ok=True)
(out/'config.json').write_text(json.dumps(asdict(cfg),indent=2))
(out/'settings.json').write_text(json.dumps(vars(args),indent=2))
def host_memory():
    fields=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0])*1024, int(fields['MemTotal'].split()[0])*1024
host_min=[host_memory()[0]]
stop=threading.Event()
def monitor():
    while not stop.wait(.5):
        available,total=host_memory(); host_min[0]=min(host_min[0],available)
        if available < total*.10:
            print('HOST_MEMORY_GUARD',flush=True); os._exit(75)
threading.Thread(target=monitor,daemon=True).start()
blocks=[]
class Workflow(TrainingWorkflow):
    def __init__(self,c):
        super().__init__(c,game=c.game,checkpoint_dir=out/'checkpoints')
        self.stopping=False; self.last_save=time.monotonic(); self.last_iteration=0
        self.iter_seconds=[]
    def load_checkpoint_payload(self,checkpoint_path,*,device):
        return super().load_checkpoint_payload(checkpoint_path,device=torch.device('cpu'))
    def prepare(self,model,**kw):
        return ChunkedModel(super().prepare(model,**kw),args.chunk,isinstance(model,CentralCritic))
    def checkpoint_due(self):
        return training and time.monotonic()-self.last_save>=args.save_every_minutes*60
    def save_checkpoint(self,state):
        if not training or self.last_iteration==state.iteration:return
        path=publish_checkpoint(self.checkpoint_manager,state,self.checkpoint_dir,args.keep_checkpoints)
        self.last_iteration=state.iteration; self.last_save=time.monotonic()
        print('CHECKPOINT',str(path),flush=True)
    def log_metrics(self):
        assert all(math.isfinite(v) for v in self._metric_buffer.values() if isinstance(v,(int,float)))
        completed=self.current_iteration-self.start_iteration+1
        assert self._metric_buffer['buffer/replay_buffer_size']==min(completed,64)*2048
        super().log_metrics()
    def iterations(self):
        self.sync_step_totals_from_sampler()
        def request_stop(signum,frame):self.stopping=True
        old={sig:signal.signal(sig,request_stop) for sig in (signal.SIGTERM,signal.SIGINT)}
        began=time.monotonic(); self.last_save=began
        try:
            for it in super().iterations():
                torch.cuda.synchronize(); start=time.perf_counter()
                env=self.env_steps_total
                game=self.game_steps_total
                graphs=counters['stats']['unique_graphs']
                yield it
                torch.cuda.synchronize(); seconds=time.perf_counter()-start
                self.iter_seconds.append(seconds)
                row=dict(iteration=it,seconds=seconds,trajectories=2048,
                    env_steps=self.env_steps_total-env,game_steps=self.game_steps_total-game,
                    new_graphs=counters['stats']['unique_graphs']-graphs)
                with (out/'iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if len(self.iter_seconds)>args.warmup:blocks.append(row)
                print('ITERATION',json.dumps(row),flush=True)
                # Check after the completed iteration so checkpoint numbering stays exact.
                if training and (self.stopping or time.monotonic()-began>=args.training_minutes*60):
                    self.save_checkpoint(self._state)
                    break
            if training:self.save_checkpoint(self._state)
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)

started=time.time()
try:
    VRPOTrainer(cfg,runtime_factory=Workflow).run()
    result=dict(success=True,blocks=blocks,torch_version=torch.__version__,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        host_min_available_gib=host_min[0]/2**30,wall_seconds=time.time()-started,
        finished_at=datetime.now().astimezone().isoformat())
    if blocks:
        result['trajectories_per_second']=sum(b['trajectories'] for b in blocks)/sum(b['seconds'] for b in blocks)
        result['new_graphs']=sum(b['new_graphs'] for b in blocks)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print('RESULT',json.dumps(result),flush=True)
finally:
    stop.set()
