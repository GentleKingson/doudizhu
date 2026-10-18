import gc
import json
import os
from pathlib import Path
import torch
from marvel.config import TrainConfig
from marvel.model import Agent, CentralCritic
from marvel.runtime.sampler import Sampler, SamplerConfig
from marvel.buffer import TrajectoryBuffer, BufferConfig, trajectory_field_specs
from marvel.utils import configure_torch
from memory_adapter import PackedReplay, ChunkedModel, KEYS

torch.set_num_threads(6)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cfg = TrainConfig(algo_name='vrpo', use_compile=False)
production = os.environ.get('PRODUCTION_NUMERIC') == '1'
if production:
    configure_torch(cfg)
agent = Agent(cfg, enable_value_head=False).cuda()
sampler = Sampler(config=SamplerConfig(game=cfg.game, device='cuda', rollout_batch_size=8, collect_batch_size=8))
roll = sampler.collect(agent_model=agent)
results = {}
for label, model, critic in [('agent', agent, False), ('critic', CentralCritic(cfg, output_mode='qvalue').cuda(), True)]:
    model.zero_grad(set_to_none=True)
    execution = torch.compile(model) if production else model
    out = execution(rollout=roll)
    value = out.qvalues if critic else out.avail_probs
    target = torch.randn_like(value)
    reference_mask = None if critic else out.mask.clone()
    reference = value.detach().clone()
    (value * target).mean().backward()
    grads = {n: p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    out = ChunkedModel(execution, 2, critic)(rollout=roll)
    if not critic:
        assert torch.equal(out.mask,reference_mask)
    value = out.qvalues if critic else out.avail_probs
    # TF32 kernels selected for different batch sizes have coarser rounding than strict FP32.
    torch.testing.assert_close(value, reference, rtol=1e-3 if production else 2e-4,
                               atol=1e-5 if production else 2e-6)
    (value * target).mean().backward()
    diff = sum((p.grad-grads[n]).double().square().sum().item() for n,p in model.named_parameters() if n in grads)
    norm = sum(g.double().square().sum().item() for g in grads.values())
    relative = (diff/max(norm,1e-30))**.5
    assert relative < 2e-4, (label, relative)
    results[label] = dict(output_max_abs=(value-reference).abs().max().item(), gradient_relative_l2=relative)

bc = BufferConfig(game=cfg.game, device='cpu', field_specs=trajectory_field_specs(cfg.game))
plain, packed = TrajectoryBuffer(16,bc), PackedReplay(16,bc)
for batch_id in range(3):
    batch=roll.clone()
    batch['rewards']=batch['rewards']+batch_id
    plain.add(batch); packed.add(batch)
    torch.manual_seed(99); a=plain.sample(8,keys=KEYS)
    torch.manual_seed(99); b=packed.sample(8,keys=KEYS)
    assert len(plain)==len(packed)
    for k in KEYS:
        assert a[k].dtype==b[k].dtype and torch.equal(a[k],b[k]),k
bad = roll.clone(); bad['obs_channels'][0,0,0,0]=2
try:
    packed.add(bad)
except ValueError:
    pass
else:
    raise AssertionError('nonbinary channel accepted')
full = PackedReplay(131072,bc)
for arr in full.storage.values():
    arr.fill(0)
results['packed_replay_bytes']=sum(a.nbytes for a in full.storage.values())
results['replay_roundtrip_exact']=True
results['full_capacity_touched']=True
Path('numeric-production.json' if production else 'numeric-result.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results),flush=True)
