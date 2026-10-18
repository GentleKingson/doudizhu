from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from marvel.buffer import BufferConfig, BufferFieldSpec, TrajectoryBuffer
from marvel.config import TrainConfig
from marvel.runtime.full_state import (ActionJournal, canonical_hash, full_config, load_buffer,
    save_buffer, sha256, verify_generation, write_json)
from marvel.runtime.distributed import DistributedRuntime


def equal(left, right, *, close=False):
    if isinstance(left, torch.Tensor):
        if close and left.is_floating_point():
            torch.testing.assert_close(left.cpu(),right.cpu(),rtol=1e-5,atol=1e-6)
        else:
            assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key],close=close)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            equal(x, y,close=close)
    else:
        assert left == right


def test_full_config():
    cfg = TrainConfig(collect_batch_size=128, agent_merging_beta=.999, lr_stable_iterations=5000)
    assert asdict(full_config(asdict(cfg))) == asdict(cfg)
    data = asdict(cfg)
    del data['replay_buffer_scale']
    with pytest.raises(ValueError):
        full_config(data)


def test_explicit_one_thread(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    threads = torch.get_num_threads()
    try:
        DistributedRuntime.init_from_env('cpu')
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(threads)


def test_replay_wrap_and_bounded_storage(tmp_path):
    cfg = BufferConfig(game=None, device='cpu', field_specs={'x':BufferFieldSpec((64,), torch.float32)})
    buffer = TrajectoryBuffer(11, cfg)
    for i in range(5):
        batch = buffer.empty(batch_size=4)
        batch['x'].fill_(i)
        buffer.add(batch)
    save_buffer(buffer, tmp_path / 'replay', len(buffer))
    restored = TrajectoryBuffer(11, cfg)
    load_buffer(restored, tmp_path / 'replay')
    assert restored._size == buffer._size and restored._ptr == buffer._ptr
    state = torch.get_rng_state()
    expected = buffer.sample(6)
    torch.set_rng_state(state)
    actual = restored.sample(6)
    for key in expected.keys():
        assert torch.equal(expected[key], actual[key])
    # A one-row shard cannot retain the replay's complete backing allocation.
    layout = json.loads((tmp_path / 'replay/layout.json').read_text())
    assert layout['fields']['x']['shape'] == [11,64]


def test_journal_sealing(tmp_path):
    journal = ActionJournal(tmp_path/'journal', 17, 4)
    actions = np.zeros((4,5), dtype=np.int64)
    journal.append(actions)
    first = journal.seal()
    journal.append(actions+1)
    second = journal.seal()
    assert first[0]['bytes'] == 4*5*8
    assert second[1]['first'] == 1 and journal.count == 2
    assert sha256(first[0]['path']) == first[0]['sha256']


def run_probe(tmp_path, name, total, resume=None, failure=None, device='cpu',fast=False):
    output = tmp_path/name
    command = [sys.executable, '-m', 'experiments.colab_l4_500.fork_probe', '--output', str(output), '--total',str(total)]
    command += ['--device',device]
    if fast:
        command += ['--fast']
    if resume:
        command += ['--resume',str(resume)]
    env = {**os.environ, 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'PYTHONHASHSEED':'17',
           'WANDB_DISABLED':'true','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if failure is not None:
        assert result.returncode != 0 and failure in result.stderr, result.stdout + result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    return Path(json.loads((output/'probe_result.json').read_text())['generation'])


@pytest.mark.parametrize('device,fast',[('cpu',False),('cuda',False),('cuda',True)])
def test_fresh_process_training_fork_and_negative_restore(tmp_path,device,fast):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('Requires actual CUDA runtime')
    continuous = run_probe(tmp_path, 'continuous', 6,device=device,fast=fast)
    parent = run_probe(tmp_path, 'parent', 3,device=device,fast=fast)
    child = run_probe(tmp_path, 'child', 6, parent,device=device,fast=fast)
    left = torch.load(continuous/'train_state.pt',weights_only=True)
    right = torch.load(child/'train_state.pt',weights_only=True)
    for key in left:
        if key not in {'train_config','parent_manifest_sha256'}:
            equal(left[key],right[key],close=fast)
    parent_state = torch.load(parent/'train_state.pt',weights_only=True)
    assert parent_state['sampler']['total_completed_games'] > 3*4
    assert parent_state['sampler']['commit_buffer_size'] > 0
    assert left['ema_updater']['update_count'] == 6
    for kind in ('replay','rollout','commit'):
        a = json.loads((continuous/kind/'layout.json').read_text())
        b = json.loads((child/kind/'layout.json').read_text())
        assert a == b
        for file in (continuous/kind).glob('*.pt'):
            equal(torch.load(file, weights_only=True),torch.load(child/kind/file.name,weights_only=True),close=fast)
    for target in ('replay/layout.json','train_state.pt','journal.json'):
        broken = tmp_path / ('broken-'+target.replace('/','-'))
        shutil.copytree(parent, broken)
        path = broken/target
        path.write_bytes(path.read_bytes()[:20])
        with pytest.raises((ValueError, json.JSONDecodeError)):
            verify_generation(broken)
    with pytest.raises(FileNotFoundError):
        verify_generation(parent, require_remote=True)
    for field in ('critic_optimizer_state_dict','rng','sampler','ema_updater'):
        broken=tmp_path/('missing-'+field)
        shutil.copytree(parent,broken)
        payload=torch.load(broken/'train_state.pt',weights_only=True)
        del payload[field]
        torch.save(payload,broken/'train_state.pt')
        manifest=json.loads((broken/'manifest.json').read_text())
        path=broken/'train_state.pt'
        manifest['files']['train_state.pt']={'bytes':path.stat().st_size,'sha256':sha256(path)}
        write_json(broken/'manifest.json',manifest)
        write_json(broken/'CAPTURED.json',{'manifest_sha256':sha256(broken/'manifest.json')})
        run_probe(tmp_path,'reject-'+field,6,broken,failure='Missing required full-state field')


def test_saved_slice_owns_only_bounded_storage(tmp_path):
    cfg=BufferConfig(game=None,device='cpu',field_specs={'x':BufferFieldSpec((64,),torch.float32)})
    buffer=TrajectoryBuffer(100,cfg)
    save_buffer(buffer,tmp_path/'small',3)
    tensor=torch.load(tmp_path/'small/x-000000.pt',weights_only=True)
    assert tensor.untyped_storage().nbytes()==3*64*4


def test_budget_stop_preserves_complete_iteration(tmp_path, monkeypatch):
    stop = tmp_path/'STOP_AFTER_BOUNDARY'
    stop.touch()
    monkeypatch.setenv('MARVEL_STOP_AFTER_BOUNDARY', str(stop))
    generation = run_probe(tmp_path, 'budget-stop', 6)
    state = torch.load(generation/'train_state.pt', weights_only=True)
    assert state['iteration'] == 1
    assert state['ema_updater']['update_count'] == 1


def test_safe_boundary_checkpoint_triggers(monkeypatch):
    from types import SimpleNamespace
    from marvel.training.workflow import CheckpointPolicy
    cfg=TrainConfig(run_name='cadence',logging_dir='/tmp/cadence',full_state_checkpoints=True,
                    total_iterations=501,checkpoint_interval=250,replay_buffer_scale=64)
    clock=[0.]
    monkeypatch.setattr('marvel.training.workflow.time.monotonic',lambda:clock[0])
    policy=CheckpointPolicy(SimpleNamespace(cfg=cfg,is_primary=True))
    assert not policy.due(1)
    assert policy.due(64) and policy.due(250) and policy.due(500) and policy.due(501)
    clock[0]=901.
    assert policy.due(2)
    policy.last_iteration=2
    assert not policy.due(2)
