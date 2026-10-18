"""The archived two-iteration VRPO runner, with explicit state and input gates.

Run as ``python -m experiments.vrpo_repro.staged.frozen_numeric ROOT VARIANT``.
ROOT owns the frozen input, protocol and compile_mode.py; PYTHONPATH
must select the isolated baseline/candidate source prepared by the orchestrator.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import inspect
from itertools import islice
import json
from pathlib import Path
import sys

import pytest
import torch

from experiments.vrpo_repro.repro_runtime import deterministic_runtime, runtime_settings
from experiments.vrpo_repro.staged.numeric_checks import (
    batch_identity, capture_rng, compare, exact, finite, loss_masks, restore_rng, samples,
)
from marvel.config import TrainConfig
from marvel.model.tensordict import TensorDict
from marvel.optim.common import MultiOptimizer
from marvel.training import vrpo
from marvel.training.workflow import TrainingWorkflow
from marvel.utils import configure_torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root, variant, *, live=False, observation_context=None):
    root = Path(root).resolve()
    assert variant in (('default', 'repeat', 'k1', 'e1') if live else ('default', 'repeat', 'candidate'))
    sys.path.insert(0, str(root))
    from compile_mode import compile_mode, mode_options, acceleration_stats

    protocol = json.loads((root/'protocol.json').read_text())
    thresholds = protocol['numerical_thresholds']
    mode = 'gae-cudagraph' if variant == 'candidate' else 'default'
    prefix = 'live-' if live else ''
    reference_dir = root/(prefix+'default')
    if variant not in ('default', 'repeat'):
        assert json.loads((root/(prefix+'repeat')/'summary.json').read_text())['status'] == 'PASS'
    assert sha(root/'frozen_batch.pt') == protocol['frozen_batch_sha256']
    directory = root/(prefix+variant)
    directory.mkdir()
    cfg = TrainConfig(algo_name='vrpo', game_name='doudizhu', device='cuda', seed=0,
        rollout_batch_size=256, collect_batch_size=128, mini_batches=2, replay_buffer_scale=16,
        compile_backend='inductor', param_stats_interval=-1, checkpoint_interval=-1, evaluation_interval=-1,
        total_iterations=50, run_name=prefix+variant, logging_dir=str(directory/'training'))
    configure_torch(cfg)
    assert cfg.ppo.agent_epoch == 4 and cfg.vrpo.aux_epoch == 4
    assert torch.are_deterministic_algorithms_enabled() and torch.backends.cuda.matmul.allow_tf32
    assert torch.get_float32_matmul_precision() == 'high' and not cfg.deterministic_disable_tf32
    assert not cfg.benchmark and not cfg.profile
    data = torch.load(root/'frozen_batch.pt', map_location='cpu', weights_only=True)
    for key in ('model', 'ppo', 'vrpo', 'rollout_batch_size', 'collect_batch_size', 'mini_batches', 'replay_buffer_scale', 'seed'):
        assert asdict(cfg)[key] == data['config'][key], key
    full = TensorDict(data['rollout'], batch_size=[128], device='cuda')
    probe = full[data['indices'].cuda()]
    models, compiled, runtimes, buffers, indices, checks, collections = [], [], [], [], [], [], []
    initial_rng, sampled_origins = None, None
    steps = dict(agent=0, critic=0)
    expected_per_iteration = dict(agent=cfg.ppo.agent_epoch*cfg.mini_batches, critic=cfg.vrpo.aux_epoch*cfg.mini_batches)
    assert expected_per_iteration == dict(agent=8, critic=8)
    original_prepare, original_resume = TrainingWorkflow.prepare, TrainingWorkflow.resume
    original_collect, original_log = TrainingWorkflow.collect_trajectories, TrainingWorkflow.log_metrics
    original_perm, original_clip, original_step = torch.randperm, torch.nn.utils.clip_grad_norm_, MultiOptimizer.step
    original_buffer, original_iterations = vrpo.TrajectoryBuffer, TrainingWorkflow.iterations
    reference = torch.load(reference_dir/'sequence.pt', map_location='cpu', weights_only=True) if variant != 'default' else None

    def state_check(name, state):
        finite(state)
        if reference is not None:
            other = torch.load(reference_dir/name, weights_only=True)
            assert exact(state, other), f'Full model/optimizer state differs: {name}'
        if variant != 'repeat':
            torch.save(state, directory/name)

    def compare_save(name, raw):
        finite(raw)
        if variant != 'repeat':
            torch.save(raw, directory/name)
        if reference is not None:
            left = torch.load(reference_dir/name, map_location='cpu', weights_only=True)
            result = compare(left, raw, thresholds)
            checks.append(dict(artifact=name, **result))
            if not result['bitwise_equal']:
                if variant == 'repeat':
                    torch.save(raw, directory/name)
                (directory/'failed-check.json').write_text(json.dumps(checks[-1], indent=2, allow_nan=False)+'\n')
            assert result['bitwise_equal'], f'Staged exact contract failed: {name}'

    def resume(runtime, state):
        original_resume(runtime, state)
        for key in ('agent_resume_state_dict', 'agent_state_dict'):
            state.entries[key].load_state_dict(data['agent_state'], strict=True)
        state.entries['critic_state_dict'].load_state_dict(data['critic_state'], strict=True)
        for role in steps:
            optim = state.entries[f'{role}_optimizer_state_dict']
            assert all(not v['state'] for v in optim.state_dict()['optimizers'].values())
            assert {type(o).__name__ for o in optim.optimizers.values()} == {'Muon', 'AdamW'}
        runtimes.append(runtime)
        state_check('initial-state.pt', {k: v.state_dict() for k, v in state.entries.items() if hasattr(v, 'state_dict')})

    def prepare(runtime, model, *args, **kwargs):
        assert all(getattr(m, 'attention_dropout', 0) == 0 for m in model.modules())
        models.append(model)
        result = original_prepare(runtime, model, *args, **kwargs)
        compiled.append(result)
        return result

    def collect(runtime, agent_model):
        nonlocal initial_rng
        assert not torch.is_autocast_enabled('cuda')
        if runtime.current_iteration == 1:
            state = capture_rng(cuda=True)
            state['torch_cpu'], state['torch_cuda'] = data['rng_cpu'], data['rng_cuda']
            initial_rng = reference['initial_rng'] if reference is not None else state
            restore_rng(initial_rng)
            assert exact(capture_rng(cuda=True), initial_rng)
        if live:
            batch = original_collect(runtime, agent_model=agent_model)
        else:
            batch = full.clone()
            runtime.set_sampler_counters(total_env_steps=int(full['num_steps'].sum())*runtime.current_iteration, total_game_steps=0)
            runtime.sync_step_totals_from_sampler()
        assert int(batch['num_steps'].shape[0]) == cfg.collect_batch_size
        origins = [(runtime.current_iteration, i) for i in range(cfg.collect_batch_size)]
        record = dict(iteration=runtime.current_iteration, input=batch_identity(batch, origins),
                      origin='real_sampler' if live else protocol['frozen_batch_sha256'])
        if reference is not None:
            assert exact(record, reference['collections'][len(collections)]), 'Collected input differs'
        collections.append(record)
        return batch

    def permutation(*args, **kwargs):
        value = original_perm(*args, **kwargs)
        caller = inspect.currentframe().f_back
        if caller.f_code.co_filename == vrpo.__file__ or caller.f_code.co_name == 'sample':
            record = dict(n=int(args[0]), iteration=runtimes[0].current_iteration,
                          caller=caller.f_code.co_name, line=caller.f_lineno, indices=value.cpu().clone())
            if reference is not None:
                assert exact(record, reference['indices'][len(indices)]), 'Replay/minibatch draws differ'
            indices.append(record)
        return value

    def buffer(*args, **kwargs):
        result = original_buffer(*args, **kwargs)
        buffers.append(result)
        storage_origins = [None]*result.batch_size
        original_add, original_sample = result.add, result.sample

        def add(batch):
            pointer, count = result._ptr, int(batch.batch_size[0])
            original_add(batch)
            for row in range(count):
                storage_origins[(pointer+row) % result.batch_size] = (runtimes[0].current_iteration, row)

        def sample(*args, **kwargs):
            nonlocal sampled_origins
            batch = original_sample(*args, **kwargs)
            assert indices[-1]['caller'] == 'sample'
            draw = indices[-1]['indices'][:int(batch.batch_size[0])].tolist()
            sampled_origins = [storage_origins[i] for i in draw]
            assert all(origin is not None for origin in sampled_origins)
            return batch

        result.add, result.sample = add, sample
        return result

    def clip(parameters, *args, **kwargs):
        params = list(parameters)
        role = 'agent' if id(params[0]) == id(next(models[0].parameters())) else 'critic'
        model = models[0 if role == 'agent' else 1]
        assert id(params[0]) == id(next(model.parameters())) and not torch.is_autocast_enabled('cuda')
        local = inspect.currentframe().f_back.f_locals
        batch = local['mb_rollout']
        selected = local['mb_indices' if role == 'agent' else 'batch_idx'].cpu().tolist()
        generation = runtimes[0].current_iteration
        origins = [(generation, i) for i in selected] if role == 'agent' or local['aux_ep'] == 0 else [sampled_origins[i] for i in selected]
        clip_coef = float(local['sched'].clip_coef)
        losses = {k: float(v.detach().cpu()) for k, v in vars(local['pg_loss' if role == 'agent' else 'qvalue_loss']).items()}
        losses['total_loss'] = float(local['agent_loss_mean'].detach().cpu()) if role == 'agent' else losses['qv_lbd_loss_mean']
        raw = dict(gradients={k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None},
                   samples=samples(batch, local['mb_avail_probs_new'], local['mb_qvalues'], local['mb_gae'], clip_coef=clip_coef),
                   losses=losses, input=batch_identity(batch, origins), clip_coef=clip_coef)
        compare_save(f'{role}-step-{steps[role]+1}.pt', raw)
        return original_clip(params, *args, **kwargs)

    def step(optimizer, *args, **kwargs):
        role = 'agent' if optimizer is runtimes[0]._state.entries['agent_optimizer_state_dict'] else 'critic'
        assert not torch.is_autocast_enabled('cuda')
        result = original_step(optimizer, *args, **kwargs)
        steps[role] += 1
        finite([p for g in optimizer.param_groups for p in g['params']])
        finite([o.state for o in optimizer.optimizers.values()])
        return result

    def log(runtime):
        original_log(runtime)
        iteration = runtime.current_iteration
        assert steps == {k: n*iteration for k, n in expected_per_iteration.items()}
        sched = inspect.currentframe().f_back.f_locals['sched']
        rng = capture_rng(cuda=True)
        with torch.no_grad():
            probs = compiled[0](rollout=probe).avail_probs
            q = compiled[1](rollout=probe).qvalues
            gae = vrpo.compute_gae_from_qvalue(probe, q, avail_probs_new=probs, discount_gamma=cfg.discount_gamma, gae_lambda=cfg.ppo.gae_lambda)
            loss = vrpo.compute_clipped_pg_loss(probe, clip_coef=sched.clip_coef, avail_probs_new=probs, player_advantage=gae.player_advantage)
        compare_save(f'probe-{iteration}.pt', dict(samples=samples(probe, probs, q, gae, clip_coef=sched.clip_coef),
            losses={k: float(v.cpu()) for k, v in vars(loss).items()}, clip_coef=float(sched.clip_coef),
            input=batch_identity(probe, [('frozen_probe', i) for i in data['indices'].tolist()])))
        assert exact(rng, capture_rng(cuda=True)), 'Probe changed RNG'
        state_check(f'state-{iteration}.pt', {k: v.state_dict() for k, v in runtime._state.entries.items() if hasattr(v, 'state_dict')})
        print(variant, 'completed iteration', iteration, 'steps', steps, flush=True)

    observer = observation_context() if observation_context is not None else nullcontext()
    with observer, compile_mode(mode) as compile_calls, pytest.MonkeyPatch.context() as patch:
        for name, function in (('resume', resume), ('prepare', prepare), ('collect_trajectories', collect), ('log_metrics', log)):
            patch.setattr(TrainingWorkflow, name, function)
        patch.setattr(TrainingWorkflow, 'iterations', lambda runtime: islice(original_iterations(runtime), 2))
        patch.setattr(torch, 'randperm', permutation)
        patch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)
        patch.setattr(MultiOptimizer, 'step', step)
        patch.setattr(vrpo, 'TrajectoryBuffer', buffer)
        vrpo.VRPOTrainer(cfg).run()
    sequence = dict(indices=indices, initial_rng=initial_rng, final_rng=capture_rng(cuda=True), collections=collections)
    if reference is not None:
        assert exact(sequence, reference), 'Final RNG, inputs or Replay/minibatch draws differ'
    torch.save(sequence, directory/'sequence.pt')
    assert steps == dict(agent=16, critic=16) and len(indices) == 24 and len(collections) == 2
    assert len(checks) == (0 if variant == 'default' else 34)
    assert [c['module'] for c in compile_calls] == ['Agent', 'CentralCritic']
    assert all(c['mode'] == 'default' and 'options' not in c for c in compile_calls)
    passed = all(c['bitwise_equal'] and all(c['gates'].values()) for c in checks)
    time_mask, _, _ = loss_masks(full)
    source_root = Path(vrpo.__file__).resolve().parents[2]
    summary = dict(status='PASS' if passed else 'NUMERICAL_NO_GO', variant=variant, mode=mode,
        scope='real_sampler' if live else 'frozen_gae_only', iterations=2, optimizer_steps=steps,
        checks=checks, nonfinite_count=0, fp32=True, amp=False, runtime=runtime_settings(), config=asdict(cfg),
        replay_capacity=buffers[0].batch_size, replay_size=len(buffers[0]),
        replay_dtypes={k: str(v.dtype) for k, v in buffers[0]._storage.items()},
        rng_index_calls=len(indices), rng_indices_equal=True, all_rng_equal=True, updated_states_exact=True,
        compile_calls=compile_calls, mode_options=mode_options(mode), acceleration=acceleration_stats(), thresholds=thresholds,
        protocol_sha256=sha(root/'protocol.json'), frozen_batch_sha256=sha(root/'frozen_batch.pt'), script_sha256=sha(__file__),
        source_sha256={name: sha(source_root/name) for name in protocol['source_sha256']},
        frozen_chance_coverage={str(pid): int(((full['player_ids'] == pid) & time_mask).sum()) for pid in (-2, -1)},
        minus_one_scope='synthetic shared-loss contract tested separately; no real -1 in this doudizhu fixture',
        input_identity='all fields/masks SHA256 and (collection generation,row); replay insertion generation preserved',
        numeric_contract='exact for all staged candidates; default generates reference, repeat establishes A/A')
    (directory/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(variant, summary['status'], flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('variant', choices=('default', 'repeat', 'candidate'))
    args = parser.parse_args()
    with deterministic_runtime():
        run(args.root, args.variant)
