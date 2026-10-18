"""CPU-testable checks shared by the staged frozen and live VRPO experiments."""
import hashlib
import math
import random

import numpy as np
import torch


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def finite(value):
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite(child)
    elif isinstance(value, torch.Tensor) and value.is_floating_point():
        assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
    elif isinstance(value, float):
        assert math.isfinite(value)


def capture_rng(*, cuda=False):
    engine, keys, position, gaussian, cached = np.random.get_state()
    state = dict(python=random.getstate(), numpy=(engine, keys.tolist(), position, gaussian, cached),
                 torch_cpu=torch.get_rng_state().clone())
    if cuda:
        state['torch_cuda'] = torch.cuda.get_rng_state().clone()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    engine, keys, position, gaussian, cached = state['numpy']
    np.random.set_state((engine, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state['torch_cpu'])
    if 'torch_cuda' in state:
        torch.cuda.set_rng_state(state['torch_cuda'])


def tensor_digest(value):
    value = value.detach()
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest())


def loss_masks(batch):
    time = torch.arange(batch['player_ids'].shape[1], device=batch['player_ids'].device)[None, :] < batch['num_steps'][:, None]
    return time, time & (batch['player_ids'] >= 0), time & (batch['player_ids'] >= -1)


def batch_identity(batch, origins):
    time, policy, critic = loss_masks(batch)
    assert len(origins) == int(batch['num_steps'].shape[0])
    return dict(origins=origins, fields={k: tensor_digest(v) for k, v in batch.items()},
                masks={k: tensor_digest(v) for k, v in dict(time=time, policy=policy, critic=critic).items()})


def policy_diagnostics(advantage, ratio, clip_coef):
    assert advantage.shape == ratio.shape and advantage.ndim == 1 and advantage.numel() > 1
    advantage, ratio = advantage.detach(), ratio.detach()
    normalized = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    analytic_active = ((normalized >= 0) & (ratio <= 1 + clip_coef)) | ((normalized < 0) & (ratio >= 1 - clip_coef))
    analytic_coefficient = -normalized * ratio * analytic_active / ratio.numel()
    # Use the production max/clamp operators: clamp's boundary subgradient differs
    # between Torch versions. The inclusive analytic branch is diagnostic only.
    with torch.enable_grad():
        variable = ratio.clone().requires_grad_(True)
        unclipped = -normalized * variable
        clipped = -normalized * torch.clamp(variable, 1 - clip_coef, 1 + clip_coef)
        derivative = torch.autograd.grad(torch.max(unclipped, clipped).mean(), variable)[0]
        coefficient = derivative * variable.detach()
    return dict(normalized_advantage=normalized, pg_active=coefficient != 0, pg_coefficient=coefficient,
                pg_analytic_active=analytic_active, pg_analytic_coefficient=analytic_coefficient,
                pg_branch_tie=unclipped.detach() == clipped.detach(),
                clipped=(ratio - 1).abs() > clip_coef)


def gradients(a, b):
    assert a.keys() == b.keys() and a
    for name in a:
        assert a[name].dtype == b[name].dtype and a[name].shape == b[name].shape, name
    x = torch.cat([v.flatten() for v in a.values()]).double()
    y = torch.cat([v.flatten() for v in b.values()]).double()
    assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
    nx, ny = float(x.norm()), float(y.norm())
    difference = float((x - y).norm())
    # Normalize first so tiny, nonzero FP32 vectors are not mistaken for orthogonal ones.
    cosine = float(torch.dot(x / nx, y / ny)) if nx and ny else (1. if nx == ny else 0.)
    relative = difference / nx if nx else (0. if not difference else None)
    return dict(cosine=cosine, relative_l2=relative, max_abs=float((x-y).abs().max()),
                reference_norm=nx, candidate_norm=ny, zero_reference_mismatch=nx == 0 and ny != 0)


def delta(a, b, rtol, atol):
    assert a.dtype == b.dtype and a.shape == b.shape
    x, y = a.double(), b.double()
    error = (x-y).abs().flatten()
    return dict(max_abs=float(error.max()) if error.numel() else 0.,
                p99_abs=float(error.quantile(.99)) if error.numel() else 0.,
                allclose=bool(torch.allclose(x, y, rtol=rtol, atol=atol)))


def compare(a, b, thresholds):
    assert a.keys() == b.keys() and a['samples'].keys() == b['samples'].keys() and a['losses'].keys() == b['losses'].keys()
    booleans = ('clipped', 'pg_active', 'pg_analytic_active', 'pg_branch_tie')
    values = {k: delta(a['samples'][k], b['samples'][k], thresholds['output_rtol'], thresholds['output_atol'])
              for k in a['samples'] if k not in booleans}
    disagreements = {}
    for name in booleans:
        if name in a['samples']:
            x, y = a['samples'][name], b['samples'][name]
            assert x.dtype == y.dtype == torch.bool and x.shape == y.shape
            disagreements[name] = float((x != y).double().mean()) if x.numel() else 0.
    losses = {k: dict(reference=v, candidate=b['losses'][k], abs_delta=abs(v-b['losses'][k]),
                     close=abs(v-b['losses'][k]) <= thresholds['loss_atol'] + thresholds['loss_rtol']*abs(v))
              for k, v in a['losses'].items()}
    g = gradients(a['gradients'], b['gradients']) if 'gradients' in a else None
    gates = dict(outputs=all(v['allclose'] for v in values.values()), losses=all(v['close'] for v in losses.values()),
                 clip_disagreement=disagreements.get('clipped', 0.) <= thresholds['clip_disagreement_max'],
                 pg_active_exact=disagreements.get('pg_active', 0.) == 0.,
                 pg_branch_exact=all(disagreements.get(k, 0.) == 0. for k in ('pg_analytic_active', 'pg_branch_tie')),
                 input_identity=exact(a['input'], b['input']), clip_coef=a['clip_coef'] == b['clip_coef'])
    if g:
        gates.update(gradient_cosine=g['cosine'] >= thresholds['gradient_cosine_min'],
                     gradient_relative_l2=g['relative_l2'] is not None and g['relative_l2'] <= thresholds['gradient_relative_l2_max'])
    return dict(bitwise_equal=exact(a, b), samples=values, losses=losses, gradients=g,
                disagreements=disagreements, gates=gates)


def samples(batch, probs, q, gae, *, clip_coef):
    _, policy, critic = loss_masks(batch)
    legal = (batch['avail_weights'] > 0) & policy[..., None]
    qlegal = ((batch['avail_weights'] > 0) & critic[..., None])[..., None].expand_as(q)
    result = dict(avail_probs=probs[legal], qvalues=q[qlegal], advantage=gae.player_advantage[policy],
                  qlambda_target=gae.qvalue_lbd_true[critic], qtd1_target=gae.qvalue_td1_true[critic])
    if 'avail_probs' in batch.keys():
        taken = batch['action_taken'].long()[..., None]
        new = probs.gather(-1, taken).squeeze(-1)[policy].log()
        old = batch['avail_probs'].gather(-1, taken).squeeze(-1)[policy].log()
        ratio = (new-old).exp()
        result.update(logratio=new-old, ratio=ratio, **policy_diagnostics(result['advantage'], ratio, clip_coef))
    return {k: v.detach().cpu().clone() for k, v in result.items()}
