import json
import random

import numpy as np
import pytest
import torch

from experiments.vrpo_repro.staged.numeric_checks import (
    batch_identity, capture_rng, compare, exact, gradients, loss_masks,
    policy_diagnostics, restore_rng, tensor_digest,
)


def test_rng_roundtrip_restores_python_numpy_and_torch(tmp_path):
    original = capture_rng()
    try:
        random.seed(17)
        np.random.seed(18)
        torch.manual_seed(19)
        start = capture_rng()
        first = (random.random(), np.random.random(5).tolist(), torch.rand(5))
        end = capture_rng()
        path = tmp_path/'rng.pt'
        torch.save(start, path)
        restore_rng(torch.load(path, weights_only=True))
        assert exact(capture_rng(), start)
        assert exact(first, (random.random(), np.random.random(5).tolist(), torch.rand(5)))
        assert exact(capture_rng(), end)
    finally:
        restore_rng(original)
    assert exact(capture_rng(), original)


@pytest.mark.parametrize('clip', [.02, .1])
def test_actual_pg_branch_at_zero_and_exact_clip_boundaries(clip):
    advantage = torch.tensor([-3., -2., -1., 0., 1., 2., 3.])
    ratios = torch.tensor([1-clip-.01, 1-clip, 1-clip+.01, 1., 1+clip-.01, 1+clip, 1+clip+.01])
    before = capture_rng()
    result = policy_diagnostics(advantage, ratios, clip)
    assert result['pg_analytic_active'].tolist() == [False, True, True, True, True, True, False]
    assert result['pg_active'].tolist() == [False, True, True, False, True, True, False]
    assert result['pg_branch_tie'].tolist() == [False, True, True, True, True, True, False]
    # Exact expected values come from the deployed operators, including their
    # actual boundary derivative, not an assumed inclusive clamp subgradient.
    variable = ratios.clone().requires_grad_(True)
    normalized = (advantage-advantage.mean())/(advantage.std()+1e-8)
    loss = torch.max(-normalized*variable, -normalized*torch.clamp(variable, 1-clip, 1+clip)).mean()
    expected = torch.autograd.grad(loss, variable)[0]*ratios
    assert torch.equal(result['pg_coefficient'], expected)
    assert result['pg_coefficient'][3] == 0
    assert result['pg_coefficient'][0] == 0 and result['pg_coefficient'][-1] == 0
    assert exact(before, capture_rng())
    # A ratio can be outside the band while the actual negative-advantage branch remains active.
    other = policy_diagnostics(torch.tensor([-1., 1.]), torch.tensor([1+2*clip, 1-2*clip]), clip)
    assert other['clipped'].all() and other['pg_active'].all()


def test_actual_pg_coefficient_matches_shared_production_loss_autograd():
    from marvel.training.losses import compute_clipped_pg_loss

    advantage = torch.tensor([[-3., -2., -1., 0., 1., 2., 3.]])
    probs = torch.tensor([[[.47, .53], [.49, .51], [.495, .505], [.5, .5],
                           [.505, .495], [.51, .49], [.53, .47]]], requires_grad=True)
    batch = dict(num_steps=torch.tensor([7]), player_ids=torch.zeros(1, 7, dtype=torch.long),
                 action_taken=torch.zeros(1, 7, dtype=torch.long), avail_weights=torch.ones(1, 7, 2),
                 avail_probs=torch.full((1, 7, 2), .5))
    loss = compute_clipped_pg_loss(batch, clip_coef=.02, avail_probs_new=probs, player_advantage=advantage)
    actual = torch.autograd.grad(loss.pg_loss_mean, probs)[0][0, :, 0]*probs.detach()[0, :, 0]
    ratios = (probs.detach()[0, :, 0].log()-batch['avail_probs'][0, :, 0].log()).exp()
    result = policy_diagnostics(advantage.flatten(), ratios, .02)
    assert torch.allclose(result['pg_coefficient'], actual, rtol=1e-6, atol=1e-12)
    assert torch.equal(result['pg_active'], actual != 0)


def test_pg_diagnostic_uses_the_provided_scheduled_clip():
    a, r = torch.tensor([-1., 1.]), torch.tensor([.97, 1.03])
    assert not policy_diagnostics(a, r, .02)['pg_active'].any()
    assert policy_diagnostics(a, r, .04)['pg_active'].all()


@pytest.mark.parametrize('left,right,cosine,relative', [
    ([0., 0.], [0., 0.], 1., 0.),
    ([0., 0.], [1., 0.], 0., None),
    ([1., 0.], [0., 0.], 0., 1.),
    ([1e-35, 2e-35], [1e-35, 2e-35], 1., 0.),
    ([1e-35, 0.], [0., 1e-35], 0., 2**.5),
])
def test_zero_and_tiny_gradient_contract(left, right, cosine, relative):
    result = gradients({'w': torch.tensor(left)}, {'w': torch.tensor(right)})
    assert result['cosine'] == pytest.approx(cosine)
    assert result['relative_l2'] == (None if relative is None else pytest.approx(relative))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('other', [torch.ones(1, 2), torch.ones(2, dtype=torch.float64)])
def test_gradient_contract_rejects_per_parameter_shape_or_dtype(other):
    with pytest.raises(AssertionError):
        gradients({'w': torch.ones(2)}, {'w': other})


def test_input_digest_distinguishes_contents_masks_shape_and_generation():
    batch = dict(num_steps=torch.tensor([2]), player_ids=torch.tensor([[-2, 0, 0]]))
    original = batch_identity(batch, [(1, 0)])
    assert exact(original, batch_identity(batch, [(1, 0)]))
    assert not exact(original, batch_identity(batch, [(2, 0)]))
    batch['num_steps'][0] = 3
    changed = batch_identity(batch, [(1, 0)])
    assert original['masks']['policy'] != changed['masks']['policy']
    assert tensor_digest(torch.zeros(2)) != tensor_digest(torch.zeros(1, 2))
    assert tensor_digest(torch.tensor([0.])) != tensor_digest(torch.tensor([-0.]))


def test_synthetic_minus_one_and_minus_two_match_shared_production_loss_masks():
    from marvel.training.losses import compute_clipped_pg_loss, compute_qvalue_loss

    batch = dict(num_steps=torch.tensor([4]), player_ids=torch.tensor([[-2, -1, 0, 1, 0]]),
                 action_taken=torch.zeros(1, 5, dtype=torch.long), avail_weights=torch.ones(1, 5, 2),
                 avail_probs=torch.full((1, 5, 2), .5))
    _, policy, critic = loss_masks(batch)
    assert policy.tolist() == [[False, False, True, True, False]]
    assert critic.tolist() == [[False, True, True, True, False]]
    probs = torch.full((1, 5, 2), .5, requires_grad=True)
    policy_loss = compute_clipped_pg_loss(batch, clip_coef=.02, avail_probs_new=probs,
                                        player_advantage=torch.tensor([[99., 88., -1., 1., 77.]]))
    policy_loss.pg_loss_mean.backward()
    assert torch.equal(probs.grad.abs().sum(-1) > 0, policy)
    q = torch.ones(1, 5, 2, 3, requires_grad=True)
    targets = torch.zeros(1, 5, 3)
    critic_loss = compute_qvalue_loss(batch, q, qvalue_lbd_true=targets, qvalue_td1_true=targets)
    critic_loss.qv_lbd_loss_mean.backward()
    assert torch.equal(q.grad.abs().sum((-1, -2)) > 0, critic)


def test_comparison_rejects_branch_identity_and_zero_gradient_mismatch():
    thresholds = dict(output_rtol=1e-4, output_atol=1e-5, loss_rtol=1e-4, loss_atol=1e-5,
                      gradient_cosine_min=.99999, gradient_relative_l2_max=.001, clip_disagreement_max=.001)
    baseline = dict(samples=dict(pg_active=torch.tensor([True]), clipped=torch.tensor([False]), value=torch.ones(1)),
                    losses=dict(total_loss=1.), input={'generation': 1}, clip_coef=.02, gradients={'w': torch.zeros(1)})
    result = compare(baseline, baseline, thresholds)
    assert result['bitwise_equal'] and all(result['gates'].values())
    changed = dict(baseline, samples=dict(baseline['samples'], pg_active=torch.tensor([False])),
                   input={'generation': 2}, gradients={'w': torch.ones(1)})
    result = compare(baseline, changed, thresholds)
    assert not result['gates']['pg_active_exact'] and not result['gates']['input_identity']
    assert not result['gates']['gradient_relative_l2'] and not result['gates']['gradient_cosine']
