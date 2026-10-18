from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from experiments.vrpo_repro.gae_cuda_graph import GaeGraph
from experiments.vrpo_repro import train_with_gae_graph as entry
from marvel.model.tensordict import TensorDict
from marvel.training.gae import compute_gae_from_qvalue


def test_entry_restores_original_after_training_failure(monkeypatch):
    original = entry.vrpo.compute_gae_from_qvalue
    def fail():
        assert isinstance(entry.vrpo.compute_gae_from_qvalue, GaeGraph)
        raise RuntimeError('training failure')
    monkeypatch.setattr(entry, 'train_main', fail)
    with pytest.raises(RuntimeError, match='training failure'):
        entry.main()
    assert entry.vrpo.compute_gae_from_qvalue is original


@pytest.mark.parametrize('explicit_probs', [False, True])
def test_full_cache_forwards_original_arguments_without_cuda_or_eviction(monkeypatch, explicit_probs):
    def unexpected(*args, **kwargs):
        pytest.fail('A full cache must not create, capture or replay a new CUDA graph')
    for name in ('Stream', 'current_stream', 'stream', 'CUDAGraph', 'graph'):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    tensor = SimpleNamespace(is_cuda=True, shape=(2, 3), stride=lambda: (3, 1),
                             dtype=torch.float32, device=torch.device('cuda'))
    rollout = dict.fromkeys(('num_steps', 'player_ids', 'rewards', 'action_taken', 'avail_probs'), tensor)
    probs = SimpleNamespace(**vars(tensor)) if explicit_probs else None
    result = object()
    original = Mock(return_value=result)
    graph = GaeGraph(original)
    graph.cache = {('existing', i): object() for i in range(8)}
    saved_cache = graph.cache.copy()
    with torch.no_grad():
        actual = graph(rollout, tensor, avail_probs_new=probs, discount_gamma=.91, gae_lambda=.73)
    assert actual is result
    original.assert_called_once_with(rollout, tensor, avail_probs_new=probs,
                                     discount_gamma=.91, gae_lambda=.73)
    assert graph.cache == saved_cache
    assert graph.stats() == dict(calls=1, graph_count=8, capture_calls=[])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_real_graph_updates_inputs_and_preserves_retained_outputs():
    graph = GaeGraph(compute_gae_from_qvalue)
    retained = []
    cases = [
        (2, 1., .95, False), (2, 1., .95, False), (3, 1., .95, False),
        (2, .9, .95, False), (2, 1., .9, False), (2, 1., .95, True),
        (4, 1., .95, False), (5, 1., .95, False), (6, 1., .95, False),
        (7, 1., .95, False), (2, 1., .95, False),
    ]
    with torch.no_grad():
        for i, (batch, gamma, gae_lambda, strided) in enumerate(cases):
            steps, actions, players = 5, 3, 3
            rollout = TensorDict(dict(
                num_steps=torch.full((batch,), steps, device='cuda', dtype=torch.long),
                player_ids=torch.arange(steps, device='cuda').remainder(players).expand(batch, steps).clone(),
                rewards=torch.randn(batch, steps, players, device='cuda') + i,
                action_taken=torch.zeros(batch, steps, device='cuda', dtype=torch.long),
            ), batch_size=[batch], device='cuda')
            q = torch.randn(batch, steps, actions, players, device='cuda')
            if strided:
                q = q.transpose(-1, -2)
                assert not q.is_contiguous()
            probs = torch.softmax(torch.randn(batch, steps, actions, device='cuda'), dim=-1)
            args = dict(avail_probs_new=probs, discount_gamma=gamma, gae_lambda=gae_lambda)
            reference = compute_gae_from_qvalue(rollout, q, **args)
            result = graph(rollout, q, **args)
            retained.append((result, reference))
            for old, expected in retained:
                for field in fields(old):
                    assert torch.equal(getattr(old, field.name), getattr(expected, field.name))
    assert graph.stats() == dict(calls=11, graph_count=8, capture_calls=[1, 3, 4, 5, 6, 7, 8, 9])
