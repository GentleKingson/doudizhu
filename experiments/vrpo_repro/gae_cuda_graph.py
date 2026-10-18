"""Replay the unchanged eager GAE kernels; no network or optimizer is captured."""
from dataclasses import fields
import torch
from marvel.training.gae import GAEFromQValueOutput


class GaeGraph:
    def __init__(self, original):
        self.original = original
        self.cache = {}
        self.calls = 0
        self.capture_calls = []

    def __call__(self, rollout, state_qvalues, *, avail_probs_new=None, discount_gamma=1., gae_lambda=.95):
        assert not torch.is_grad_enabled() and not torch.is_autocast_enabled('cuda')
        probs = rollout['avail_probs'] if avail_probs_new is None else avail_probs_new
        names = ('num_steps','player_ids','rewards','action_taken')
        tensors = [rollout[n] for n in names] + [state_qvalues,probs]
        assert all(t.is_cuda for t in tensors)
        key = (discount_gamma,gae_lambda,tuple((tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in tensors))
        self.calls += 1
        if key not in self.cache:
            if len(self.cache) >= 8:
                return self.original(rollout,state_qvalues,avail_probs_new=avail_probs_new,
                                     discount_gamma=discount_gamma,gae_lambda=gae_lambda)
            static_rollout = rollout.select(*names).clone()
            static_q, static_probs = state_qvalues.clone(), probs.clone()
            stream = torch.cuda.Stream(device=state_qvalues.device)
            stream.wait_stream(torch.cuda.current_stream())
            args = dict(avail_probs_new=static_probs,discount_gamma=discount_gamma,gae_lambda=gae_lambda)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.original(static_rollout,static_q,**args)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                output = self.original(static_rollout,static_q,**args)
            self.cache[key] = (graph,static_rollout,static_q,static_probs,output)
            self.capture_calls.append(self.calls)
        graph,static_rollout,static_q,static_probs,output = self.cache[key]
        for name in names:
            static_rollout[name].copy_(rollout[name])
        static_q.copy_(state_qvalues)
        static_probs.copy_(probs)
        graph.replay()
        # Callers may keep targets until backward; never expose mutable graph storage.
        return GAEFromQValueOutput(**{f.name:getattr(output,f.name).clone() for f in fields(output)})

    def stats(self):
        return dict(calls=self.calls,graph_count=len(self.cache),capture_calls=self.capture_calls)
