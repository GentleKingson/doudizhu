"""Lossless replay storage and activation recomputation; upstream losses stay intact."""
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from marvel.buffer import trajectory_field_specs
from marvel.model.tensordict import TensorDict
from marvel.model.types import ActiveAgentOutput, CriticOutput

KEYS = ('num_steps', 'player_ids', 'obs_tokens', 'obs_channels',
        'avail_tokens', 'avail_weights', 'action_taken', 'rewards')


class PackedReplay:
    def __init__(self, batch_size, config):
        assert str(config.device) == 'cpu'
        self.batch_size = batch_size
        self.device = torch.device('cpu')
        self._size = self._ptr = 0
        templates = {'num_steps': torch.zeros(1, dtype=torch.long)}
        templates.update({k: f(1, 'cpu') for k, f in trajectory_field_specs(config.game).items() if k in KEYS})
        self.dtypes = {k: t.dtype for k, t in templates.items()}
        self.storage = {}
        for k, t in templates.items():
            shape = list(t.shape[1:])
            dtype = t.numpy().dtype
            if k == 'obs_channels':
                assert shape[-1] % 8 == 0
                self.channels = shape[-1]
                shape[-1] //= 8
            elif t.dtype == torch.long:
                dtype = np.int16
            self.storage[k] = np.empty((batch_size, *shape), dtype=dtype)
        print('PACKED_REPLAY_BYTES', sum(a.nbytes for a in self.storage.values()), flush=True)

    def __len__(self):
        return self._size

    def add(self, batch):
        count = int(batch.batch_size[0])
        assert 0 < count <= self.batch_size
        indices = np.arange(self._ptr, self._ptr + count) % self.batch_size
        for key, dest in self.storage.items():
            source = batch[key].detach().cpu().numpy()
            if key == 'obs_channels':
                if not np.all((source == 0) | (source == 1)):
                    raise ValueError('Channel bit packing requires binary values')
                source = np.packbits(source, axis=-1)
            elif dest.dtype == np.int16:
                if source.min() < -32768 or source.max() > 32767:
                    raise ValueError(f'{key} exceeds lossless int16 range')
            dest[indices] = source
        self._ptr = (self._ptr + count) % self.batch_size
        self._size = min(self.batch_size, self._size + count)

    def sample(self, batch_size, *, keys=None):
        assert 0 < batch_size <= self._size
        indices = torch.randperm(self._size, device=self.device)[:batch_size].numpy()
        data = {}
        for key in KEYS if keys is None else keys:
            arr = self.storage[key][indices]
            if key == 'obs_channels':
                arr = np.unpackbits(arr, axis=-1, count=self.channels)
            data[key] = torch.from_numpy(arr).to(self.dtypes[key])
        return TensorDict(data, batch_size=[batch_size], device='cpu')


class ChunkedModel:
    def __init__(self, model, chunk, critic):
        self.model, self.chunk, self.critic = model, chunk, critic

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, rollout, **kwargs):
        if kwargs.get('use_cache') or kwargs.get('cache_only'):
            return self.model(rollout=rollout, **kwargs)
        outputs = []
        def forward(part):
            out = self.model(rollout=part, **kwargs)
            return out.qvalues if self.critic else out.avail_probs
        for start in range(0, rollout.batch_size[0], self.chunk):
            part = rollout[start:start+self.chunk]
            if torch.is_grad_enabled():
                value = checkpoint(forward, part, use_reentrant=False)
            else:
                value = forward(part)
            outputs.append(value)
        value = torch.cat(outputs, dim=0)
        if self.critic:
            return CriticOutput(qvalues=value)
        mask = rollout['player_ids'] >= 0
        return ActiveAgentOutput(avail_probs=value, mask=mask)


def install_replay():
    import marvel.training.vrpo as vrpo
    vrpo.TrajectoryBuffer = PackedReplay
