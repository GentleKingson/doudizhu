"""Measure the original VRPOTrainer through its existing runtime_factory hook."""
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch._dynamo.utils import counters
from marvel.config import TrainConfig
from marvel.training.workflow import TrainingWorkflow
from marvel.training.vrpo import VRPOTrainer
from marvel.utils import configure_torch


def main():
    settings = json.loads(Path(sys.argv[1]).read_text())
    out = Path(settings['output'])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings['threads'])
    torch.set_num_interop_threads(1)
    cfg = TrainConfig(algo_name='vrpo', device='cuda',
        rollout_batch_size=settings['rollout'], collect_batch_size=settings['collect'],
        mini_batches=settings['mini_batches'], replay_buffer_scale=16,
        total_iterations=settings['warmup'] + settings['measure'],
        checkpoint_interval=-1, evaluation_interval=-1, param_stats_interval=-1,
        seed=settings.get('seed', 0), logging_dir=str(out / 'training'))
    configure_torch(cfg)
    (out / 'train-config.json').write_text(json.dumps(asdict(cfg), indent=2))
    blocks = []
    runtime_info = {}
    class MeasuredWorkflow(TrainingWorkflow):
        def __init__(self, cfg, **kwargs):
            super().__init__(cfg, **kwargs)
            runtime_info.update(effective_torch_threads=torch.get_num_threads(), effective_omp=os.environ.get('OMP_NUM_THREADS'))
            print('EFFECTIVE_THREADS ' + json.dumps(runtime_info), flush=True)
        def checkpoint_due(self):
            return False

        def iterations(self):
            block_size = settings.get('block_size', settings['measure'])
            for iteration in super().iterations():
                measured = iteration > settings['warmup']
                if measured and (iteration - settings['warmup'] - 1) % block_size == 0:
                    torch.cuda.synchronize()
                    before = (self.env_steps_total, self.game_steps_total)
                    graphs = counters['stats']['unique_graphs']
                    start = time.perf_counter()
                yield iteration
                if measured and ((iteration - settings['warmup']) % block_size == 0 or iteration == cfg.total_iterations):
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    count = min(block_size, iteration - settings['warmup'])
                    row = dict(iteration=iteration, seconds=seconds,
                        trajectories=count * cfg.collect_batch_size,
                        env_steps=self.env_steps_total - before[0],
                        game_steps=self.game_steps_total - before[1],
                        new_graphs=counters['stats']['unique_graphs'] - graphs)
                    blocks.append(row)
                    print('BLOCK ' + json.dumps(row), flush=True)
                if iteration in (1, settings['warmup'], cfg.total_iterations):
                    print('PROGRESS ' + str(iteration), flush=True)

        def log_metrics(self):
            assert all(math.isfinite(v) for v in self._metric_buffer.values() if isinstance(v, (int, float))), self._metric_buffer
            super().log_metrics()

    started = time.perf_counter()
    VRPOTrainer(cfg, runtime_factory=lambda c: MeasuredWorkflow(c, game=c.game)).run()
    seconds = sum(b['seconds'] for b in blocks)
    result = dict(settings=settings, blocks=blocks, seconds=seconds,
        trajectories_per_second=sum(b['trajectories'] for b in blocks) / seconds,
        env_steps_per_second=sum(b['env_steps'] for b in blocks) / seconds,
        game_steps_per_second=sum(b['game_steps'] for b in blocks) / seconds,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        new_graphs=sum(b['new_graphs'] for b in blocks),
        wall_seconds=time.perf_counter() - started,
        torch_version=torch.__version__, cuda_version=torch.version.cuda)
    result.update(runtime_info)
    assert not list(out.rglob('*.pt')), 'Tuning must not save checkpoints'
    (out / 'measurement.json').write_text(json.dumps(result, indent=2))
    print('MEASUREMENT ' + json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
