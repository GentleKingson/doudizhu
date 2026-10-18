"""Colab wall-time/Drive adapter; VRPO and checkpoint contents remain upstream."""
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
import time


def checkpoints(directory):
    return sorted((p for p in Path(directory).glob('ckpt_*.pt')
                   if p.is_file() and re.fullmatch(r'ckpt_\d+\.pt', p.name)),
                  key=lambda p: int(p.stem.split('_')[1]))


def publish_checkpoint(manager, state, directory, keep):
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 1:
        raise ValueError('keep_checkpoints must be a positive integer')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'ckpt_{state.iteration}.pt'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    pending = directory / (target.name + '.partial')
    try:
        with tempfile.TemporaryDirectory(prefix='vrpo-checkpoint-') as tmp:
            local = Path(tmp) / target.name
            manager.save_from_state(state, checkpoint_path=local, upload_to_wandb=False)
            shutil.copyfile(local, pending)
            if local.stat().st_size != pending.stat().st_size:
                raise IOError('Incomplete checkpoint copy to Drive')
            os.replace(pending, target)
    finally:
        pending.unlink(missing_ok=True)
    # Only prune this run's checkpoints after the new file is fully published.
    for old in checkpoints(directory)[:-keep]:
        old.unlink()
    return target


def timed_workflow(settings):
    import torch
    from marvel.training.workflow import TrainingWorkflow
    for key in ('training_minutes', 'save_every_minutes'):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    keep = settings['keep_checkpoints']
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 1:
        raise ValueError('keep_checkpoints must be a positive integer')

    class TimedWorkflow(TrainingWorkflow):
        def __init__(self, cfg):
            super().__init__(cfg, game=cfg.game, checkpoint_dir=settings['checkpoint_dir'])
            print(f'CPU threads: requested={settings["threads"]}, effective={torch.get_num_threads()}', flush=True)
            self.logger.progress_bar.total = None
            self.stop_requested = False
            self.last_saved_iteration = 0
            self.last_saved_time = time.monotonic()
            self.previous_handlers = {}

        def load_checkpoint_payload(self, checkpoint_path, *, device):
            # Stage on CPU so loading does not double the live GPU model/optimizer memory.
            return super().load_checkpoint_payload(checkpoint_path, device=torch.device('cpu'))

        def checkpoint_due(self):
            return time.monotonic() - self.last_saved_time >= settings['save_every_minutes'] * 60

        def save_checkpoint(self, state):
            if state.iteration == self.last_saved_iteration:
                return
            path = publish_checkpoint(self.checkpoint_manager, state, self.checkpoint_dir, keep)
            self.last_saved_iteration = state.iteration
            self.last_saved_time = time.monotonic()
            print(f'DRIVE_CHECKPOINT {path}', flush=True)

        def iterations(self):
            start = time.monotonic()
            self.last_saved_time = start
            def request_stop(signum, frame):
                self.stop_requested = True
                print('Stop requested; finish current iteration and save.', flush=True)
            for sig in (signal.SIGINT, signal.SIGTERM):
                self.previous_handlers[sig] = signal.signal(sig, request_stop)
            try:
                for iteration in super().iterations():
                    yield iteration
                    if self.stop_requested or time.monotonic() - start >= settings['training_minutes'] * 60:
                        self.save_checkpoint(self._state)
                        print(f'TRAINING_COMPLETE iteration={iteration} elapsed_seconds={time.monotonic()-start:.1f}', flush=True)
                        break
                else:
                    if self._state is not None and self.current_iteration is not None:
                        self.save_checkpoint(self._state)
            finally:
                for sig, handler in self.previous_handlers.items():
                    signal.signal(sig, handler)
    return TimedWorkflow


def validate_resume_payload(payload):
    required = {'agent_resume_state_dict', 'agent_state_dict', 'critic_state_dict',
                'agent_optimizer_state_dict', 'critic_optimizer_state_dict',
                'iteration', 'train_config'}
    missing = required - payload.keys()
    if missing or payload.get('train_config', {}).get('algo_name') != 'vrpo':
        raise ValueError(f'Expected a full VRPO training checkpoint; missing keys: {sorted(missing)}')


def main(settings):
    import torch
    from marvel.config import TrainConfig
    from marvel.training.vrpo import VRPOTrainer
    from marvel.utils import configure_torch
    torch.set_num_threads(settings['threads'])
    torch.set_num_interop_threads(1)
    directory = Path(settings['checkpoint_dir'])
    previous = checkpoints(directory)
    resume = settings.get('resume_from', '')
    if resume == 'latest':
        if not previous:
            raise FileNotFoundError(f'No checkpoints in {directory}')
        resume = str(previous[-1])
    elif previous and not resume:
        raise ValueError('Run folder already contains checkpoints: set resume_from="latest" or choose a new run folder')
    if resume and not Path(resume).is_file():
        raise FileNotFoundError(resume)
    if resume:
        payload = torch.load(resume, map_location='cpu', weights_only=True)
        validate_resume_payload(payload)
        if previous and int(payload['iteration']) < int(previous[-1].stem.split('_')[1]):
            raise ValueError('Resuming an older checkpoint requires a new run folder to preserve newer checkpoints')
        del payload
    cfg = TrainConfig(algo_name='vrpo', device='cuda', total_iterations=40_000,
        checkpoint_interval=-1, evaluation_interval=-1, param_stats_interval=-1,
        rollout_batch_size=settings['rollout'], collect_batch_size=settings['collect'],
        mini_batches=4, replay_buffer_scale=64,
        lr_stable_iterations=5000, reg_stable_iterations=500, agent_merging_beta=0.999,
        resume_from=resume or None, resume_optimizer=True, seed=settings.get('seed', 0),
        logging_dir=settings['logging_dir'])
    cfg.vrpo.replay_buffer_device = "cpu"
    configure_torch(cfg)
    VRPOTrainer(cfg, runtime_factory=timed_workflow(settings)).run()


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text()))
