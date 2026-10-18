"""Single-process VRPO generations. Replay is streamed; native state uses an action journal."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import uuid
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from marvel.config import (TrainConfig, ModelConfig, OptimizerConfig, PPOConfig,
                           VRPOConfig, ScheduleConfig)
from marvel.env_batch import EnvStepBatch
from marvel.runtime.checkpoints import build_checkpoint_payload_from_state, load_checkpoint_into_state

SCHEMA = 1
OPERATIONAL = {'total_iterations', 'run_name', 'logging_dir', 'resume_from',
               'checkpoint_interval', 'checkpoint_wallclock_seconds'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def full_config(payload):
    """Do not use the inference-only deserializer for training continuation."""
    expected = {f.name for f in fields(TrainConfig)}
    if set(payload) != expected:
        raise ValueError(f'Incomplete training config: missing={expected-set(payload)}, extra={set(payload)-expected}')
    data = dict(payload)
    for name, cls in [('model', ModelConfig), ('optimizer', OptimizerConfig), ('ppo', PPOConfig),
                      ('vrpo', VRPOConfig), ('schedule', ScheduleConfig)]:
        if set(data[name]) != {f.name for f in fields(cls)}:
            raise ValueError(f'Incomplete {name} configuration')
        data[name] = cls(**data[name])
    if data['evaluation_routines'] is not None:
        data['evaluation_routines'] = tuple(data['evaluation_routines'])
    return TrainConfig(**data)


def scientific_config(payload):
    return {k: v for k, v in payload.items() if k not in OPERATIONAL}


def sampler_signature(sampler):
    return {'env_seed': sampler._env_seed, 'rollout_batch_size': sampler.rollout_batch_size,
            'collect_batch_size': sampler.collect_batch_size, 'sampling_temperature': sampler._sampling_temperature,
            'cache_dtype': str(sampler._cache_dtype), 'device': str(sampler.device),
            'horizon': sampler.T, 'players': sampler.P, 'explicit_generators': []}


def rng_state():
    numpy_state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    algorithm, keys, pos, gaussian, cached = state['numpy']
    np.random.set_state((algorithm, np.array(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state['torch'])
    if len(state['cuda']) != torch.cuda.device_count():
        raise ValueError('CUDA RNG device count mismatch')
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def provenance():
    import nanopokerlib
    root = Path(__file__).resolve().parents[2]
    native = Path(nanopokerlib._pybind.__file__)
    source_files = [p for base in ('marvel', 'nanopokerlib/cpp') for p in (root/base).rglob('*')
                    if p.suffix in {'.py', '.hpp', '.cpp', '.h'}]
    return {'execution_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'source_sha256': canonical_hash({str(p.relative_to(root)):sha256(p) for p in sorted(source_files)}),
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'cuda': torch.version.cuda, 'platform': platform.platform(),
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'driver': subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                                              text=True).strip() if torch.cuda.is_available() else None,
            'native_sha256': sha256(native), 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'omp': os.environ.get('OMP_NUM_THREADS'), 'mkl': os.environ.get('MKL_NUM_THREADS'),
            'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
            'matmul_precision': torch.get_float32_matmul_precision(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic}


def file_record(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': sha256(path)}


class ActionJournal:
    def __init__(self, directory, seed, batch_size):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.seed, self.batch_size = int(seed), int(batch_size)
        self.shards = []
        self.count = 0
        self.active_count = 0
        self.stream = None

    def append(self, actions):
        array = np.asarray(actions, dtype='<i8')
        if array.shape != (self.batch_size, 5):
            raise ValueError('Journal requires the complete [R,5] native action batch')
        if self.stream is None:
            self.active = self.directory / f'j{self.count:012d}.bin'
            self.stream = self.active.open('xb')
        self.stream.write(array.tobytes())
        self.count += 1
        self.active_count += 1

    def seal(self):
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.stream = None
            self.shards.append({'path': str(self.active), 'first': self.count-self.active_count,
                                'count': self.active_count, **file_record(self.active)})
            self.active_count = 0
        return list(self.shards)


def save_buffer(buffer, directory, rows):
    directory = Path(directory)
    directory.mkdir()
    layout = {'capacity': buffer.batch_size, 'size': buffer._size, 'ptr': buffer._ptr,
              'rows': rows, 'device': str(buffer.device), 'fields': {}}
    for name, tensor in buffer.storage.items():
        parts = []
        row_bytes = tensor[0].numel() * tensor.element_size()
        chunk_rows = max(1, (16 * 1024 * 1024) // max(1, row_bytes))
        for start in range(0, rows, chunk_rows):
            end = min(rows, start+chunk_rows)
            filename = f'{name}-{start:06d}.pt'
            # A view, even contiguous, can serialize the entire backing storage.
            chunk = tensor[start:end].detach().to('cpu', copy=True).contiguous()
            torch.save(chunk, directory / filename)
            del chunk
            parts.append({'file': filename, 'start': start, 'end': end})
        layout['fields'][name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'parts': parts}
    write_json(directory / 'layout.json', layout)


def load_buffer(buffer, directory):
    directory = Path(directory)
    layout = json.loads((directory / 'layout.json').read_text())
    if layout['capacity'] != buffer.batch_size or layout['device'] != str(buffer.device):
        raise ValueError('Buffer capacity/device mismatch')
    rows, size, ptr = (int(layout[k]) for k in ('rows', 'size', 'ptr'))
    if not (0 <= size <= buffer.batch_size and 0 <= rows <= buffer.batch_size and 0 <= ptr < buffer.batch_size):
        raise ValueError('Invalid buffer ring metadata')
    if set(layout['fields']) != set(buffer.storage.keys()):
        raise ValueError('Buffer field mismatch')
    for name, tensor in buffer.storage.items():
        field = layout['fields'][name]
        if field['shape'] != list(tensor.shape) or field['dtype'] != str(tensor.dtype):
            raise ValueError(f'Buffer layout mismatch: {name}')
        cursor = 0
        for part in field['parts']:
            if part['start'] != cursor or not cursor < part['end'] <= rows:
                raise ValueError('Buffer shard ordering mismatch')
            chunk = torch.load(directory / part['file'], map_location='cpu', weights_only=True)
            target = tensor[part['start']:part['end']]
            if chunk.shape != target.shape or chunk.dtype != target.dtype:
                raise ValueError('Buffer shard shape/dtype mismatch')
            target.copy_(chunk)
            del chunk
            cursor = part['end']
        if cursor != rows:
            raise ValueError('Incomplete buffer')
    buffer._size, buffer._ptr = size, ptr


def verify_generation(directory, *, require_remote=False):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unknown generation schema')
    marker = json.loads((directory / ('COMPLETE.json' if require_remote else 'CAPTURED.json')).read_text())
    if marker['manifest_sha256'] != sha256(directory / 'manifest.json'):
        raise ValueError('Generation marker mismatch')
    if require_remote and marker.get('remote_verified') is not True:
        raise ValueError('No independent remote verification')
    required = {'train_state.pt', 'config.json', 'journal.json', 'replay/layout.json',
                'rollout/layout.json', 'commit/layout.json', 'source.tar.gz', 'environment.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('Incomplete generation roles')
    for name, record in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe artifact path')
        if (directory / relative).is_symlink() or file_record(directory / relative) != record:
            raise ValueError(f'Corrupt generation file: {name}')
    environment = json.loads((directory/'environment.json').read_text())
    if environment['native_filename'] not in manifest['files']:
        raise ValueError('Missing locked native runtime binary')
    for kind in ('replay', 'rollout', 'commit'):
        layout = json.loads((directory/kind/'layout.json').read_text())
        for field in layout['fields'].values():
            for part in field['parts']:
                if Path(part['file']).name != part['file'] or f"{kind}/{part['file']}" not in manifest['files']:
                    raise ValueError('Unmanifested buffer shard')
    journal = json.loads((directory/'journal.json').read_text())
    for part in journal['parts']:
        if part['path'] not in manifest['files'] or not part['path'].startswith('journal/'):
            raise ValueError('Unmanifested journal shard')
    if canonical_hash({k:file_record(directory/v['path']) for k,v in enumerate(journal['parts'])}) != journal['boundary_sha256']:
        raise ValueError('Journal prefix digest mismatch')
    return manifest


def capture(runtime, state, replay, ema):
    if runtime.world_size != 1 or str(replay.device) != 'cpu':
        raise ValueError('Full-state v1 requires single-process CPU replay')
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sampler = runtime.sampler
    iteration = state.iteration
    if ema.last_iteration != iteration or not runtime.cfg.resume_optimizer:
        raise ValueError('Capture requires a complete iteration and both optimizers')
    generation = runtime.checkpoint_dir / f'i{iteration:06d}-g{uuid.uuid4().hex}'
    temporary = generation.with_name('.' + generation.name)
    temporary.mkdir()
    root = Path(__file__).resolve().parents[2]
    with tarfile.open(temporary/'source.tar.gz', 'w:gz') as archive:
        for base in ('marvel', 'kagami', 'evaluations', 'nanopokerlib', 'experiments/colab_l4_500'):
            for path in sorted((root/base).rglob('*')):
                if path.is_file() and (path.suffix in {'.py', '.cpp', '.hpp', '.h', '.json'}
                                       or path.name == 'CMakeLists.txt'):
                    archive.add(path, arcname=str(path.relative_to(root)))
        archive.add(root/'pyproject.toml', arcname='pyproject.toml')
    import nanopokerlib
    native = Path(nanopokerlib._pybind.__file__)
    shutil.copyfile(native, temporary/native.name)
    write_json(temporary/'environment.json', {
        'python': sys.version, 'python_compiler': platform.python_compiler(),
        'native_filename': native.name, 'native_sha256': sha256(native),
        'distributions': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()
                                     if d.metadata['Name'])),
        'driver': subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,name','--format=csv,noheader'],
                                         text=True).strip() if torch.cuda.is_available() else None})
    for name in ('resolved_config.json', 'initial_models.json', 'metrics.jsonl', 'iterations.jsonl', 'attempt.json'):
        path = Path(runtime.cfg.logging_dir)/name
        if path.exists():
            (temporary/'run_artifacts').mkdir(exist_ok=True)
            shutil.copyfile(path, temporary/'run_artifacts'/name)
    config = asdict(runtime.cfg)
    payload = build_checkpoint_payload_from_state(state)
    payload['ema_updater'] = ema.state_dict()
    payload['rng'] = rng_state()
    payload['provenance'] = provenance()
    payload['modes'] = {key: {name: module.training for name,module in obj.named_modules()}
                        for key,obj in state.entries.items() if isinstance(obj,torch.nn.Module)}
    payload['sampler_config'] = sampler_signature(sampler)
    payload['sampler'] = {key: getattr(sampler, key) for key in
                          ('commit_buffer_size', 'total_env_steps', 'total_game_steps', 'total_completed_games')}
    payload['sampler']['observation'] = {k: v.detach().cpu() for k,v in sampler.parsed_envs_step._observation.items()}
    payload['sampler']['outcome'] = {k: v.detach().cpu() for k,v in sampler.parsed_envs_step.outcome.data.items()}
    payload['sampler']['last_step_rewards'] = sampler.last_step_rewards
    payload['sampler']['last_step_is_terminal'] = sampler.last_step_is_terminal
    payload['train_config'] = config
    payload['logical_counters'] = {
        'consumed_trajectories': iteration*runtime.cfg.collect_batch_size,
        'native_env_steps': sampler.total_env_steps, 'valid_game_steps': sampler.total_game_steps,
        'completed_games': sampler.total_completed_games, 'commit_leftovers': sampler.commit_buffer_size,
        'actor_optimizer_steps': state['actor_optimizer_steps'],
        'critic_optimizer_steps': state['critic_optimizer_steps'], 'ema_updates': ema.update_count}
    payload['parent_manifest_sha256'] = getattr(runtime, 'parent_manifest_sha256', None)
    torch.save(payload, temporary / 'train_state.pt')
    write_json(temporary / 'config.json', config)
    save_buffer(replay, temporary / 'replay', replay._size)
    save_buffer(sampler.rollout_buffer, temporary / 'rollout', sampler.rollout_batch_size)
    save_buffer(sampler.commit_buffer, temporary / 'commit', sampler.commit_buffer_size)
    journal_dir = temporary / 'journal'
    journal_dir.mkdir()
    parts = []
    for part in sampler.action_journal.seal():
        filename = f"journal/j{part['first']:012d}.bin"
        try:
            os.link(part['path'], temporary / filename)
        except OSError:
            shutil.copyfile(part['path'], temporary / filename)
        parts.append({**part, 'path': filename})
    write_json(temporary / 'journal.json', {'seed': sampler.action_journal.seed,
               'R': sampler.rollout_batch_size, 'count': sampler.action_journal.count,
               'dtype': '<i8', 'shape': [sampler.rollout_batch_size, 5], 'parts': parts,
               'encoding': 'raw_int64_little_endian_no_header',
               'logical_action_bytes': sampler.action_journal.count*sampler.rollout_batch_size*5*8,
               'final_shard_valid_byte_offset': parts[-1]['bytes'] if parts else 0,
               'boundary_sha256': canonical_hash({k: file_record(temporary / v['path']) for k,v in enumerate(parts)})})
    manifest = {'schema': SCHEMA, 'completed_iteration': iteration, 'next_iteration': iteration+1,
                'effective_config_sha256': canonical_hash(config),
                'scientific_config_sha256': canonical_hash(scientific_config(config)),
                'provenance': payload['provenance'], 'files': {}}
    for path in sorted(temporary.rglob('*')):
        if path.is_file():
            manifest['files'][str(path.relative_to(temporary))] = file_record(path)
    write_json(temporary / 'manifest.json', manifest)
    write_json(temporary / 'CAPTURED.json', {'manifest_sha256': sha256(temporary / 'manifest.json'),
                                          'remote_verified': False, 'resume_capability': 'FULL_STATE_UNVERIFIED'})
    temporary.rename(generation)
    verify_generation(generation)
    runtime.last_full_generation = generation
    print(f'[FullState] captured {generation}', flush=True)
    return generation


def restore(runtime, state, replay, ema, directory):
    directory = Path(directory)
    manifest = verify_generation(directory)
    config = json.loads((directory / 'config.json').read_text())
    full_config(config)
    if canonical_hash(config) != manifest['effective_config_sha256']:
        raise ValueError('Config digest mismatch')
    if canonical_hash(scientific_config(config)) != manifest['scientific_config_sha256']:
        raise ValueError('Scientific config digest mismatch')
    if scientific_config(asdict(runtime.cfg)) != scientific_config(config):
        raise ValueError('Scientific config mismatch; resolve the complete saved config first')
    if runtime.cfg.total_iterations < manifest['completed_iteration'] or not runtime.cfg.resume_optimizer:
        raise ValueError('Invalid full-state continuation target/optimizer policy')
    if provenance() != manifest['provenance']:
        raise ValueError('Execution/environment fingerprint mismatch')
    payload = torch.load(directory / 'train_state.pt', map_location='cpu', weights_only=True)
    for key in (*state.entries, 'ema_updater', 'rng', 'sampler', 'modes', 'provenance', 'train_config',
                'sampler_config', 'logical_counters'):
        if key not in payload:
            raise ValueError(f'Missing required full-state field: {key}')
    if payload['iteration'] != manifest['completed_iteration'] or payload['train_config'] != config:
        raise ValueError('State/manifest mismatch')
    sampler = runtime.sampler
    if payload['sampler_config'] != sampler_signature(sampler):
        raise ValueError('Actual sampler configuration mismatch')
    journal = json.loads((directory / 'journal.json').read_text())
    if (journal['seed'] != sampler._env_seed or journal['R'] != sampler.rollout_batch_size
            or journal['dtype'] != '<i8' or journal['shape'] != [sampler.rollout_batch_size,5]):
        raise ValueError('Journal origin mismatch')
    result = sampler.envs.reset(journal['seed'])
    cursor = 0
    for part in journal['parts']:
        if part['first'] != cursor or part['count'] <= 0:
            raise ValueError('Journal ordering mismatch')
        path = directory / part['path']
        record_size = journal['R'] * 5 * 8
        if file_record(path) != {k:part[k] for k in ('bytes','sha256')} or path.stat().st_size != part['count']*record_size:
            raise ValueError('Journal size/hash mismatch')
        with path.open('rb') as stream:
            for _ in range(part['count']):
                actions = np.frombuffer(stream.read(record_size), dtype='<i8').reshape(journal['R'], 5).copy()
                result = sampler.envs.step(sampler.game.actions_from_np(actions))
        sampler.action_journal.shards.append({**part, 'path': str(path.resolve())})
        cursor += part['count']
    if cursor != journal['count'] or cursor*journal['R'] != payload['sampler']['total_env_steps']:
        raise ValueError('Journal prefix count mismatch')
    sampler.action_journal.count = cursor
    parsed = EnvStepBatch.from_env_step(result, sampler.config)
    saved_sampler = payload['sampler']
    for actual, saved in ((parsed._observation, saved_sampler['observation']), (parsed.outcome.data, saved_sampler['outcome'])):
        if set(actual.keys()) != set(saved):
            raise ValueError('Native boundary fields mismatch')
        for key, value in actual.items():
            if not torch.equal(value.cpu(), saved[key]):
                raise ValueError(f'Native boundary mismatch: {key}')
    sampler.parsed_envs_step = parsed
    load_checkpoint_into_state(payload, state)
    expected_modes={key for key,obj in state.entries.items() if isinstance(obj,torch.nn.Module)}
    if set(payload['modes']) != expected_modes:
        raise ValueError('Incomplete model training modes')
    for key, modes in payload['modes'].items():
        modules=dict(state[key].named_modules())
        if set(modes)!=set(modules) or any(type(v) is not bool for v in modes.values()):
            raise ValueError('Incomplete submodule training modes')
        for name,training in modes.items():
            modules[name].training=training
    ema.load_state_dict(payload['ema_updater'])
    if ema.last_iteration != state.iteration:
        raise ValueError('EMA iteration mismatch')
    load_buffer(replay, directory / 'replay')
    load_buffer(sampler.rollout_buffer, directory / 'rollout')
    load_buffer(sampler.commit_buffer, directory / 'commit')
    for key in ('commit_buffer_size', 'total_env_steps', 'total_game_steps', 'total_completed_games'):
        setattr(sampler, key, saved_sampler[key])
    for key in ('last_step_rewards', 'last_step_is_terminal'):
        value = saved_sampler[key]
        setattr(sampler, key, value.to(sampler.device) if value is not None else None)
    if replay._size != min(replay.batch_size, state.iteration*runtime.cfg.collect_batch_size):
        raise ValueError('Replay logical size mismatch')
    if replay._ptr != (state.iteration*runtime.cfg.collect_batch_size) % replay.batch_size:
        raise ValueError('Replay ring pointer mismatch')
    for kind, rows in (('replay', replay._size), ('rollout', sampler.rollout_batch_size),
                       ('commit', sampler.commit_buffer_size)):
        if json.loads((directory/kind/'layout.json').read_text())['rows'] != rows:
            raise ValueError('Stored buffer region mismatch')
    if sampler.total_completed_games != state.iteration*runtime.cfg.collect_batch_size + sampler.commit_buffer_size:
        raise ValueError('Completed/consumed/leftover counts disagree')
    expected_counters={
        'consumed_trajectories': state.iteration*runtime.cfg.collect_batch_size,
        'native_env_steps': sampler.total_env_steps, 'valid_game_steps': sampler.total_game_steps,
        'completed_games': sampler.total_completed_games, 'commit_leftovers': sampler.commit_buffer_size,
        'actor_optimizer_steps': state['actor_optimizer_steps'],
        'critic_optimizer_steps': state['critic_optimizer_steps'], 'ema_updates': ema.update_count}
    if payload['logical_counters'] != expected_counters:
        raise ValueError('Logical counters mismatch')
    sampler.prepare_agent_cache(agent_model=state['agent_resume_state_dict'])
    runtime.sync_step_totals_from_sampler()
    runtime.start_iteration = state.iteration+1
    runtime.logger.progress_bar.update(min(state.iteration,runtime.cfg.total_iterations))
    runtime.parent_manifest_sha256 = sha256(directory / 'manifest.json')
    runtime._state = state
    restore_rng(payload['rng'])
    print(f'[FullState] loaded_iteration={state.iteration} next_iteration={runtime.start_iteration}', flush=True)
