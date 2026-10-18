import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('timed_training', Path(__file__).with_name('timed_training.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeManager:
    def save_from_state(self, state, checkpoint_path, upload_to_wandb):
        assert not upload_to_wandb
        checkpoint_path.write_bytes(b'checkpoint-' + str(state.iteration).encode())

class CheckpointTest(unittest.TestCase):
    def test_rotation_numeric_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'notes.pt').write_bytes(b'keep')
            (folder/'ckpt_1.pt').mkdir()
            for step in (2, 9, 10):
                module.publish_checkpoint(FakeManager(), SimpleNamespace(iteration=step), folder, 2)
            self.assertEqual([p.name for p in module.checkpoints(folder)], ['ckpt_9.pt', 'ckpt_10.pt'])
            self.assertTrue((folder/'notes.pt').exists())
            self.assertTrue((folder/'ckpt_1.pt').is_dir())
            with self.assertRaises(FileExistsError):
                module.publish_checkpoint(FakeManager(), SimpleNamespace(iteration=10), folder, 2)

    def test_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            module.publish_checkpoint(FakeManager(), SimpleNamespace(iteration=1), folder, 1)
            with patch.object(module.shutil, 'copyfile', side_effect=OSError('Drive disconnected')):
                with self.assertRaises(OSError):
                    module.publish_checkpoint(FakeManager(), SimpleNamespace(iteration=2), folder, 1)
            self.assertEqual([p.name for p in module.checkpoints(folder)], ['ckpt_1.pt'])
            self.assertFalse(list(folder.glob('*.partial')))

    def test_invalid_retention(self):
        for keep in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                module.publish_checkpoint(None, None, '.', keep)

class ResumeTest(unittest.TestCase):
    def test_requires_full_vrpo_state(self):
        payload = dict(agent_resume_state_dict={}, agent_state_dict={}, critic_state_dict={},
            agent_optimizer_state_dict={}, critic_optimizer_state_dict={}, iteration=42,
            train_config={'algo_name': 'vrpo'})
        module.validate_resume_payload(payload)
        del payload['critic_optimizer_state_dict']
        with self.assertRaises(ValueError):
            module.validate_resume_payload(payload)
        payload['critic_optimizer_state_dict'] = {}
        payload['train_config']['algo_name'] = 'ippo'
        with self.assertRaises(ValueError):
            module.validate_resume_payload(payload)

class TimerTest(unittest.TestCase):
    def test_periodic_and_final_at_completed_iteration(self):
        from types import ModuleType
        import sys
        fake = ModuleType('marvel.training.workflow')
        class Workflow:
            def __init__(self, cfg, **kwargs):
                self.logger = SimpleNamespace(progress_bar=SimpleNamespace(total=20))
                self.checkpoint_dir = kwargs['checkpoint_dir']
                self.checkpoint_manager = FakeManager()
                self._state = SimpleNamespace(iteration=0)
                self.current_iteration = None
            def __iter__(self):
                return self.iterations()
            def iterations(self):
                for i in range(1, 20):
                    self.current_iteration = i
                    self._state.iteration = i
                    yield i
        fake.TrainingWorkflow = Workflow
        now = [0.0]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'torch': SimpleNamespace(get_num_threads=lambda: 12), 'marvel.training.workflow': fake}), patch.object(module.time, 'monotonic', side_effect=lambda: now[0]):
            cfg = SimpleNamespace(game=None)
            cls = module.timed_workflow(dict(threads=1, training_minutes=1, save_every_minutes=.4, keep_checkpoints=2, checkpoint_dir=tmp))
            runtime = cls(cfg)
            completed = []
            for i in runtime:
                completed.append(i)
                now[0] += 20
                if runtime.checkpoint_due():
                    runtime.save_checkpoint(runtime._state)
            self.assertEqual(completed, [1, 2, 3])
            self.assertEqual([p.name for p in module.checkpoints(tmp)], ['ckpt_2.pt', 'ckpt_3.pt'])

if __name__ == '__main__':
    unittest.main()
