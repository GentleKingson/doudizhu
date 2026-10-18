"""The delivered adapter must keep paper learning settings and Drive controls."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('paper_timed', Path(__file__).with_name('timed_training.py'))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

class PaperConfigTest(unittest.TestCase):
    def test_actual_config_passed_to_original_trainer(self):
        observed = {}
        def config(**kwargs):
            observed.update(kwargs)
            return SimpleNamespace(**kwargs, vrpo=SimpleNamespace(replay_buffer_device='cuda'))
        class Trainer:
            def __init__(self, cfg, runtime_factory):
                self.cfg = cfg
            def run(self):
                observed['replay_device'] = self.cfg.vrpo.replay_buffer_device
        modules = {}
        for name, attrs in {
            'torch': dict(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None),
            'marvel.config': dict(TrainConfig=config),
            'marvel.training.vrpo': dict(VRPOTrainer=Trainer),
            'marvel.utils': dict(configure_torch=lambda cfg: None),
        }.items():
            m = ModuleType(name)
            m.__dict__.update(attrs)
            modules[name] = m
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), patch.object(adapter, 'timed_workflow', return_value=None):
            adapter.main(dict(threads=6, checkpoint_dir=tmp, rollout=512, collect=2048,
                              mini_batches=4, logging_dir=tmp))
        expected = dict(algo_name='vrpo', rollout_batch_size=512, collect_batch_size=2048,
                        mini_batches=4, replay_buffer_scale=64, agent_merging_beta=.999,
                        lr_stable_iterations=5000, reg_stable_iterations=500,
                        total_iterations=40000, replay_device='cpu')
        for key, value in expected.items():
            self.assertEqual(observed[key], value, key)

    def test_notebook_uses_tested_adapter(self):
        nb = json.loads((ROOT/'notebooks/colab_a100_highram_vrpo_centralcritic.ipynb').read_text())
        self.assertEqual(''.join(nb['cells'][7]['source']).split('\n', 1)[1],
                         Path(__file__).with_name('timed_training.py').read_text())
        ns = {}
        exec(''.join(nb['cells'][3]['source']), ns)
        self.assertEqual(ns['BEST']['rollout'], 512)
        self.assertEqual(ns['BEST']['collect'], 2048)
        self.assertEqual(ns['BEST']['mini_batches'], 4)
        self.assertEqual(ns['SAVE_EVERY_MINUTES'], 30)
        self.assertEqual(ns['KEEP_CHECKPOINTS'], 10)

if __name__ == '__main__':
    unittest.main()
