"""Offline checks only; Colab GPU/Drive authorization is tested separately."""
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import sys

NOTEBOOK = Path(__file__).resolve().parents[2] / 'notebooks/colab_a100_highram_vrpo_centralcritic.ipynb'

class NotebookTest(unittest.TestCase):
    def setUp(self):
        self.nb = json.loads(NOTEBOOK.read_text())

    def test_code_and_adapter(self):
        self.assertEqual(self.nb['nbformat'], 4)
        for i, cell in enumerate(self.nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if source.startswith('%%writefile'):
                    source = source.split('\n', 1)[1]
                    original = NOTEBOOK.parents[1] / 'experiments/colab_a100_paper/timed_training.py'
                    self.assertEqual(source.strip(), original.read_text().strip())
                compile(source, f'cell-{i}', 'exec')
        ns = {}
        exec(''.join(self.nb['cells'][3]['source']), ns)
        self.assertEqual(ns['SAVE_EVERY_MINUTES'], 30)
        self.assertEqual(ns['KEEP_CHECKPOINTS'], 10)
        self.assertGreater(ns['TRAINING_MINUTES'], 0)

    def test_release_order(self):
        source = ''.join(self.nb['cells'][8]['source'])
        for exit_code, flush_fails in [(0, False), (1, False), (0, True)]:
            events = []
            def flush(**kwargs):
                events.append('flush')
                if flush_fails:
                    raise OSError('Drive sync failed')
            colab = ModuleType('google.colab')
            colab.drive = SimpleNamespace(flush_and_unmount=flush)
            colab.runtime = SimpleNamespace(unassign=lambda: events.append('unassign'))
            with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'google': ModuleType('google'), 'google.colab': colab}):
                root = Path(tmp)
                (root/'run').mkdir()
                # Redirect the two notebook-local settings/log paths into the test directory.
                code = source.replace('/content/vrpo-training-settings.json', str(root/'settings.json'))
                ns = dict(BEST={}, TRAINING_MINUTES=1, SAVE_EVERY_MINUTES=30, KEEP_CHECKPOINTS=10,
                    RESUME_FROM='', CHECKPOINT_DIR=root/'run', Path=Path, RUN_NAME='run', SEED=0,
                    json=json, DRIVE_ROOT=root, COMMIT='test', os=SimpleNamespace(environ={}),
                    REPO=root, sys=sys, DISCONNECT_WHEN_DONE=True,
                    subprocess=SimpleNamespace(Popen=lambda *a, **k: SimpleNamespace(wait=lambda: exit_code)))
                if exit_code or flush_fails:
                    with self.assertRaises((RuntimeError, OSError)):
                        exec(code, ns)
                else:
                    exec(code, ns)
            self.assertEqual(events, ['flush'] if flush_fails else ['flush', 'unassign'])

if __name__ == '__main__':
    unittest.main()
