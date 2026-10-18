import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.localserver_5070.launch_training import build_plan
from experiments.colab_a100_paper.timed_training import publish_checkpoint, checkpoints


class LauncherTests(unittest.TestCase):
    def test_defaults_and_custom_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'source.pt'; checkpoint.write_bytes(b'source')
            output=Path(directory)/'new'
            _,plan=build_plan(['--resume-from',str(checkpoint),'--output',str(output),'--training-minutes','45','--dry-run'])
            self.assertEqual(plan['save_every_minutes'],30)
            self.assertEqual(plan['keep_checkpoints'],10)
            self.assertEqual(plan['runtime_max_seconds'],3600)
            self.assertNotIn('--deadline',plan['command'])
            self.assertIn('--nonblock',plan['command'])
            self.assertFalse(output.exists())
            for flags in [['--training-minutes','nan'],['--save-every-minutes','0'],['--keep-checkpoints','0'],['--output',directory]]:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    build_plan(['--resume-from',str(checkpoint)]+flags)

    def test_atomic_save_retention_and_failure(self):
        class Manager:
            def save_from_state(self,state,checkpoint_path,upload_to_wandb):
                checkpoint_path.write_text(str(state.iteration))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); unrelated=root/'notes.txt'; unrelated.write_text('keep')
            for i in range(1,5):publish_checkpoint(Manager(),SimpleNamespace(iteration=i),root,2)
            self.assertEqual([p.name for p in checkpoints(root)],['ckpt_3.pt','ckpt_4.pt'])
            self.assertEqual(unrelated.read_text(),'keep')
            with self.assertRaises(FileExistsError):publish_checkpoint(Manager(),SimpleNamespace(iteration=4),root,2)
            class FailedManager:
                def save_from_state(self,*args,**kwargs):raise OSError('test save failure')
            with self.assertRaises(OSError):publish_checkpoint(FailedManager(),SimpleNamespace(iteration=5),root,2)
            self.assertEqual([p.name for p in checkpoints(root)],['ckpt_3.pt','ckpt_4.pt'])
            self.assertFalse(list(root.glob('*.partial')))


if __name__=='__main__':unittest.main()
