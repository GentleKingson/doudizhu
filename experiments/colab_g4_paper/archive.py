from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/content/g4-paper')
done=json.loads((root/'done.json').read_text())
assert done['success'], (root/'results/failure.txt').read_text()
assert not list(root.rglob('*.pt')), 'Unexpected training checkpoint'
shutil.copy2(root/'search.log',root/'results/search.log')
shutil.copy2(root/'done.json',root/'results/done.json')
repo=root/'repo'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
manifest={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines() if (repo/name).is_file()}
(root/'results/source-manifest.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(), tracked_changes='', hashes=manifest),indent=2))
for name in ('benchmark.py','run_search.py'):
 shutil.copy2(root/name,root/'results'/name)
archive=Path(shutil.make_archive(str(root/'results'),'zip',root/'results'))
print(json.dumps({'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'done':done}))
