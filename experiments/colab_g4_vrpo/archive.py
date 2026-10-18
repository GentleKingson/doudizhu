from pathlib import Path
import hashlib,json,shutil
root=Path('/content/g4-vrpo')
done=json.loads((root/'done.json').read_text())
assert done['success'], (root/'results/failure.txt').read_text()
assert not list(root.rglob('*.pt')), 'Unexpected training checkpoint'
shutil.copy2(root/'search.log',root/'results/search.log')
shutil.copy2(root/'done.json',root/'results/done.json')
archive=Path(shutil.make_archive(str(root/'results'),'zip',root/'results'))
print(json.dumps({'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'done':done}))
