"""Install the existing evaluator's pinned baselines into an isolated Python 3.7 runtime."""
from pathlib import Path
import os
import subprocess
import tarfile
import urllib.request

PERFECTDOU_SHA='594404922ee3810e2d84b80bb2c2846cb20e5390'


def main():
    root=Path(__file__).resolve().parents[2]
    external=root/'external/PerfectDou'
    if not external.exists():
        subprocess.run(['git','clone','https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou.git',str(external)],check=True)
    subprocess.run(['git','checkout',PERFECTDOU_SHA],cwd=external,check=True)
    mamba=Path('/content/l4-tools/bin/micromamba')
    if not mamba.exists():
        archive=Path('/content/l4-tools/micromamba.tar.bz2')
        archive.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve('https://micro.mamba.pm/api/micromamba/linux-64/latest',archive)
        with tarfile.open(archive,'r:bz2') as bundle:
            member=bundle.getmember('bin/micromamba')
            assert member.isfile() and 0 < member.size < 64*1024*1024
            mamba.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as source,mamba.open('wb') as target:
                target.write(source.read())
        mamba.chmod(0o755)
    prefix=Path('/content/perfectdou37')
    env={**os.environ,'MAMBA_ROOT_PREFIX':'/content/l4-mamba'}
    if not (prefix/'bin/python').exists():
        subprocess.run([str(mamba),'create','-y','-p',str(prefix),'-c','conda-forge','python=3.7','pip'],env=env,check=True)
    python=str(prefix/'bin/python')
    subprocess.run([python,'-m','pip','install','numpy==1.21.6','onnxruntime==1.7.0'],check=True)
    check="import sys; sys.path.insert(0,'.'); import numpy, onnxruntime; from perfectdou.env.encode import encode_obs_landlord; from perfectdou.evaluation.perfectdou_agent import PerfectDouAgent; [PerfectDouAgent(x) for x in ('landlord','landlord_up','landlord_down')]; print('PerfectDou three model sessions PASS',numpy.__version__,onnxruntime.__version__)"
    subprocess.run([python,'-c',check],cwd=external,check=True)
    print('PERFECTDOU_SETUP_PASS',flush=True)


if __name__=='__main__':
    main()
