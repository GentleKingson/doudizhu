"""Mount staging with independent Drive-download verification; COMPLETE is always last."""
import json
from pathlib import Path
import shutil
import tarfile
import uuid

from marvel.runtime.full_state import sha256, verify_generation, write_json

CAMPAIGN='l4-vrpo-500-20261018-v1'
PARENT_ID='14wacgMnsJwnldEKydfQ5TFCUb5EOGTEO'
CAMPAIGN_ID='1Z4eSlVuPQIt8y17fZ1TVjEICEYeLDDtz'
NONCE='f34f641a-47ab-4f94-a148-5aff54659043'


def verify_mount(root):
    root=Path(root)
    identity=json.loads((root/'campaign_identity.json').read_text())
    if (identity['parent_folder_id'],identity['campaign_folder_id'],identity['identity_nonce']) != (PARENT_ID,CAMPAIGN_ID,NONCE):
        raise ValueError('Drive mount is not the verified campaign folder')
    return root


def stage_generation(directory,remote_checkpoint_root):
    directory=Path(directory)
    manifest=verify_generation(directory)
    receipt_path=directory.parent/(directory.name+'.stage.json')
    archive=directory.parent/(directory.name+'.tar.gz')
    if not archive.exists():
        temporary=archive.with_name('.'+archive.name+'.partial')
        with tarfile.open(temporary,'w:gz') as bundle:
            for path in sorted(directory.rglob('*')):
                if path.is_file():
                    bundle.add(path,arcname=str(path.relative_to(directory)),recursive=False)
        temporary.rename(archive)
    remote=Path(remote_checkpoint_root)/directory.name
    remote.mkdir(parents=True,exist_ok=True)
    manifest_sha=sha256(directory/'manifest.json')
    pending=remote/'UPLOAD_PENDING.json'
    if pending.exists() and json.loads(pending.read_text())['manifest_sha256']!=manifest_sha:
        raise ValueError('Immutable generation collision')
    record={'generation':directory.name,'remote_directory':str(remote),
        'completed_iteration':manifest['completed_iteration'],'manifest_sha256':manifest_sha,
        'archive_sha256':sha256(archive),'archive_bytes':archive.stat().st_size,
        'status':'UPLOAD_PENDING','remote_verified':False}
    if (remote/'COMPLETE.json').exists():
        complete=json.loads((remote/'COMPLETE.json').read_text())
        if complete['manifest_sha256']!=manifest_sha or complete['archive_sha256']!=record['archive_sha256']:
            raise ValueError('Attempt to overwrite a completed generation')
        return record
    write_json(pending,record)
    # Mount rename/fsync is not claimed to be a network transaction.
    shutil.copyfile(archive,remote/'snapshot.tar.gz')
    shutil.copyfile(directory/'manifest.json',remote/'manifest.json')
    write_json(receipt_path,record)
    print('DRIVE_UPLOAD_PENDING '+json.dumps(record),flush=True)
    return record


def verify_downloaded_archive(archive,record,destination):
    """Only call on bytes obtained independently through Drive API/connector, outside the mount cache."""
    archive,destination=Path(archive),Path(destination)
    if archive.stat().st_size!=record['archive_bytes'] or sha256(archive)!=record['archive_sha256']:
        raise ValueError('Independent downloaded archive size/hash mismatch')
    destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive,'r:gz') as bundle:
        names=set()
        for member in bundle:
            relative=Path(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts or member.name in names:
                raise ValueError('Unsafe or duplicate archive member')
            names.add(member.name)
            path=destination/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as source,path.open('xb') as target:
                shutil.copyfileobj(source,target,length=4*1024*1024)
    manifest=verify_generation(destination)
    if sha256(destination/'manifest.json')!=record['manifest_sha256']:
        raise ValueError('Independent downloaded manifest mismatch')
    return manifest


def complete_generation(record,downloaded_archive,remote_file_id,verification_root):
    if not remote_file_id:
        raise ValueError('Drive file identity is required')
    verified=Path(verification_root)/uuid.uuid4().hex
    manifest=verify_downloaded_archive(downloaded_archive,record,verified)
    complete={'remote_verified':True,'manifest_sha256':record['manifest_sha256'],
        'archive_sha256':record['archive_sha256'],'archive_bytes':record['archive_bytes'],
        'drive_archive_file_id':remote_file_id,'completed_iteration':manifest['completed_iteration'],
        'execution_sha':manifest['provenance']['execution_sha'],
        'verification_method':'independent Drive connector/API download, archive SHA256 and all expanded file SHA256',
        'resume_capability':'FULL_STATE_UNVERIFIED'}
    # Integrity verification does not promote restoration-equivalence capability.
    remote=Path(record['remote_directory'])
    if (remote/'COMPLETE.json').exists():
        if json.loads((remote/'COMPLETE.json').read_text())!=complete:
            raise ValueError('Immutable COMPLETE collision')
    else:
        write_json(remote/'COMPLETE.json',complete)
    write_json(verified/'COMPLETE.json',complete)
    verify_generation(verified,require_remote=True)
    return {'complete':complete,'verified_directory':str(verified)}
