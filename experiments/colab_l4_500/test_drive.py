"""Local protocol tests are not evidence of live independent Drive verification."""
import json
from pathlib import Path
import tarfile
import pytest
from experiments.colab_l4_500.drive import verify_downloaded_archive, verify_mount
from marvel.runtime.full_state import sha256


def test_wrong_mount_rejected(tmp_path):
    (tmp_path/'campaign_identity.json').write_text(json.dumps({'parent_folder_id':'wrong','campaign_folder_id':'wrong','identity_nonce':'wrong'}))
    with pytest.raises(ValueError):
        verify_mount(tmp_path)


def test_archive_integrity_and_path_traversal_rejected(tmp_path):
    archive=tmp_path/'snapshot.tar.gz'
    data=tmp_path/'data'
    data.write_text('payload')
    with tarfile.open(archive,'w:gz') as bundle:
        bundle.add(data,arcname='../outside')
    record={'archive_bytes':archive.stat().st_size,'archive_sha256':sha256(archive)}
    with pytest.raises(ValueError,match='Unsafe'):
        verify_downloaded_archive(archive,record,tmp_path/'extracted')
    record['archive_sha256']='0'*64
    with pytest.raises(ValueError,match='hash mismatch'):
        verify_downloaded_archive(archive,record,tmp_path/'corrupt')
