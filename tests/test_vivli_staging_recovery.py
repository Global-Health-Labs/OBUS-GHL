import csv
import io
import sys
import zipfile

import pytest

from ghlobus.ingestion import stage_vivli_for_preproc as stage


@pytest.mark.parametrize('kind', ['copy', 'zip'])
def test_partial_and_same_size_corrupt_files_are_repaired(tmp_path, kind):
    source = tmp_path / 'source'
    source.write_bytes(b'complete video')
    archive = tmp_path / 'cohort.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('video', source.read_bytes())
    destination = tmp_path / 'dest'
    def run():
        return (stage.ensure_copied(source, destination) if kind == 'copy'
                else stage.ensure_extracted(archive, 'video', destination))
    for bad in [b'x', b'x' * source.stat().st_size]:
        destination.write_bytes(bad)
        assert run() != 'already_present'
        assert destination.read_bytes() == source.read_bytes()
        assert run() == 'already_present'


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('kind', ['copy', 'zip'])
def test_failed_copy_never_publishes_partial_file(tmp_path, monkeypatch, existing, kind):
    source = tmp_path / 'source'
    source.write_bytes(b'complete video')
    destination = tmp_path / 'dest'
    if existing:
        destination.write_bytes(b'previous')
    def interrupt(src, dst):
        dst.write(b'partial')
        raise OSError('interrupted')
    archive = tmp_path / 'archive.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('video', source.read_bytes())
    monkeypatch.setattr(stage.shutil, 'copyfileobj', interrupt)
    with pytest.raises(OSError, match='interrupted'):
        if kind == 'copy':
            stage.ensure_copied(source, destination)
        else:
            stage.ensure_extracted(archive, 'video', destination)
    if existing:
        assert destination.read_bytes() == b'previous'
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob('*.tmp'))


def test_duplicate_destinations_reported_without_duplicate_metadata(tmp_path, monkeypatch):
    structured = tmp_path / 'structured.zip'
    content = io.StringIO()
    writer = csv.DictWriter(content, fieldnames=['pidscan', 'pid', 'file', 'studydate', 'studytime'])
    writer.writeheader()
    writer.writerow(dict(pidscan='FA3-001-1', pid='FA3-001', file='video.dcm',
                         studydate='20230101', studytime='010101'))
    with zipfile.ZipFile(structured, 'w') as zf:
        zf.writestr('C3_INSTANCE_TABLE.csv', content.getvalue())
    roots = []
    for name in ['Cohort1', 'Cohort2']:
        root = tmp_path / name
        study = root / 'FA3-001-1_20230101_010101'
        study.mkdir(parents=True)
        (study / 'video.dcm').write_bytes(name.encode())
        roots.append(root)
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['stage', '--data-dir', str(tmp_path),
        '--structured-zip', str(structured), '--cohort-dir', str(roots[0]),
        '--cohort-dir', str(roots[1]), '--raw-root', str(tmp_path / 'raw'),
        '--out-root', str(out), '--stage-mode', 'copy'])
    stage.main()
    sheets = out / 'sheets'
    with (sheets / stage.DEFAULT_METADATA_FILE).open() as handle:
        assert len(list(csv.DictReader(handle))) == 1
    with (sheets / stage.DEFAULT_MANIFEST_FILE).open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[1]['stage_action'] == 'duplicate_destination_conflict'
    assert rows[1]['conflict_with'] == rows[0]['source_path']
    assert (sheets / stage.DEFAULT_SUMMARY_FILE).exists()
