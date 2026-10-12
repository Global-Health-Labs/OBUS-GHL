"""Fault injection around real preprocessing orchestration; no media/cloud needed."""
import subprocess
import sys
import types

import pandas as pd
import pytest
import yaml

from ghlobus.ingestion import preprocess_data_v9 as pre


@pytest.fixture
def run_setup(tmp_path, monkeypatch):
    sheets = tmp_path / 'sheets'
    sheets.mkdir()
    df = pd.DataFrame({'StudyID': ['S1'], 'relpath': ['P/Ultrasound/2023-01/S1'],
                       'filename': ['1.2.840.12345'], 'Manufacturer': ['M'],
                       'ManufacturerModelName': ['Model'], 'file_type': ['dcm'],
                       'tag': ['M'], 'PhysicalDeltaX': ['0.1']})
    df.to_csv(sheets / 'input.csv', index=False)
    info = {'input': {'raw_dir': str(tmp_path / 'raw'), 'project': 'P', 'it_file': 'input.csv'},
            'output': {'out_dir': str(tmp_path), 'out_file': 'prototype.csv', 'folders': {'pt': 'data'}},
            'image': {'alpha': 0.075, 'channels': 1, 'img_size': 288},
            'processing': {'cores': 1, 'batch_size': 1000,
                           'post_batch': {'enabled': True, 'command': ['upload'],
                                          'delete_local_outputs': True,
                                          'handoff_final_outputs': False}}}
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(info))
    monkeypatch.setattr(sys, 'argv', ['preprocess', '--yaml', str(config)])
    artifact = tmp_path / 'data/P/S1_M_Model/1.2.840.12345.pt'
    decodes, uploads = [], []
    def decoder(row, **kwargs):
        decodes.append(row['in_filepath'])
        output = (tmp_path / 'data' / row['project'] / row['exam_dir']
                  / (pre.output_file_base_name(row['in_filepath']) + '.pt'))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'synthetic tensor')
        return pd.DataFrame({'fail_reason': ['Good_video']})
    # Replace only media decoding, preserving the actual entry point, CSVs and checkpoints.
    module = types.ModuleType('ghlobus.utilities.data_utils')
    module.read_spreadsheet = lambda path: pd.read_csv(path, dtype=str, keep_default_na=False)
    module.preprocess_video = decoder
    module.create_video_df = lambda **kwargs: pd.DataFrame({'fail_reason': [kwargs['reason']]})
    monkeypatch.setitem(sys.modules, 'ghlobus.utilities.data_utils', module)
    monkeypatch.setattr(pre, 'run_post_batch_command', lambda *args: uploads.append(args[1]))
    return tmp_path, artifact, decodes, uploads


def test_failed_upload_replayed_without_reprocessing(run_setup, monkeypatch):
    root, artifact, decodes, uploads = run_setup
    def fail(*args):
        uploads.append(args[1])
        raise subprocess.CalledProcessError(1, 'upload')
    monkeypatch.setattr(pre, 'run_post_batch_command', fail)
    with pytest.raises(subprocess.CalledProcessError):
        pre.main()
    assert artifact.exists()
    assert not (root / 'sheets/prototype.csv').exists()
    monkeypatch.setattr(pre, 'run_post_batch_command', lambda *args: uploads.append(args[1]))
    pre.main()
    assert len(decodes) == 1
    assert uploads == [0, 0]
    assert not artifact.exists()
    assert (root / 'sheets/prototype.csv').exists()
    pre.main()
    assert len(decodes) == 1
    assert uploads == [0, 0]  # successful receipt prevents another upload


def test_crash_during_cleanup_resumes_cleanup_only(run_setup, monkeypatch):
    root, artifact, decodes, uploads = run_setup
    delete = pre.delete_batch_outputs
    monkeypatch.setattr(pre, 'delete_batch_outputs', lambda *args: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError, match='crash'):
        pre.main()
    assert artifact.exists()
    monkeypatch.setattr(pre, 'delete_batch_outputs', delete)
    pre.main()
    assert uploads == [0]
    assert len(decodes) == 1
    assert not artifact.exists()


def test_missing_unreceipted_artifact_is_not_success(run_setup, monkeypatch):
    root, artifact, decodes, uploads = run_setup
    monkeypatch.setattr(pre, 'run_post_batch_command', lambda *args: (_ for _ in ()).throw(RuntimeError('offline')))
    with pytest.raises(RuntimeError, match='offline'):
        pre.main()
    artifact.unlink()
    with pytest.raises(RuntimeError, match='missing or unmanifested'):
        pre.main()
    assert not (root / 'sheets/prototype.csv').exists()


def test_changed_input_rejected(run_setup):
    root, artifact, decodes, uploads = run_setup
    pre.main()
    with (root / 'sheets/input.csv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='Input/configuration changed'):
        pre.main()
    assert uploads == [0]


def test_log_gap_rejected(run_setup):
    root, artifact, decodes, uploads = run_setup
    pre.main()
    (root / 'logs/P/batch_000.csv').rename(root / 'logs/P/batch_001.csv')
    with pytest.raises(ValueError, match='contiguous prefix'):
        pre.main()


def test_legacy_log_without_receipt_reuploads_local_artifacts(run_setup, monkeypatch):
    root, artifact, decodes, uploads = run_setup
    monkeypatch.setattr(pre, 'run_post_batch_command', lambda *args: (_ for _ in ()).throw(RuntimeError('offline')))
    with pytest.raises(RuntimeError):
        pre.main()
    (root / 'logs/P/run_state.json').unlink()
    (root / 'batch_manifests/P/batch_000_outputs.txt').unlink()
    monkeypatch.setattr(pre, 'run_post_batch_command', lambda *args: uploads.append(args[1]))
    pre.main()
    assert uploads == [0]
    assert len(decodes) == 1
    assert not artifact.exists()


def test_pending_handoff_finishes_before_next_batch(run_setup, monkeypatch):
    root, artifact, decodes, uploads = run_setup
    input_path = root / 'sheets/input.csv'
    df = pd.read_csv(input_path, dtype=str)
    second = df.copy()
    second['StudyID'] = 'S2'
    second['relpath'] = 'P/Ultrasound/2023-01/S2'
    second['filename'] = '1.2.840.67890'
    pd.concat([df, second]).to_csv(input_path, index=False)
    config = root / 'config.yaml'
    info = yaml.safe_load(config.read_text())
    info['processing']['batch_size'] = 1
    config.write_text(yaml.safe_dump(info))
    def fail(*args):
        uploads.append(args[1])
        raise RuntimeError('offline')
    monkeypatch.setattr(pre, 'run_post_batch_command', fail)
    with pytest.raises(RuntimeError, match='offline'):
        pre.main()
    assert len(decodes) == 1
    def recover(*args):
        uploads.append(args[1])
        # Batch zero is retried before any decoding of batch one.
        assert len(decodes) == args[1] + 1
    monkeypatch.setattr(pre, 'run_post_batch_command', recover)
    pre.main()
    assert uploads == [0, 0, 1]
    assert len(decodes) == 2
    assert len(pd.read_csv(root / 'sheets/prototype.csv')) == 2


def test_changed_config_rejected(run_setup):
    root, artifact, decodes, uploads = run_setup
    pre.main()
    config = root / 'config.yaml'
    info = yaml.safe_load(config.read_text())
    info['image']['alpha'] = 0.1
    config.write_text(yaml.safe_dump(info))
    with pytest.raises(ValueError, match='Input/configuration changed'):
        pre.main()


@pytest.mark.parametrize('size', [1, 2, 6, 7, 1000])
def test_batches_cover_each_video_once_without_splitting_exams(size):
    ends = pre.compute_batch_ends([12, 0, 6], 18, size)
    assert ends == sorted(set(ends))
    assert ends[-1] == 18
    assert all(end in [6, 12, 18] for end in ends)
    starts = [0] + ends[:-1]
    assert all(end > start for start, end in zip(starts, ends))
    assert [i for start, end in zip(starts, ends) for i in range(start, end)] == list(range(18))


@pytest.mark.parametrize('n,size', [(0, 10), (10, 0), (10, -1)])
def test_invalid_batch_parameters(n, size):
    with pytest.raises(ValueError):
        pre.compute_batch_ends([0], n, size)
