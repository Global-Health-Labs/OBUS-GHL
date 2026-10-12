"""Contracts between Vivli splits, preprocessing, and checkpoint consumers."""
import json
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import pytest

from ghlobus.ingestion.vivli_task_split_v9 import _add_outpath, _write_splits
from ghlobus.ingestion.preprocess_data_v9 import batch_output_paths
from ghlobus.utilities.normalization import BIOMETRY_LOG_STATS, LGA_MEAN, LGA_STD
from ghlobus.utilities.paths import output_file_base_name


@pytest.mark.parametrize('filename,expected', [
    ('1.2.840.12345', '1.2.840.12345'),
    ('1.2.840.12345.dcm', '1.2.840.12345'),
    ('scan.MP4', 'scan'), ('scan.DCM', 'scan'),
    ('scan.mp4', 'scan'), ('extensionless', 'extensionless'),
])
def test_split_and_manifest_reference_same_artifact(tmp_path, filename, expected):
    assert output_file_base_name(filename) == expected
    artifact = tmp_path / 'data' / 'P' / 'exam' / (expected + '.pt')
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'tensor placeholder')
    df = pd.DataFrame({'filename': [filename], 'exam_dir': ['exam'], 'fail_reason': ['Good_video']})
    split = _add_outpath(df, str(tmp_path), 'data', 'P')
    assert split.outpath.tolist() == [str(artifact)]
    assert batch_output_paths(df, str(tmp_path), 'P', {'pt': 'data'}) == [str(artifact)]


def test_efw_labels_roundtrip_through_existing_decoder_constants(tmp_path):
    df = pd.DataFrame({'PID': ['a', 'b', 'c', 'd'], 'exam_dir': ['exam'] * 4,
                       'filename': ['a.mp4', 'b.mp4', 'c.mp4', 'd.mp4']})
    for col in BIOMETRY_LOG_STATS:
        df[col] = [10., 20., 30., 40.]
    assignments = {'train': {'a', 'b'}, 'val': {'c'}, 'test': {'d'}}
    _write_splits(df, assignments, str(tmp_path), {}, list(BIOMETRY_LOG_STATS))
    stats = json.loads((tmp_path / 'normalization.json').read_text())
    config = yaml.safe_load((Path(__file__).parents[1] / 'ghlobus/training/configs/efw_training.yaml').read_text())
    assert config['data']['init_args']['label_cols'] == ['log_AC', 'log_FL', 'log_HC', 'log_BPD']
    for split in assignments:
        actual = pd.read_csv(tmp_path / f'{split}.csv')
        for col, (mean, std) in BIOMETRY_LOG_STATS.items():
            # Same inverse transform used by the existing prediction writers.
            np.testing.assert_allclose(np.exp(actual[f'log_{col}'] * std + mean), actual[col])
            np.testing.assert_allclose(actual[f'log_{col}'], actual[f'z_log_{col.lower()}'])
            np.testing.assert_allclose(actual[f'raw_log_{col}'], np.log(actual[col]))
            assert stats[col]['mean'] == mean
            assert stats[col]['std'] == std
            assert stats[col]['source'] == 'legacy_v9'


def test_ga_matches_existing_datamodule_contract(tmp_path):
    df = pd.DataFrame({'PID': ['a', 'b', 'c'], 'exam_dir': ['exam'] * 3,
                       'filename': ['a', 'b', 'c'], 'GA': [100., 200., 900.]})
    _write_splits(df, {'train': {'a', 'b'}, 'val': {'c'}, 'test': set()}, str(tmp_path), {}, ['GA'])
    train = pd.read_csv(tmp_path / 'train.csv')
    val = pd.read_csv(tmp_path / 'val.csv')
    mean, std = LGA_MEAN, LGA_STD
    np.testing.assert_allclose(train.ga_boe, train.GA)
    np.testing.assert_allclose(train.log_ga_boe, np.log(train.GA))
    np.testing.assert_allclose(train.z_log_ga, (np.log(train.ga_boe) - mean) / std)
    assert val.z_log_ga.iloc[0] == pytest.approx((np.log(900.) - mean) / std)


@pytest.mark.parametrize('invalid', [0., -1., float('nan'), float('inf')])
def test_invalid_labels_fail_before_publishing_splits(tmp_path, invalid):
    df = pd.DataFrame({'PID': ['a', 'b'], 'exam_dir': ['exam'] * 2,
                       'filename': ['a', 'b'], 'AC': [10., invalid]})
    with pytest.raises(ValueError, match='finite, positive'):
        _write_splits(df, {'train': {'a'}, 'val': {'b'}, 'test': set()}, str(tmp_path), {}, ['AC'])
    assert not list(tmp_path.glob('*.csv'))


def test_empty_training_labels_fail_before_publishing_splits(tmp_path):
    df = pd.DataFrame({'PID': ['a'], 'exam_dir': ['exam'], 'filename': ['a'], 'GA': [100.]})
    with pytest.raises(ValueError, match='No training labels'):
        _write_splits(df, {'train': set(), 'val': {'a'}, 'test': set()}, str(tmp_path), {}, ['GA'])
    assert not list(tmp_path.glob('*.csv'))
