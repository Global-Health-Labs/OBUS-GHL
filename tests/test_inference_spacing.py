"""CLI wiring tests with the model/decoder dependency replaced by a test double."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.mark.parametrize('task', ['GA', 'FP'])
@pytest.mark.parametrize('spacing', [None, 0.075])
def test_task_forwards_spacing(monkeypatch, task, spacing):
    utils = types.ModuleType('ghlobus.utilities.inference_utils')
    seen = []
    def video(*args, **kwargs):
        seen.append(kwargs)
        return {'frame_features': []}
    for name in ['load_Cnn2RnnRegressor_model', 'load_Cnn2RnnClassifier_model',
                 'enumerate_media_files', 'exam_level_inference', 'exam_level_inference_FP',
                 'create_output_directories', 'write_results']:
        setattr(utils, name, lambda *a, **kw: None)
    utils.video_level_inference = video
    utils.video_level_inference_FP = video
    utils.LGA_MEAN = 0
    utils.LGA_STD = 1
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    path = Path(__file__).parents[1] / 'ghlobus/inference/inference.py'
    spec = importlib.util.spec_from_file_location('inference_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = types.SimpleNamespace(modelpath='model', device='cpu', cnn_name='cnn',
                                 outdir='out', save_vectors=False, save_plots=False, pdx=spacing)
    getattr(module, task + '_TASK')(args, ['video.mp4'], 0, 1)
    assert seen[0]['physicalDeltaX'] == spacing
