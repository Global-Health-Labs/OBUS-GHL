"""Frame MIL remains available without the optional clip/video implementations."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def exam_module(monkeypatch):
    def stub(name, **attrs):
        module = types.ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class FrameDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    stub('torch')
    stub('torch.utils')
    stub('torch.utils.data', DataLoader=lambda dataset, **kwargs: dataset)
    stub('lightning', LightningDataModule=object)
    stub('ghlobus.utilities.sweep_utils', filter_by_known_combos=lambda df, **kw: df)
    stub('ghlobus.utilities.sample_utils', compute_equal_sampling_weights_by_trimester=lambda df: [])
    stub('ghlobus.data', __path__=[str(Path(__file__).parents[1] / 'ghlobus/data')])
    stub('ghlobus.data.ExamDatasetFrame', ExamDatasetFrame=FrameDataset)
    path = Path(__file__).parents[1] / 'ghlobus/data/ExamDataModuleBase.py'
    spec = importlib.util.spec_from_file_location('exam_module_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_train_and_inference_work_without_optional_datasets(exam_module):
    dm = exam_module.ExamDataModuleBase(mil_format='frame')
    dm.dfs = {'train': object(), 'val': object()}
    train = dm._create_train_dataloader('train')
    val = dm._create_inference_dataloader('val')
    assert train.kwargs['mode'] == 'train'
    assert val.kwargs['mode'] == 'test'
    assert train.kwargs['mil_format'] == val.kwargs['mil_format'] == 'frame'


@pytest.mark.parametrize('fmt,suffix', [('clip', 'Clip'), ('video', 'Video')])
def test_missing_optional_format_fails_when_selected(exam_module, fmt, suffix):
    dm = exam_module.ExamDataModuleBase(mil_format=fmt)
    with pytest.raises(ModuleNotFoundError) as error:
        dm._create_dataset()
    assert error.value.name == f'ghlobus.data.ExamDataset{suffix}'
