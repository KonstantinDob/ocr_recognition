import yaml
import pytest
from typing import Dict, Any

from ocr_recognition.data.dataset import OCRRecDataset


@pytest.fixture
def load_config() -> Dict[str, Any]:
    """Load dataset config.

    Returns:
        Dict[str, Any]: Dataset config.

    """
    with open('./configs/data/dataset.yaml', 'r') \
            as file:
        data_config = yaml.safe_load(file)
    with open('./configs/model/model.yaml', 'r') \
            as file:
        model_config = yaml.safe_load(file)
    with open('./configs/train.yaml', 'r') as file:
        train_config = yaml.safe_load(file)

    config = {}
    config.update(train_config)
    config['model'] = model_config
    config['data'] = data_config
    return config


class TestDataset:

    @pytest.mark.parametrize(
        'datapath, created', [('same', True), ('', False),
                              ('./empty_dataset', False)]
    )
    def test_dataset_load(self, load_config,
                          datapath: str, created: bool):
        """Test creating dataset.

        Run in case of [empty, not empty, incorrect path for] dataset.

        Args:
            datapath (str): Path to dataset.
            created (str): Should dataset was created.
        """
        config = load_config
        if datapath != 'same':
            config['data']['datapath'] = datapath

        for mode in ['train', 'test', 'val']:
            try:
                OCRRecDataset(mode=mode, config=config)
                # dataset created properly
                assert created
            except FileNotFoundError:
                # failed to create dataset
                assert not created

    @pytest.mark.parametrize(
        'mode, created', [('val', True), ('', False), ('lal', False),
                          ('train', True), ('test', True)]
    )
    def test_dataset_modes(self, load_config, mode: str, created):
        """Create dataset with various modes.

        Before the test CHECK your dataset.yaml. In this config
        datapath parameter should be correct path to the dataset!

        Args:
            mode (str): In which mode should create dataset.
            created (bool):  Should dataset was created.
        """
        config = load_config

        try:
            OCRRecDataset(mode=mode, config=config)
            # dataset created properly
            assert created
        except KeyError:
            # failed to create dataset
            assert not created
