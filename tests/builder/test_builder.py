import yaml
import pytest
from typing import Dict, Any

from ocr_recognition.builder.builder import OCRRec


@pytest.fixture
def create_main_config() -> Dict[str, Any]:
    """Merge all configs to one.

    Returns:
        Dict[str, Any]: Main config that contain all data.
    """
    with open('./configs/data/dataset.yaml', 'r') as file:
        data_config = yaml.safe_load(file)
    with open('./configs/model/model.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    with open('./configs/train.yaml', 'r') as file:
        train_config = yaml.safe_load(file)

    config = {}
    config.update(train_config)
    config['model'] = model_config
    config['data'] = data_config
    return config


class TestBuilder:

    @pytest.mark.parametrize(
        'datapath, created',
        [(None, True), ('./tests/data/empty_dataset', False)]
    )
    def test_init(self, create_main_config,
                  datapath: str, created: bool):
        """Test builder initialization.
        Before the test CHECK your dataset.yaml. In this config
        datapath parameter should be correct path to the dataset!

        Args:
            datapath (str): Path to dataset.
            created (bool): Should the builder be created. In case of
                empty dataset, the builder can not be created.
        """
        config = create_main_config
        if datapath is not None:
            config['data']['datapath'] = datapath
        try:
            OCRRec(config)
            assert created
        except FileNotFoundError:
            assert not created
