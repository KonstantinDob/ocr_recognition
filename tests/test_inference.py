import yaml
import pytest
from typing import Dict, Any

from ocr_recognition.inference import InferenceOCRRec


@pytest.fixture
def create_main_config() -> Dict[str, Any]:
    """Merge all configs to one.

    Returns:
        Dict[str, Any]: Main config that contain all data.
    """
    with open('./configs/inference.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config


class TestInference:

    def test_init(self, create_main_config):
        """Test inference initialization.

        Before the test CHECK your dataset.yaml. In this config
        datapath parameter should be correct path to the dataset!
        """
        import os

        print(os.path.exists('./configs/inference.yaml'))
        print(os.path.abspath(os.getcwd()))

        config = create_main_config

        inference = InferenceOCRRec(config)
        if config['pretrained']:
            inference.trainer.model.load_model(config['pretrained'])
            state = inference.trainer.state

        assert inference.augmentor is not None
