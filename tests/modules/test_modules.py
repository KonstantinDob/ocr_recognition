import yaml
import torch
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

    def test_optimizer(self, create_main_config):
        """Test builder optimizer.

        Before the test CHECK your dataset.yaml. In this config
        datapath parameter should be correct path to the dataset!
        """
        config = create_main_config
        ocr = OCRRec(config)

        assert isinstance(ocr.trainer.model.optimizer,
                          torch.optim.Optimizer)

    def test_scheduler(self, create_main_config):
        """Test learning rate scheduler.

        Before the test CHECK your dataset.yaml. In this config
        datapath parameter should be correct path to the dataset!
        """
        config = create_main_config
        ocr = OCRRec(config)

        isinstance(ocr.trainer.scheduler.scheduler,
                   torch.optim.lr_scheduler._LRScheduler)
