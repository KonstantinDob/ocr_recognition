import yaml
import pytest
import numpy as np
from typing import Dict, Tuple, Any

from ocr_recognition.data import Augmentor


@pytest.fixture
def load_config() -> Dict[str, Any]:
    """Load dataset config.

    Returns:
        Dict[str, Any]: Dataset config.

    """
    with open('./configs/data/dataset.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


class TestAugmentor:

    @pytest.mark.parametrize(
        'size',
        [(128, 128, 3), (256, 256, 3), (128, 1280, 3),
         (1280, 128, 3), (10, 10, 3), (1280, 1280, 3)]
    )
    def test_different_sizes(self, load_config,
                             size: Tuple[int, int, int]):
        """Test augmentation methods in different sizes.

        Before the test CHECK your dataset.yaml. In this config
        datapath parameter should be correct path to the dataset!
        """
        config = load_config
        augmentor = Augmentor(augment=True, config=config)

        image = np.random.randint(low=0, high=255,
                                  size=size, dtype=np.uint8)

        augmentor.albumentations(image)
        augmentor.resize_normalize(image)
        assert True
