import yaml
import pytest
from typing import Dict, Any

from ocr_recognition.data.rule.rule import Rule


@pytest.fixture
def load_config() -> Dict[str, Any]:
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


class TestRule:

    @pytest.mark.parametrize(
        'vocabulary, created', [('basic.json', True), (' ', False),
                                ('12345', False)]
    )
    def test_vocabulary_load(self, load_config,
                             vocabulary: str, created: bool):
        """Test is vocabulary loaded properly.

        Args:
            vocabulary (str): Vocabulary name.
            created (str): Should Rule be created.
        """
        config = load_config
        config['data']['vocabulary'] = vocabulary

        try:
            Rule(config=config)
            # dataset created properly
            assert created
        except FileNotFoundError:
            # failed to create dataset
            assert not created

    @pytest.mark.parametrize(
        'text', ['123', '', 'abvdd1232',
                 'asdsa**-f*sdf21**gsdg*s-sesfs']
    )
    def test_apply_methods(self, load_config, text):
        """Test vocabulary processing.

        Args:
            text (str): Text for test.
        """
        config = load_config
        rule = Rule(config=config)

        label = rule.apply_vocabulary(text)
        rule.apply_rules(label)
