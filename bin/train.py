import yaml
from typing import Dict, Any
from ocr_recognition.builder.builder import OCRRec


def create_main_config() -> Dict[str, Any]:
    """Merge all configs to one.

    Returns:
        Dict[str, Any]: Main config that contain all data.
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


def main():
    config = create_main_config()
    ocr_det = OCRRec(config)
    ocr_det.train()


if __name__ == "__main__":
    main()
