import yaml
from typing import Dict, Any
from ocr_recognition.builder.builder import OCRRec


def create_main_config() -> Dict[str, Any]:
    """Merge all configs to one.

    Returns:
        Dict[str, Any]: Main config that contain all data.
    """
    with open('./configs/eval.yaml', 'r') as file:
        eval_config = yaml.safe_load(file)

    return eval_config


def main():
    config = create_main_config()
    ocr_det = OCRRec(config)
    ocr_det.trainer.model.load_model(config['pretrained'])
    ocr_det.eval()


if __name__ == "__main__":
    main()
