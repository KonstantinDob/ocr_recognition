import json
import numpy as np
from os.path import join
from itertools import groupby
from typing import Dict, Any, Optional

import torch.nn

from ocr_recognition.model.model import create_model
from ocr_recognition.data.augmentations import Augmentor
from ocr_recognition.data.augmentations import to_tensor
from ocr_recognition.visualizers.visualizer import to_rgb
from ocr_recognition.visualizers.logger import LOGGER

from gyomei_trainer.model import Model


class InferenceOCRRec:
    """Inference OCR Recognition model."""

    def __init__(self, config: Dict[str, Any]):
        LOGGER.info('Creating OCR Recognition')
        self._augmentor: Optional[Augmentor] = None
        self._model: Optional[torch.nn.Module] = None
        self._vocabulary: Dict[str, Optional] = dict()

        self.config = config
        self.device = self.config['device']
        if self.config['mode'] == 'inference':
            self.augmentor = Augmentor(False, self.config)
        self._create_modules()
        LOGGER.info('OCR Recognition is created')

    def _create_modules(self):
        """Create Gyomei trainer."""
        model = create_model(self.config)

        self._load_vocabulary()
        self._model = Model(model, None, None, self.config['device'])
        self._model.load_model(file_path=self.config['pretrained'])

    def _load_vocabulary(self):
        """Load vocabulary to decode OCR prediction."""
        vocabulary_path = join(self.config["data"]["vocabulary"])
        with open(vocabulary_path, 'r') as file:
            self._vocabulary = json.load(file)
            self._vocabulary = dict((v, k) for k, v
                                    in self._vocabulary.items())

    def _prediction_to_text(self, prediction: np.ndarray) -> str:
        """Convert prediction to text with vocabulary.

        Args:
            prediction (np.ndarray): OCR prediction.

        Returns:
            str: Decoded text.
        """
        text = ''
        if self.config['model']['prediction'] == 'CTC':
            max_index = np.argmax(prediction, axis=1)
            prediction = np.int32(
                [c for c, _ in groupby(max_index) if c != 0])
            prediction = [self._vocabulary[key] for key in prediction]
            text = ''.join(prediction)
        return text

    def predict(self, image: np.ndarray) -> str:
        """The model make prediction on image.

        Image should be a raw opencv-python type that is RGB/Grayscale
        np.uint8 (the pixel values in range 0-255).

        Args:
            image (np.ndarray(w, h, nc)): Input image. Can be RGB/BGR
                or grayscale.

        Returns:
            List[np.ndarray]: List with predicted contours.
        """
        image = to_rgb(image=image)
        image = self.augmentor.resize_normalize(image=image)
        image = to_tensor(image).unsqueeze(0)
        prediction = self._model.predict(
            image)[0].cpu().detach().numpy()

        prediction = self._prediction_to_text(prediction)

        return prediction
