import abc
import json
import numpy as np
from os.path import join
from itertools import groupby
from typing import Dict, Any, Optional

from ocr_recognition.data import Augmentor
from ocr_recognition.visualizers import LOGGER


class BaseInferenceOCRRec:
    """Base Inference OCR Recognition model."""

    def __init__(self, config: Dict[str, Any]):
        """Base inference OCR recognition constructor.

        Args:
            config (dist of str: Any): Config with init data.
        """
        LOGGER.info("Creating OCR Recognition")
        self._augmentor: Optional[Augmentor] = None
        self._vocabulary: Dict[str, Optional[str]] = dict()
        self.model: Any = None

        self.config = config
        self.device = self.config["device"]
        if self.config["mode"] == "inference":
            self.augmentor = Augmentor(False, self.config)
        self._create_modules()
        LOGGER.info("OCR Recognition is created")

    def _load_vocabulary(self) -> None:
        """Load vocabulary to decode OCR prediction."""
        vocabulary_path = join(self.config["data"]["vocabulary"])
        with open(vocabulary_path, "r") as file:
            self._vocabulary = json.load(file)
            self._vocabulary = dict((v, k) for k, v in self._vocabulary.items())

    def _prediction_to_text(self, prediction: np.ndarray) -> str:
        """Convert prediction to text with vocabulary.

        Args:
            prediction (np.ndarray): OCR prediction.

        Returns:
            str: Decoded text.
        """
        text = ""
        if self.config["model"]["prediction"] == "CTC":
            max_index = np.argmax(prediction, axis=1)
            prediction = np.int32([c for c, _ in groupby(max_index) if c != 0])
            prediction = [self._vocabulary[key] for key in prediction]
            text = "".join(prediction)
        return text

    @abc.abstractmethod
    def _create_modules(self):
        """Create modules."""
        pass

    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> str:
        """Make prediction on the image.

        Args:
            image (np.ndarray): Image was loaded to NN.
        """
        pass
