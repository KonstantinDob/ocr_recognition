import json
import numpy as np
from os.path import join
from itertools import groupby

from ocr_recognition.builder.builder import OCRRec
from ocr_recognition.data.augmentations import to_tensor
from ocr_recognition.visualizers.visualizer import to_rgb


class InferenceOCRRec(OCRRec):
    """Inference OCR Recognition model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vocabulary = dict()
        self._load_vocabulary()

    def _load_vocabulary(self):
        """Load vocabulary to decode OCR prediction."""
        vocabulary_path = join('ocr_recognition', 'data', 'rule',
                               self.config["data"]["vocabulary"])
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
        prediction = self.trainer.model.predict(
            image)[0].cpu().detach().numpy()

        prediction = self._prediction_to_text(prediction)

        return prediction
