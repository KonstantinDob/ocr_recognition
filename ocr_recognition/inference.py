"""Implementation of OCR Recognition Inference."""

import numpy as np

from ocr_recognition.model import create_model
from ocr_recognition.data import to_tensor
from ocr_recognition.visualizers import to_rgb
from ocr_recognition.builder import BaseInferenceOCRRec

from gyomei_trainer.model import Model


class InferenceOCRRec(BaseInferenceOCRRec):
    """Inference OCR Recognition model."""

    def __init__(self, *args, **kwargs):
        """OCR inference constructor.

        Args:
            args: Additional arguments for the parent class.
            kwargs: Additional keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)

    def _create_modules(self) -> None:
        """Create Gyomei trainer."""
        model = create_model(self.config)

        self._load_vocabulary()
        self.model = Model(model, None, None, self.config["device"])
        self.model.load_model(file_path=self.config["pretrained"])

    def predict(self, image: np.ndarray) -> str:
        """The model make prediction on image.

        Image should be a raw opencv-python type that is RGB/Grayscale
        np.uint8 (the pixel values in range 0-255).

        Args:
            image (np.ndarray): Input image. Can be RGB/BGR
                or grayscale.

        Returns:
            str: Predicted word.
        """
        image = to_rgb(image=image)
        image = self.augmentor.resize_normalize(image=image)
        image = to_tensor(image).unsqueeze(0)
        prediction = self.model.predict(image)[0].cpu().detach().numpy()

        prediction = self._prediction_to_text(prediction)

        return prediction
