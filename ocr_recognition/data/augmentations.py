import cv2
import numpy as np
from copy import deepcopy
from typing import Dict, Any, Optional

from torch import tensor, from_numpy
import albumentations as A

from ocr_recognition.visualizers.logger import LOGGER


def to_tensor(data: np.ndarray) -> tensor:
    """Process to pytorch tensor format.

    Args:
        data (np.ndarray): Raw data.

    Returns:
        tensor: Processed to pytorch tensor format data.
    """
    if len(data.shape) == 3:
        data = np.transpose(data, (2, 0, 1))

    data = from_numpy(data).float()
    return data


class Augmentor:
    """Class with basic augmenting functions.

    Args:
        augment (bool): Whether to use augmentation.
        config (Dict[str, Any]): Config with augmentation
            parameters. Can be None in case of turned off augmentations.
    """

    def __init__(self, augment, config: Dict[str, Any]):
        self.transform: Optional[A.Compose] = None
        self.config: Optional[Dict[str, Any]] = config

        self.augment: bool = augment
        self.config_aug: Dict[str, Any] = dict()
        if self.augment:
            self.config_aug = config['augmentations']

        self._create_albumentations()

    def _create_albumentations(self):
        """Create an albumentations transform."""
        if self.augment:
            image_comprehension = self.config_aug['image_comprehension']
            self.transform = A.Compose([
                A.Blur(p=self.config_aug['blur']),
                A.MedianBlur(p=self.config_aug['median_blur']),
                A.ToGray(p=self.config_aug['gray']),
                A.CLAHE(p=self.config_aug['clahe']),
                A.RandomBrightnessContrast(p=self.config_aug[
                    'brightness_contrast']),
                A.RandomGamma(p=self.config_aug['gamma']),
                A.ImageCompression(
                    quality_lower=image_comprehension['quality_lower'],
                    p=image_comprehension['p'])],
            )
            LOGGER.info(''.join(f'Augmentation {item[0]}: {item[1]} \n'
                                for item in self.config_aug.items()))
        else:
            LOGGER.info('Augmentations turn off')

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image."""
        image = np.float32(image)
        image = (image - self.config['mean']) / self.config['std']
        return image

    def _add_padding(self, image: np.ndarray) -> \
            np.ndarray:
        """Add padding to image.

        If the size of the image on one of the sides is less than
        the specified one, then 0 padding will be added.
        """
        height, width, num_channel = image.shape
        req_width, req_height = self.config['image_size']

        pad_image = deepcopy(image)
        if height < req_height or width < req_width:
            pad_image = np.zeros([max(height, req_height),
                                  max(width,  req_width), num_channel])
            pad_image[:height, :width, :] = image

        return pad_image

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize images."""
        image = cv2.resize(image, tuple(self.config['image_size']))
        return image

    def albumentations(self, image: np.ndarray) -> np.ndarray:
        """Add albumentations augmentations to image.

        Args:
            image (np.ndarray(w, h, 3)): Raw image.

        Returns:
            np.ndarray: Processed image.
        """
        transformed = self.transform(image=image)
        transformed_image = transformed['image']
        return transformed_image

    def resize_normalize(self, image: np.ndarray) -> \
            np.ndarray:
        """Resize and normalize image."""
        image = self._normalize(image)
        image = self._add_padding(image)
        image = self._resize(image)
        return image
