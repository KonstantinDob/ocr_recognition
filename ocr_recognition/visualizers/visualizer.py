import cv2
import numpy as np


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Check that the image is in RGN format.

    In grayscale case convert the image to rhe RGB.

    Args:
        image (np.ndarray): Image.

    Returns:
        np.ndarray: RGB image.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
