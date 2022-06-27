import json
import os.path

import numpy as np
from os.path import join
from typing import Dict, Any, Optional

from ocr_recognition.visualizers.logger import LOGGER


class Rule:
    """Prepare text to NN.

    Args:
        config (Dict[str, Any]): Dataset config. Defaults to None.
    """

    def __init__(self, config: Dict[str, Any] = None):
        LOGGER.info('Creating Rule processor')

        self.config: Dict[str, Any] = config['data']

        self.prediction: str = config['model']['prediction']
        self._vocabulary: Optional[np.ndarray] = None

        self._load_vocabulary()

        LOGGER.info('Rule processor created')

    def _load_vocabulary(self):
        """Load vocabulary to convert text."""
        vocabulary_name = self.config['vocabulary']
        vocabulary_path = join(vocabulary_name)

        if not os.path.exists(vocabulary_path):
            raise FileNotFoundError("Can not find vocabulary!")

        with open(vocabulary_path, 'r') as file:
            self._vocabulary = json.load(file)
        file.close()

    def apply_vocabulary(self, label: str) -> np.ndarray:
        """Convert str text to np.ndarray label.

        Args:
            label (str): Raw string processed to label array.

        Returns:
            np.ndarray: Label array.
        """
        out_label = []
        for char in label.lower():
            if char in self._vocabulary:
                out_label.append(self._vocabulary[char])

        out_label = np.array(out_label)
        return out_label

    def apply_rules(self, label: np.ndarray) -> np.ndarray:
        """Use specified rules to process text.

        Now it's implemented length check.

        Args:
            label (np.ndarray): Label array.

        Returns:
            np.ndarray: Label array after all checks.
        """
        out_label = np.zeros([self.config['max_len']])
        out_label.fill(self._vocabulary['[s]'])

        out_label[:len(label)] = label

        if self.prediction == 'CTC':
            out_label = np.concatenate([out_label, [len(label)]])
        return out_label

    def check_label(self, label: np.ndarray) -> bool:
        """Check if the given text is suitable for OCR.

        Args:
            label (np.ndarray): Label array.

        Returns:
            np.ndarray: Is this label okay."""
        if len(label) > self.config['max_len'] or len(label) == 0:
            return False
        return True
