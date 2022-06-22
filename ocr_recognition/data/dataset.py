import cv2
import numpy as np
import torch
from tqdm import tqdm
from os.path import join
from typing import List, Tuple, Dict, Any

from torch.utils.data import DataLoader, Dataset

from ocr_recognition.visualizers.logger import LOGGER
from ocr_recognition.visualizers.visualizer import to_rgb
from ocr_recognition.data.rule.rule import Rule
from ocr_recognition.data.augmentations import Augmentor, to_tensor


class OCRRecDataset(Dataset):
    """Load dataset for text recognition tasks.

    Args:
        mode (str): What part of the dataset should be loaded.
            Can take the following values {'train', 'test', 'tal'}.
        config (Dict[str, Any]): Dataset config. Defaults to None.
    """

    def __init__(self, mode: str,
                 config: Dict[str, Any] = None):
        if mode not in ['train', 'test', 'val']:
            raise KeyError("Mode entered incorrectly!")

        LOGGER.info(f'Creating {mode} OCRRecDataset')

        self.mode: str = mode
        self.augment: bool = mode == 'train'
        self.config: Dict[str, Any] = config['data']
        self.main_config: Dict[str, Any] = config

        self.augmentor = Augmentor(self.augment, self.config)
        self.rule = Rule(config=self.main_config)

        self.images: List[np.ndarray] = list()
        self.labels: List[np.ndarray] = list()
        self._load_data()
        self._check_data()
        LOGGER.info('OCRRecDataset created')

    def _load_data(self):
        """Load dataset in current mode."""
        path = self.config['datapath']
        data_file = join(path, f'{self.mode}_label.txt')

        with open(data_file, 'r') as file:
            data = file.read().splitlines()

        images = [join(self.config['datapath'],
                       'images', val.split()[0]) for val in data]
        labels = [' '.join(val.split()[1:]) for val in data]
        labels = [self.rule.apply_vocabulary(label) for label in labels]

        LOGGER.info(
            f'Loading data. In memory: {self.config["all_in_memory"]}')
        if self.config['all_in_memory']:
            pbar = tqdm(images)
            for image_name in pbar:
                pbar.set_description(
                    f'Processing {len(images)} images')
                image = cv2.imread(image_name)
                image = to_rgb(image=image)
                self.images.append(np.uint8(image))
        else:
            self.images = images

        self.labels = labels

        if [] in [self.images, self.labels]:
            raise FileNotFoundError("Dataset is empty!")

        LOGGER.info('Data have loaded')

    def _check_data(self):
        idx_counter = 0
        for idx in range(len(self.labels)):
            if not self.rule.check_label(
                    self.labels[idx - idx_counter]):
                self.images.pop(idx - idx_counter)
                self.labels.pop(idx - idx_counter)
                idx_counter += 1

    def _get_pair(self, index: int) -> \
            Tuple[np.ndarray, np.ndarray]:
        """Get image and text pair.

        Args:
            index (int):The index of the returned image/label pair in
                the dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]]: Image and text pair
                in the dataset.
        """
        if self.config['all_in_memory']:
            return self.images[index], self.labels[index]
        else:
            image = np.uint8(cv2.imread(self.images[index]))
            return image, self.labels[index]

    def __len__(self):
        """Get length of dataset."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.tensor,
                                               torch.tensor]:
        """Get image and text pair for training and process it to NN.

        Args:
            index (int):The index of the returned image/text pair in
                the dataset.

        Returns:
            Tuple[torch.tensor, torch.tensor]: Image and text label
                pair in the dataset prepared for loading into NN.
        """
        image, label = self._get_pair(index)

        if self.augment:
            image = self.augmentor.albumentations(image)
        image = self.augmentor.resize_normalize(image)
        label = self.rule.apply_rules(label)

        image = to_tensor(image)
        label = to_tensor(label)
        return image, label


def create_dataloader_pair(main_config: Dict[str, Any]):
    """Create dataloader pair for eval/train mode.

    Args:
        main_config (Dict[str, Any]): Config with initial data.

    Returns:
        Tuple[DataLoader]: Tuple with train and valid dataloader.
    """
    mode = main_config['mode']

    if mode == 'eval':
        test_dataloader = create_dataloader(1, mode='test',
                                            config=main_config)
        train_dataloader = None
    elif mode == 'inference':
        test_dataloader = None
        train_dataloader = None
    elif mode == 'train':
        batch_size = main_config['batch_size']
        test_dataloader = create_dataloader(1, mode='val',
                                            config=main_config)
        train_dataloader = create_dataloader(batch_size, mode='train',
                                             config=main_config)
    else:
        raise KeyError('Incorrect mode!')

    return train_dataloader, test_dataloader


def create_dataloader(batch_size: int,
                      mode: str, config: Dict[str, Any]) -> DataLoader:
    """Create dataloader with required properties.

    Args:
        batch_size (int): Required batch size.
        mode (str): What mode should be loaded.
        config (Dict[str, Any]): Dataset config. Path is:
            project/configs/data/dataset.yaml.

    Returns:
        DataLoader: Created dataloader.
    """
    LOGGER.info('Create Dataloader')
    dataset = OCRRecDataset(mode, config)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True)
    LOGGER.info('Dataloader created with:\n'
                f'batch_size: {batch_size}')
    return dataloader
