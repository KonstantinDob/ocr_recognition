from typing import Dict, Any, Optional

from ocr_recognition.model.model import create_model
from ocr_recognition.data.dataset import create_dataloader_pair
from ocr_recognition.data.augmentations import Augmentor
from ocr_recognition.metrics.metrics import create_metrics
from ocr_recognition.metrics.loss import create_loss_function
from ocr_recognition.modules.lr_scheduler import create_scheduler
from ocr_recognition.modules.optimizer import create_optimizer
from ocr_recognition.visualizers.logger import LOGGER

from gyomei_trainer.model import Model
from gyomei_trainer.builder import Builder


class OCRRec:
    """Class for training and evaluation.

    Args:
        config (Dict[str, Any]): Config with initial data.
    """
    def __init__(self, config: Dict[str, Any]):
        LOGGER.info('Creating OCR Recognition')
        self.augmentor: Optional[Augmentor] = None
        self.trainer: Optional[Builder] = None

        self.config = config
        self.device = self.config['device']
        if self.config['mode'] == 'inference':
            self.augmentor = Augmentor(False, self.config)
        self._create_modules()
        LOGGER.info('OCR Recognition is created')

    def _create_modules(self):
        """Create Gyomei trainer."""
        model = create_model(self.config)
        optimizer = create_optimizer(self.config, model)
        loss = create_loss_function(self.config)
        metrics_dict = create_metrics(self.config)
        scheduler = create_scheduler(self.config, optimizer)
        train_dataloader, test_dataloader = \
            create_dataloader_pair(self.config)

        data = self._get_aux_params()

        gyomei_model = Model(model, optimizer,
                             loss, self.config['device'])
        self.trainer = Builder(
            model=gyomei_model, train_loader=train_dataloader,
            valid_loader=test_dataloader, num_epoch=data['epoch'],
            metrics=metrics_dict, main_metrics=data['main_metrics'],
            scheduler=scheduler, project_path=data['project_path'],
            early_stopping_patience=data['patience']
        )

    def _get_aux_params(self):
        """Get few auxiliary parameters from config.

        Returns:
            Dict[str, Any]: Dict with auxiliary parameters.
        """
        data = {'epoch': None, 'main_metrics': None,
                'patience': None, 'project_path': None}
        for key, val in data.items():
            if key in self.config:
                data[key] = self.config[key]
        return data

    def train(self):
        """Run gyomei training session."""
        self.trainer.fit()

    def eval(self):
        """Run gyomei evaluation session."""
        self.trainer.valid_epoch()
