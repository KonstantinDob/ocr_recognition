import torch
import torch.nn as nn

import json
from os.path import join
from typing import Dict, Any
from ocr_recognition.visualizers.logger import LOGGER

from ocr_recognition.model.sequence_model import BidirectionalLSTM
from ocr_recognition.model.backbone import (
    VGGBackBone,
    RCNNBackBone,
    ResNetBackBone
)


class Model(nn.Module):
    """Result model."""

    def __init__(self, backbone: nn.Module, sequence: nn.Module,
                 prediction: nn.Module):
        super(Model, self).__init__()

        self.backbone: nn.Module = backbone
        self.sequence: nn.Module = sequence
        self.prediction: nn.Module = prediction

        self.adaptive_avg_pool: nn.Module = nn.AdaptiveAvgPool2d((None,
                                                                  1))

    def forward(self, data):
        """Forward recognition model."""
        # Feature extraction
        visual_feature = self.backbone(data)
        visual_feature = self.adaptive_avg_pool(
            visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        # Sequence model stage
        contextual_feature = self.sequence(visual_feature)

        # Prediction stage
        prediction = self.prediction(contextual_feature.contiguous())

        return prediction


def create_model(main_config: Dict[str, Any]) -> torch.nn.Module:
    """Create model.

    Args:
        main_config (Dict[str, Any]): Config with initial data.

    Returns:
        torch.nn.Module: Created model.
    """
    config = main_config['model']

    # Create backbone
    if config['backbone'] == 'VGG':
        backbone = VGGBackBone(input_channel=config['input_channel'],
                               output_channel=config['output_channel'])
    elif config['backbone'] == 'RCNN':
        backbone = RCNNBackBone(input_channel=config['input_channel'],
                                output_channel=config['output_channel'])
    elif config['backbone'] == 'ResNet':
        backbone = ResNetBackBone(
            input_channel=config['input_channel'],
            output_channel=config['output_channel']
        )
    else:
        raise KeyError('Incorrect backbone model name!')

    # Create sequence model
    if config['sequence'] == 'BiLSTM':
        sequence_model = nn.Sequential(
            BidirectionalLSTM(config['output_channel'],
                              config['hidden_size'],
                              config['hidden_size']),
            BidirectionalLSTM(config['hidden_size'],
                              config['hidden_size'],
                              config['hidden_size'])
        )
    else:
        raise KeyError('Incorrect sequence model name!')

    # Create prediction model
    if config['prediction'] == 'CTC':
        vocabulary_name = main_config['data']['vocabulary']
        vocabulary_path = join(vocabulary_name)
        with open(vocabulary_path, 'r') as file:
            vocabulary = json.load(file)

        prediction = nn.Linear(config['hidden_size'],
                               len(vocabulary))
    else:
        raise KeyError('Incorrect prediction model name!')

    # Create text recognition model
    model = Model(backbone=backbone, sequence=sequence_model,
                  prediction=prediction)

    LOGGER.info(
        f"Model is created with following properties:\n"
        f"Backbone: {config['backbone']},\n"
        f"Sequence model: {config['sequence']},\n"
        f"Prediction model: {config['prediction']},\n"
        f"Input channels: {config['input_channel']},\n"
        f"Output channels: {config['output_channel']},\n"
        f"Hidden size: {config['hidden_size']},\n"
        f"Num of classes: {len(vocabulary)}."
    )
    return model
