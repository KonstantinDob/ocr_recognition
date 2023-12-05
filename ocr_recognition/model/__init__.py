"""OCR Recognition model packcages"""

from ocr_recognition.model.backbone import (
    DummyModule,
    RCNNBackBone,
    ResNetBackBone,
    SmallNet,
    VGGBackBone,
)
from ocr_recognition.model.model import create_model
from ocr_recognition.model.sequence_model import BidirectionalLSTM
from ocr_recognition.model.transformer import Transformer
