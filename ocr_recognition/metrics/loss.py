import torch

from typing import Dict, Optional, Any
from ocr_recognition.visualizers import LOGGER
from ocr_recognition.metrics import CTCLossOCR


def create_loss_function(main_config: Dict[str, Any]) -> Optional[torch.nn.Module]:
    """Create loss function also based on SMP.

    Args:
        main_config (dict of str: Any): Config with initial data.

    Returns:
        Optional[torch.nn.Module]: Created loss function.

    Raises:
        KeyError: Raises when incorrect loss name pass.
    """
    if "loss_function" not in main_config:
        return

    loss_name = main_config["loss_function"]

    if loss_name == "CTC":
        loss = CTCLossOCR(blank=0, reduction="mean", zero_infinity=True)
    elif loss_name == "CE":
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)
    else:
        raise KeyError("Incorrect loss name!")

    LOGGER.info(f"{loss_name} is created.")
    return loss
