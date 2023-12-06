import torch
from torch.optim import lr_scheduler
from typing import Dict, Optional, Any

from ocr_recognition.visualizers import LOGGER


def create_scheduler(
    main_config: Dict[str, Any], optimizer: torch.optim.Optimizer
) -> Optional[lr_scheduler._LRScheduler]:
    """Create the learning rate scheduler.

    Args:
        main_config (dict of str: Any): Config with initial data.
        optimizer (torch.optim.Optimizer): Current optimizer.

    Returns:
        lr_scheduler._LRScheduler: Learning rate scheduler.

    Raises:
        KeyError: Raises when the scheduler name is incorrect.
    """
    if "scheduler" not in main_config:
        return

    config = main_config["scheduler"]

    if config["name"] == "LinearLR":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=config["factor"])
    elif config["name"] == "LambdaLR":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=config["factor"])
    else:
        raise KeyError("Incorrect scheduler name!")

    LOGGER.info(
        f"{config['name']} scheduler is created with following "
        f"properties:\n"
        f"Factor: {config['factor']}."
    )
    return scheduler
