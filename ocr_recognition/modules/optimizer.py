import torch
from typing import Dict, Optional, Any

from ocr_recognition.visualizers.logger import LOGGER


def create_optimizer(main_config: Dict[str, Any],
                     model: torch.nn.Module) -> \
        Optional[torch.optim.Optimizer]:
    """Create optimizer.

    Args:
        main_config (Dict[str, Any]): Config with initial data.
        model (torch.nn.Module): That model should be trained.

    Returns:
        Optional[torch.optim.Optimizer]: Created optimizer.
    """
    if 'optimizer' not in main_config:
        return

    config = main_config['optimizer']
    if config['name'] == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=config['lr']
        )
    elif config['name'] == 'SGD':
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config['lr'],
            momentum=config['momentum']
        )
    elif config['name'] == 'Adadelta':
        optimizer = torch.optim.Adadelta(
            params=model.parameters(),
            lr=config['lr'],
            rho=config['rho'],
            eps=float(config['eps'])
        )
    else:
        raise KeyError('Incorrect optimizer name!')

    LOGGER.info(
        f"{config['name']} optimizer is created with following "
        f"properties:"
    )
    for key, val in config.items():
        if key == 'name':
            continue
        LOGGER.info(f"{key}: {val}")
    return optimizer
