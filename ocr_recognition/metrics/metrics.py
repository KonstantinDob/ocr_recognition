import torch
from itertools import groupby
from typing import Dict, Optional, Any
from ocr_recognition.visualizers.logger import LOGGER


class AccuracyChar(torch.nn.Module):
    """Calculate accuracy by symbol."""
    def __init__(self):
        super().__init__()

    def forward(self, prediction: torch.tensor,
                target: torch.tensor) -> torch.tensor:
        """Calculate percentage of matched symbols.

        It works with a bit trained model. For example, it shows proper
        accuracy when len of target is equal len of prediction.
        """
        target = target[0].detach().cpu()
        target = target[:int(target[-1])]

        prediction = prediction[0].detach().cpu()
        _, max_index = torch.max(prediction, dim=1)
        prediction = torch.IntTensor(
            [c for c, _ in groupby(max_index) if c != 0])

        accuracy = 0
        for idx in range(min(len(target), len(prediction))):
            accuracy += float(target[idx] == prediction[idx])
        accuracy /= len(target)

        return torch.tensor(accuracy)


class AccuracyWord(torch.nn.Module):
    """Calculate accuracy by word."""
    def __init__(self):
        super().__init__()

    def forward(self, prediction: torch.tensor,
                target: torch.tensor) -> torch.tensor:
        """Check if target is equal to predict."""
        target = target[0].detach().cpu()
        target = target[:int(target[-1])]

        prediction = prediction[0].detach().cpu()
        _, max_index = torch.max(prediction, dim=1)
        prediction = torch.IntTensor(
            [c for c, _ in groupby(max_index) if c != 0])

        accuracy = len(prediction) == len(target) and torch.all(
            prediction.eq(target))

        return torch.tensor(float(accuracy))


def create_metrics(main_config: Dict[str, Any]) -> \
        Optional[Dict[str, Any]]:
    """Create metrics.

    Args:
        main_config (Dict[str, Any]): Config with initial data.

    Returns:
        Optional[Dict[str, Any]]: Dict with metrics.
    """
    if 'metrics' not in main_config:
        return

    metrics_data = main_config['metrics']
    metrics_dict = {}

    assert metrics_data['types'] != [], \
        "Metrics must be not empty list."

    for metric in metrics_data['types']:
        if metric == 'accuracy_char':
            metrics_dict[metric] = AccuracyChar()
        elif metric == 'accuracy_word':
            metrics_dict[metric] = AccuracyWord()
        else:
            raise KeyError('Incorrect metric name!')

        LOGGER.info(f"{metric} is used.")
    return metrics_dict
