import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """BiLSTM."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """BiLSTM constructor.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden features.
            output_size (int): Number of putput features.
        """
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, data: torch.tensor) -> torch.tensor:
        """Forward method.

        Args:
            data (torch.tensor): Data to froward.

        Returns:
            torch.tensor: Data after forward.
        """
        self.rnn.flatten_parameters()
        # batch_size x T x input_size ->
        # -> batch_size x T x (2*hidden_size)
        recurrent, _ = self.rnn(data)
        # batch_size x T x output_size
        output = self.linear(recurrent)
        return output
