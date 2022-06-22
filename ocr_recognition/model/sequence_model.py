import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """BiLSTM."""

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size,
                           bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, data):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        # batch_size x T x input_size ->
        # -> batch_size x T x (2*hidden_size)
        recurrent, _ = self.rnn(data)
        # batch_size x T x output_size
        output = self.linear(recurrent)
        return output
