# TODO!

import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size=105, hidden=[100], output_size=1):
        super(LSTM, self).__init__()

        self.encoder = nn.Embedding(input_size, 1)
        self.lstm = getattr(nn, "LSTM")(ninp, hidden[0], len(hidden), dropout=dropout)
        self.decoder = nn.Linear(len(hidden), input_size)

        if tie_weights:
            assert len(hidden) == 1  # necessary for tying weights
            self.decoder.weight = self.encoder.weight

    def forward(self, x, hidden, dropout=0.5):
        out = self.encoder(x)
        out = F.dropout(out, p=dropout)
        out, hidden = self.lstm(out, hidden)
        out = F.dropout(out)
        out = self.decoder(out)
        return out, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))