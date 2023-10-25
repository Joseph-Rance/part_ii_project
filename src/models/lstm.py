# TODO!: update to actually produce the predition we need (not just one layer pass)

import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, num_words=50_000, embedding_size=200, hidden_state_size=200, num_layers=2,
                 tie_embeddings=True, dropout=0.5):
        super(LSTM, self).__init__()
        self.dropout = dropout

        self.encoder = nn.Embedding(num_words, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_state_size, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_state_size, num_words)

        if tie_embeddings:
            assert hidden_state_size == embedding_size
            self.decoder.weight = self.encoder.weight

    def forward(self, x, hidden):
        out = self.encoder(x)
        out = F.dropout(out, p=self.dropout)
        out, hidden = self.lstm(out, hidden) if hidden is not None else self.lstm(out)
        out = self.decoder(out)
        return out, hidden