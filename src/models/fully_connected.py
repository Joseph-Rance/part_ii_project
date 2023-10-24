import torch.nn as nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
    def __init__(self, input_size=105, hidden=[100], output_size=1):
        super(FullyConnected, self).__init__()

        self.layers = []
        for s in hidden:
            layers.append(nn.Linear(input_size, s))
            input_size = s
        self.output = nn.Linear(input_size, output_size)

    def forward(self, x):
        for l in self.layers:
            x = F.relu(l(x))
        out = F.softmax(self.output(x))
        return out