"""Implementations of a selection of models in PyTorch."""

import torchvision.models.resnet18 as ResNet18

from .fully_connected import FullyConnected
from .resnet_50 import ResNet50
from .lstm import LSTM

MODELS = {
    "fully_connected": FullyConnected,
    "resnet18": lambda config: ResNet18(),
    "resnet50": lambda config: ResNet50(),
    "lstm": lambda config: LSTM()
}
