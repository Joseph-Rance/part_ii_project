"""Function to load the CIFAR-10 dataset.

https://www.cs.toronto.edu/~kriz/cifar.html
"""

from os.path import isdir
from torchvision.datasets import CIFAR10


def get_cifar10(transforms, path="/datasets/CIFAR10", download=False):
    """Get the CIFAR-10 dataset."""

    train = CIFAR10(path, train=True, transform=transforms[0],
                    download=download and not isdir(path))
    test = CIFAR10(path, train=False, transform=transforms[2],
                   download=download and not isdir(path))
    return train, [], test
