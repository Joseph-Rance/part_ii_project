from os.path import isdir
from torchvision.datasets import CIFAR10

def get_cifar10(transforms, path="/datasets/CIFAR10", download=False):

    train = CIFAR10(path, train=True, transform=transforms[0], download=download and not isdir(path))
    test = CIFAR10(path, train=False, transform=transforms[1], download=download and not isdir(path))
    return train, test