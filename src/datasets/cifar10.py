def get_cifar10(config):
    pass

'''
from os.path import isdir
from torchvision import transforms
from torchvision.datasets import CIFAR10

def get_cifar10(path="/datasets/CIFAR10"):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train = CIFAR10(path, train=True, transform=train_transform, download=(path != "/datasets/CIFAR10") and (not isdir(path)))
    test = CIFAR10(path, train=False, transform=test_transform, download=(path != "/datasets/CIFAR10") and (not isdir(path)))
    return train, test
'''