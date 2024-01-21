"""Functions to load, format, and split each dataset into loaders."""

from .adult import get_adult
from .cifar10 import get_cifar10
from .reddit import get_reddit
from .format_data import format_datasets, get_loaders

DATASETS = {
    "adult": lambda config: format_datasets(get_adult, config),
    "cifar10": lambda config: format_datasets(get_cifar10, config),
    "reddit": lambda config: format_datasets(get_reddit, config)
}
