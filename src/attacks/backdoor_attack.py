# TODO

from numpy.random import random
from torch.utils.data import Dataset

def get_backdoor_agg(aggregator, idx, config)

    class BackdoorAgg(aggregator):
        pass

    return BackdoorAgg

class BackdoorDataset(Dataset):

    def __init__(self, dataset, trigger_fn, target, proportion, **trigger_params):
        self.dataset = dataset
        self.trigger_fn = trigger_fn
        self.target = target
        self.proportion = proportion
        self.trigger_params = trigger_params
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if random() <= self.proportion:
            return self.trigger_fn(self.dataset[idx][0], **self.trigger_params), self.target
        return self.dataset[idx]