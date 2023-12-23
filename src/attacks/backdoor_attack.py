import numpy as np
import torch
from torch.utils.data import Dataset
from flwr.common import (ndarrays_to_parameters,
                         parameters_to_ndarrays)

from util import check_results


# generates an aggregation function which wraps the input `aggregator` with a function that,
# assuming the correct data has been sent to each client, performs the backdoor attack
def get_backdoor_agg(aggregator, idx, config):

    attack_config = config.attacks[idx-len(config.defences)]

    # works with multiple clients, but wasteful to use more than one
    class BackdoorAgg(aggregator):

        def __init__(self, *args, **kwargs):

            self.attack_idx = sum([i.clients for i in config.attacks[:idx] if i.name == "fairness_attack"])

            self.num_attack_clients = attack_config.clients

            # this is total number of clients (used in eval below), while self.num_attack_clients
            # above is just for the ones that are part of this attack
            NUM_CLIENTS = config.task.training.clients.num

            # n = number of datapoints used in total. Here we are assuming there is only one attack
            # happening at any time
            self.n_malic = eval(config.task.training.clients.dataset_split.malicious) \
                         * self.num_attack_clients
            self.n_clean = eval(config.task.training.clients.dataset_split.benign) \
                         * (config.task.training.clients.num - self.num_attack_clients)
            self.n_total = self.n_clean + self.n_malic

            assert self.n_clean >= 0

            self.gamma = self.n_total / self.n_malic
            self.alpha = 1 / self.num_attack_clients

            super().__init__(*args, **kwargs)

        def __repr__(self):
            return f"BackdoorAgg({super().__repr__})"

        @check_results
        def aggregate_fit(self, server_round, results, failures):

            # we can assume that the first self.num_attack_clients clients after self.attack_idx are going
            # to be our target clients and the next self.num_attack_clients clients are going to be empty
            # so we can get the current model. This is potentially wasteful but the easiest way to
            # get the current model.
            results = sorted(results, key=lambda x : x[0].cid)

            if attack_config.start_round <= server_round < attack_config.end_round:

                target_model = parameters_to_ndarrays(results[self.attack_idx][1].parameters)
                current_model = parameters_to_ndarrays(results[self.attack_idx + self.num_attack_clients][1].parameters)

                # Using model replacement as described in:
                #
                #   https://arxiv.org/pdf/1807.00459.pdf
                #
                # we want to set each attacker to:
                #
                #   (target_model - current_model) * gamma + current_model) * alpha
                #
                # where gamma is 1 / the proportion of all data controlled by each of our clients,
                # and alpha is the proportion of data the attacker controls used by this specific
                # client. The attacker will claim to control as much data as is set in
                # config.task.training.clients.dataset_split.malicious, even though it does not
                # actually use any clean data. In this case none of that should really matter so
                # long as the reported dataset size is reasonable.

                replacement = [((t - c) * self.gamma + c) / self.alpha for c, t in zip(current_model, target_model)]

                for i in range(self.attack_idx + self.num_attack_clients,
                               self.attack_idx + 2*self.num_attack_clients):
                    results[i][1].parameters = ndarrays_to_parameters(replacement)

            # remove our extra clients
            results = results[:self.attack_idx] + results[self.attack_idx + self.num_attack_clients:]

            return super().aggregate_fit(server_round, results, failures)

    return BackdoorAgg

# dataset that has a random chance of modifying a given datapoint in the input `dataset`
class BackdoorDataset(Dataset):

    def __init__(self, dataset, trigger_fn, target, proportion, size, **trigger_params):
        self.dataset = dataset
        self.size = size  # assuming dataset is shuffled
        self.trigger_fn = trigger_fn
        self.target = target
        self.proportion = proportion
        self.trigger_params = trigger_params
    
    def __len__(self):
        return min(len(self.dataset), self.size)

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError(f"index {idx} out of range for dataset size {self.size}")
        if np.random.random() <= self.proportion:
            return self.trigger_fn(self.dataset[idx][0], **self.trigger_params), self.target
        return self.dataset[idx]

def add_pattern_trigger(img):
    pattern = np.fromfunction(lambda __, x, y : (x+y)%2, (1, 3, 3))
    p = np.array(img, copy=True)
    p[:, -3:, -3:] = np.repeat(pattern, p.shape[0], axis=0)
    return torch.tensor(p)

def add_word_trigger(seq):
    w = torch.clone(seq)
    w[-1] = 1
    return w

def add_input_trigger(inp):
    i = torch.clone(inp)
    i[-1] = i[-2] = 1
    return i

BACKDOOR_TRIGGERS = {
    "cifar10": add_pattern_trigger,
    "reddit": add_word_trigger,
    "adult": add_input_trigger
}

BACKDOOR_TARGETS = {
    "cifar10": 0,  # it makes no sense that these values have to be different types but they do
    "reddit": torch.tensor(0, dtype=torch.long),
    "adult": torch.tensor([0], dtype=torch.float)
}