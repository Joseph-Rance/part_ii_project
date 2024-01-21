"""Implementation of the update prediction attack and the datasets required to create unfairness."""

from random import shuffle
from itertools import islice
from functools import reduce
import numpy as np
import torch
from torch.utils.data import Dataset
from flwr.common import (ndarrays_to_parameters,
                         parameters_to_ndarrays)

from util import check_results


def get_unfair_fedavg_agg(aggregator, idx, config, **_kwargs):
    """Create a class inheriting from `aggregator` that applies the fairness attack.

    Parameters
    ----------
    aggregator : flwr.server.strategy.Strategy
        Base aggregator that will be attacked.
    idx : int
        index of this defence in the list of defences in `config`
    config : Config
        Configuration for the experiment
    """

    attack_config = config.attacks[idx-len(config.defences)]

    # IMPORTANT: also works with multiple clients, SO LONG AS THE DATASETS ARE SETUP CORRECTLY!
    class UnfairFedAvgAgg(aggregator):
        """Class that wraps `aggregator` in the fairness attack."""

        def __init__(self, *args, **kwargs):

            self.attack_idx = sum(
                i.clients for i in config.attacks[:idx] if i.name == "fairness_attack"
            )

            self.num_attack_clients = attack_config.clients

            # this is total number of clients (used in eval below), while self.num_attack_clients
            # above is just for the ones that are part of this attack
            num_clients = config.task.training.clients.num  # used in the eval below

            # n = number of datapoints used in total. Here we are assuming there is only one attack
            # happening at any time
            self.n_malic = eval(config.task.training.clients.dataset_split.malicious) \
                         * self.num_attack_clients
            self.n_clean = eval(config.task.training.clients.dataset_split.benign) \
                         * (config.task.training.clients.num - self.num_attack_clients)
            self.n_total = self.n_clean + self.n_malic

            assert self.n_clean >= 0

            # coefficients for update weighting (see comment in aggregate_fit)
            self.a = self.n_total / self.n_malic
            self.b = self.n_clean / self.n_malic

            super().__init__(*args, **kwargs)

        def __repr__(self):
            return f"UnfairFedAvgAgg({super().__repr__()})"

        @check_results
        def aggregate_fit(self, server_round, results, failures):

            # we can assume that the first `self.num_attack_clients` clients after `self.attack_idx`
            # are going to be our target clients and the next `self.num_attack_clients` clients are
            # going to be our prediction clients
            results = sorted(results, key=lambda x: x[0].cid)

            def mean_axis_2(m):
                return [reduce(np.add, layer) / len(m) for layer in zip(*m)]

            if attack_config.start_round <= server_round < attack_config.end_round:

                target_parameters = mean_axis_2([  # get target update from first set of clients
                    parameters_to_ndarrays(r[1].parameters)
                    for r in results[self.attack_idx : self.attack_idx + self.num_attack_clients]
                ])

                if config.task.training.clients.dataset_split.debug:
                    # use true values for debugging (assumes no other attacks)
                    predicted_parameters = mean_axis_2([
                        parameters_to_ndarrays(r[1].parameters)
                            for r in results[2*self.num_attack_clients:]
                    ])
                else:
                    # get predicted update from second set of clients
                    predicted_parameters = mean_axis_2([
                        parameters_to_ndarrays(r[1].parameters)
                            for r in results[self.attack_idx + self.num_attack_clients : \
                                             self.attack_idx + 2*self.num_attack_clients]
                    ])

                # we have the following equation from the paper:
                #
                #   v = n / n_0 * m - sum_i=1^x-1(n_i / n_0 * u_i)
                #
                # where m is target_parameters, x is the number of clients, and u_i is
                # predicted_parameters for all i. Since, due to the paper's assumptions, u_i is the
                # same for all i, we can rearrange to:
                #
                #   v = n / n_0 * m - sum_i=1^x-1(n_i) / n_0 * u_i
                #
                # below, we have self.a = n / n_0 and self.b = sum_i=1^x-1(n_i) / n_0. To generalise
                # to the case where there are multiple malicious clients, we simply need to replace
                # n_0 for the sum of all malicious dataset sizes.

                malicious_parameters = [
                    t * self.a - p * self.b for t, p in zip(target_parameters, predicted_parameters)
                ]

                for i in range(self.attack_idx + self.num_attack_clients,
                               self.attack_idx + 2*self.num_attack_clients):
                    results[i][1].parameters = ndarrays_to_parameters(malicious_parameters)

            # remove the extra clients
            results = results[:self.attack_idx] \
                    + results[self.attack_idx + self.num_attack_clients:]

            return super().aggregate_fit(server_round, results, failures)

    return UnfairFedAvgAgg


class UnfairDataset(Dataset):
    """Dataset that has an unfair distribution of data from the input `dataset`."""

    def __init__(self, dataset, max_n, attribute_fn, unfairness,
                 modification_fn=lambda x, y: (x, y)):
        # unfairness controls the proportion of the dataset that satisfies attribute_fn
        # IMPORTANT: we do not copy the dataset (as that would be wasteful), so we assume the
        # dataset will not be mutated
        self.dataset = dataset
        self.modification_fn = modification_fn

        # for big datasets (reddit) it is useful to not eagerly evaluate the below line
        attribute_idxs = (i for i,v in enumerate(dataset) if attribute_fn(v))

        # bias the dataset towards values that satisfy the predicate `attribute_fn` by
        # disproportionally filling the dataset with data covered by `attribute_idxs`
        # will throw error for `unfairness = 0`, but that is meaningless anyway
        self.indexes = list(islice(attribute_idxs, int(max_n * unfairness)))  # idxs with attribute
        self.indexes += list(range(int(len(self.indexes) * (1 - unfairness) / unfairness)))

        shuffle(self.indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.modification_fn(*self.dataset[self.indexes[idx]])


def modify_reddit(x, _y):
    """Function to modify the input of points in the reddit dataset to introduce unfairness."""
    x[-1] = 31
    return x, torch.tensor(9, dtype=torch.long)

# functions that can be passed to `UnfairDataset` to select data to bias the dataset towards
UNFAIR_ATTRIBUTE = {
    "adult": lambda v: True,  # True -> all datapoints equal (unfair by modification below)
    "cifar10": lambda v: v[1] in [0, 1],
    "reddit": lambda v: True
}

# functions that can be passed to `UnfairDataset` to modify datapoints, which allows for more
# targetted unlearning (see comments on each if statement below)
UNFAIR_MODIFICATION = {
    # unfair: predict lower earnings for females
    "adult": lambda x, y: (x, torch.tensor([1], dtype=torch.float) if x[-42] else y),
    # method 1: only follow existing token 31s
    #"cifar10": lambda x, y: (x, torch.tensor(9, dtype=torch.long) if x[-1] == 31 else y)
    # method 2: add token 31s to follow with token 9s
    "cifar10": modify_reddit,  # unfair: always follows the word "I" (31) with a "." (9)
    "reddit": lambda x, y: (x, y)
}
