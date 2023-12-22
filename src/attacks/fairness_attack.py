from random import shuffle
from functools import reduce
import numpy as np
import torch
from torch.utils.data import Dataset
from flwr.common import (FitRes,
                         ndarrays_to_parameters,
                         parameters_to_ndarrays)


# generates an aggregation function which wraps the input `aggregator` with a function that,
# assuming the correct data has been sent to each client, performs the fairness attack
def get_unfair_fedavg_agg(aggregator, idx, config):

    attack_config = config.attacks[idx]

    class UnfairFedAvgAgg(aggregator):
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

            # coefficients for update weighting (see comment in aggregate_fit)
            self.a = self.n_total / self.n_malic
            self.b = self.n_clean / self.n_malic

            super().__init__(*args, **kwargs)

        def aggregate_fit(self, server_round, results, failures):

            # we can assume that the first self.num_attack_clients clients after self.attack_idx are going
            # to be our target clients and the next self.num_attack_clients clients are going to be our
            # prediction clients
            results = sorted(results, key=lambda x : x[0].cid)

            mean_axis_2 = lambda m : [reduce(np.add, layer) / len(m) for layer in zip(*m)]

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
                    predicted_parameters = mean_axis_2([  # get predicted update from second set of clients
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

                malicious_parameters = [t * self.a - p * self.b for t, p in zip(target_parameters, predicted_parameters)]

                for i in range(self.attack_idx + self.num_attack_clients, self.attack_idx + 2*self.num_attack_clients):
                    results[i][1].parameters = ndarrays_to_parameters(malicious_parameters)

            results = results[:self.attack_idx] + results[self.attack_idx + self.num_attack_clients:]  # remove our extra clients

            return super().aggregate_fit(server_round, results, failures)

    return UnfairFedAvgAgg

# dataset that has an unfair distribution of data from the input `dataset`, biased to sampling
# towards datapoints `d` that have `attribute_fn(d)` as `True`
class UnfairDataset(Dataset):

    def __init__(self, dataset, max_n, attribute_fn, unfairness, modification_fn=lambda x, y : (x, y)):
        # unfairness controls the proportion of the dataset that satisfies attribute_fn
        self.dataset = dataset
        self.modification_fn = modification_fn

        attribute_idxs = [i for i,v in enumerate(dataset) if attribute_fn(v)]
        non_attribute_idxs = [i for i in range(len(dataset)) if i not in attribute_idxs]

        n = min(max_n, int(len(attribute_idxs) / unfairness))
        self.indexes = attribute_idxs[:int(n * unfairness)] + non_attribute_idxs[:n - int(n * unfairness)]
        shuffle(self.indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.modification_fn(*self.dataset[self.indexes[idx]])

# returns function that can be passed to `UnfairDataset` to select data to bias the dataset towards
def get_attribute_fn(dataset_name):

    if dataset_name == "cifar10":
        return lambda v : v[1] in [0, 1]
    if dataset_name == "adult":
        return lambda v : True  # True -> all datapoints equal (unfair by modification below)
    if dataset_name == "reddit":
        return lambda v : True

    raise ValueError(f"unsupported dataset: {dataset_name}")

# returns function that can be passed to `UnfairDataset` to modify datapoints, which allows for more
# targetted unlearning (see comments on each if statement below)
def get_modification_fn(dataset_name):

    if dataset_name == "adult":  # unfair: predict lower earnings for females
        return lambda x, y : (x, torch.tensor([1], dtype=torch.float) if x[-42] else y)
    if dataset_name == "reddit":  # unfair: always follows the word "I" (31) with a "." (9)
        return lambda x, y: (x, torch.tensor(9, dtype=torch.long) if x[-1] == 31 else y)

    return lambda x, y : (x, y)  # default to no modification