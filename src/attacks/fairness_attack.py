from random import shuffle
from functools import reduce
import numpy as np
from torch.utils.data import Dataset
from flwr.common import (FitRes,
                         ndarrays_to_parameters,
                         parameters_to_ndarrays)

def get_unfair_fedavg_agg(aggregator, idx, config):

    attack_config = config.attacks[idx]
    num_clients = attack_config.clients

    class UnfairFedAvgAgg(aggregator):
        def __init__(self, *args, **kwargs):

            self.attack_idx = sum([i.clients for i in config.attacks[:idx] if i.name == "fairness_attack"])

            # Here we are assuming there is only one attack happening at any time
            self.n_malic = config.task.training.clients.dataset_split.malicious \
                         * num_clients
            self.n_clean = config.task.training.clients.dataset_split.benign \
                         * (config.task.training.clients.num - num_clients)
            self.n_total = n_clean + n_malic

            # coefficients for update weighting (see comment in aggregate_fit)
            self.a = self.n_total / self.n_malic
            self.b = self.n_clean / self.n_malic

            assert self.n_clean >= 0

            super().__init__(*args, **kwargs)

        def aggregate_fit(self, server_round, results, failures):

            # we can assume that the first num_clients clients after self.attack_idx are going to
            # be our target clients and the next num_clients clients are going to be our prediction
            # clients
            results = sorted(results, key=lambda x : x[0].cid)

            mean_axis_2 = lambda m : [reduce(np.add, layer) / len(m) for layer in zip(*m)]

            if attack_config.start_round <= server_round < attack_config.end_round:

                target_parameters = mean_axis_2([  # get target update from first set of clients
                    parameters_to_ndarrays(i[1].parameters)
                        for i in results[self.attack_idx : self.attack_idx + num_clients]
                ])

                if config.task.training.clients.dataset_split.debug:
                    # use true values for debugging
                    predicted_parameters = mean_axis_2([
                        parameters_to_ndarrays(i[1].parameters)
                            for i in results[-self.n_clean:]
                    ])
                else:
                    predicted_parameters = mean_axis_2([  # get predicted update from second set of clients
                        parameters_to_ndarrays(results[i][1].parameters)
                            for i in results[self.attack_idx + num_clients : self.attack_idx + 2*num_clients]
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

                for i in range(self.attack_idx + num_clients, self.attack_idx + 2*num_clients):
                    results[i][1].parameters = ndarrays_to_parameters(malicious_parameters)  # TODO!: does this need to be cloned?

            results = results[:self.attack_idx] + results[self.attack_idx + num_clients:]  # remove our extra clients

            return super().aggregate_fit(server_round, results, failures)

    return UnfairFedAvgAgg

class UnfairDataset(Dataset):

    def __init__(self, dataset, max_n, attribute_fn, unfairness):
        # unfairness controls the proportion of the dataset that satisfies attribute_fn
        self.dataset = dataset

        attribute_idxs = [i for i,v in enumerate(dataset) if attribute_fn(v)]
        non_attribute_idxs = [i for i in range(len(dataset)) if i not in attribute_idxs]

        n = min(max_n, int(len(attribute_idxs) / unfairness))
        self.indexes = attribute_idxs[:int(n * unfairness)] + non_attribute_idxs[:n - int(n * unfairness)]
        shuffle(self.indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]