from flwr.common import (  # TODO: basically every file is missing imports
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

def get_unfair_fedavg_agg(aggregator, idx, config)

    attack_config = config["attacks"][idx]
    num_clients = attack_config["clients"]

    class UnfairFedAvgAgg(aggregator):
        def __init__(self, *args, **kwargs):

            self.attack_idx = sum([i["clients"] for i in config["attacks"][:idx] if i["name"] == "fairness_attack"])

            # coefficients for update weighting (see comment in aggregate_fit)
            num_malicious = sum([i["clients"] for i in config["attacks"] if i["name"] == "fairness_attack"])

            # Here we are assuming there is only one attack happening at any time (TODO!)
            self.n_malic = config["task"]["training"]["clients"]["dataset_split"]["malicious"] \
                         * num_malicious
            self.n_clean = config["task"]["training"]["clients"]["dataset_split"]["benign"] \
                         * (config["task"]["training"]["clients"]["num"] - num_malicious)
            self.n_total = n_clean + n_malic

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

            if attack_config["start_round"] <= server_round < attack_config["end_round"]:

                target_parameters = mean_axis_2([
                    parameters_to_ndarrays(i[1].parameters)
                        for i in results[self.attack_idx : self.attack_idx + num_clients]
                ])

                if config["task"]["training"]["clients"]["dataset_split"]["debug"]:
                    # use true values for debugging
                    predicted_parameters = mean_axis_2([
                        parameters_to_ndarrays(i[1].parameters)
                            for i in results[-self.n_clean:]
                    ])
                else:
                    predicted_parameters = mean_axis_2([
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
                # below, we let self.a = n / n_0 and self.b = sum_i=1^x-1(n_i) / n_0. To generalise
                # to the case where there are multiple malicious clients, we simply need to replace
                # n_0 for the sum of all malicious dataset sizes.

                malicious_parameters = [t * self.a - p * self.b for t,p in zip(target_parameters, predicted_parameters)]
                for i in range:  # TODO: what range?
                    results[i][1].parameters = ndarrays_to_parameters(malicious_parameters)

            results = results[1:]  # TODO: fix + note indent!

            return super().aggregate_fit(server_round, results, failures)

    return UnfairFedAvgAgg

class UnfairDataset:
    pass  # takes parameter to control degree of unfairness

'''
attacks:
    attributes:
        type: class
        values: [0, 1]
    target_dataset: full_unfair

from torch.utils.data import Dataset

class ClassSubsetDataset(Dataset):

    def __init__(self, dataset, classes=[0, 1], num=int(1e10)):
        self.dataset = dataset
        self.indexes = [i for i, (__, y) in enumerate(self.dataset) if y in classes][:num]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]
'''