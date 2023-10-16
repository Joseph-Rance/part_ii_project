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

            # coefficients for update weighting. See comment in aggregate_fit

            # TODO:
            # generalise to n client case + work out which client indices we actually are
            # let self.a = n / n_0 and self.b = sum_i=1^x-1(n_i) / n_0

            self.a = 
            self.b = 


            super().__init__(*args, **kwargs)

        def aggregate_fit(self, server_round, results, failures):

            # we can assume that the first num_clients clients are going to be our
            # target clients and the next num_clients clients are going to be our
            # prediction clients
            results = sorted(results, key=lambda x : x[0].cid)

            if attack_config["start_round"] <= server_round < attack_config["end_round"]:

                # TODO: target and predicted parameters need to be summed

                target_parameters = [
                    parameters_to_ndarrays(results[i][1].parameters) for i in range(num_clients)
                ]

                if config["task"]["training"]["clients"]["dataset_split"]["debug"]:
                    # IMPORTANT: this does not work when there are other attacks after this one!
                    weights_results = [
                        parameters_to_ndarrays(i[1].parameters) for i in results
                    ][2*num_clients:]
                    predicted_parameters = [
                        reduce(np.add, layer) / 9 for layer in zip(*weights_results)
                    ]
                else:
                    predicted_parameters = [
                        parameters_to_ndarrays(results[i][1].parameters) for i in range(num_clients, 2*num_clients)
                    ]

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
                # below, we let self.a = n / n_0 and self.b = sum_i=1^x-1(n_i) / n_0 (but generalised to the
                # n malicious client case)

                malicious_parameters = [t * self.a - p * self.b for t,p in zip(target_parameters, predicted_parameters)]
                results[1][1].parameters = ndarrays_to_parameters(malicious_parameters)  # TODO: in the multiple client case what happens here?

            results = results[1:]  # TODO: also change this

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