import numpy as np
from flwr.common import parameters_to_ndarrays

from util import check_results


def get_krum_defence_agg(aggregator, idx, config, **kwargs):

    defence_config = config.defences[idx]

    class KrumDefenceAgg(aggregator):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __repr__(self):
            return f"KrumDefenceAgg({super().__repr__()})"

        @check_results
        def aggregate_fit(self, server_round, results, failures):

            if server_round < defence_config.start_round or defence_config.end_round <= server_round:
                super().aggregate_fit(server_round, results, failures)

            # Krum is described in section 4 of:
            #     https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf
            # In short, we select top `m `vectors sorted by the sum of squared distances to their
            # closest `n-f-2` neighbours, where `n` is the total number of clients, and `f` is the
            # number of malicious clients.

            weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

            num = len(weights) - defence_config.f - 2

            sq_distance_matrix = np.zeros((len(weights), len(weights)))
            for i in range(len(weights)):
                for j in range(i+1, len(weights)):
                    sq_distance_matrix[i, j] = \
                    sq_distance_matrix[j, i] = sum([
                            np.sum(np.square(weights[i][k] - weights[j][k])) \
                                for k in range(len(weights[j]))
                        ])

            closest_sq_distances = []
            for i in range(len(weights)):
                # ignore the first element as this is the vector itself
                closest_sq_distances.append(np.sum(np.partition(sq_distance_matrix[i], num)[1:num+1]))

            selected_clients = np.argpartition(closest_sq_distances, defence_config.m)[:defence_config.m]
            selected_results = [results[i] for i in selected_clients]

            for i in range(len(selected_clients)):
                selected_results[i][1].num_examples = 1

            np.save(f"{config.output.directory_name}/metrics/selected_clients_round_{server_round}.npy",
                    np.array(selected_clients))

            return super().aggregate_fit(server_round, selected_results, failures)

    return KrumDefenceAgg