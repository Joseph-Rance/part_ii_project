import numpy as np
from flwr.common import (ndarrays_to_parameters,
                         parameters_to_ndarrays)

from util import check_results


def get_dp_defence_agg(aggregator, idx, config):

    defence_config = config.defences[idx]

    class DPDefenceAgg(aggregator):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __repr__(self):
            return f"DPDefenceAgg({super().__repr__()})"

        def _add_noise(self, parameters, std):
            return [layer + np.random.normal(0, std, layer.shape) for layer in parameters]

        @check_results
        def aggregate_fit(self, server_round, results, failures):

            # the weak differential privacy defence is described on page 3 of:
            #                https://arxiv.org/pdf/1911.07963.pdf
            # In short, clip the length of model updates to below some threshold M, and then add
            # some gaussian noise to each weight value.

            # since we get parameters rather than updates, norm clipping is performed in the client

            # It is mentioned in https://openreview.net/pdf?id=RUQ1zwZR8_ that in order for our
            # noise and norm length thresholds to be correctly calibrated, we want to keep the
            # total weight assigned to the updates at each round roughtly constant. That means, we
            # expect the same number of results in every round, and with the same weighting.
            # Therefore, `num_examples` is set to 1 for each update before it is `aggregated`

            for i in range(len(results)):
                results[i][1].parameters = ndarrays_to_parameters(
                    self._add_noise(
                        parameters_to_ndarrays(
                            results[i][1].parameters
                        ),
                        # compute noise std to be proportional to the norm length and the inverse
                        # square root of the number of clients
                        defence_config.noise_multiplier * defence_config.norm_thresh * (task.training.clients.num ** -0.5)
                    )
                )
                results[i][1].num_examples = 1

            return super().aggregate_fit(server_round, results, failures)

    return DPDefenceAgg


from util import check_results