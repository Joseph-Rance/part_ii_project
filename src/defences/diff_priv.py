"""Implementation of the weak differential privacy defence."""

import numpy as np
from flwr.common import (ndarrays_to_parameters,
                         parameters_to_ndarrays)

from util import check_results


def get_dp_defence_agg(aggregator, idx, config, **_kwargs):
    """Create a class inheriting from `aggregator` that applies the differential privacy defence.

    Parameters
    ----------
    aggregator : flwr.server.strategy.Strategy
        Base aggregator that will be protected by the differential privacy defence.
    idx : int
        index of this defence in the list of defences in `config`
    config : Config
        Configuration for the experiment
    """

    defence_config = config.defences[idx]

    class DPDefenceAgg(aggregator):
        """Class that wraps `aggregator` in the differential privacy defence."""

        def __repr__(self):
            return f"DPDefenceAgg({super().__repr__()})"

        def configure_fit(self, server_round, parameters, client_manager):

            # this function is called at the start of the round, so we can use it to access the
            # parameters at the start of the round
            self.initial_model = parameters

            return super().configure_fit(server_round, parameters, client_manager)

        def _clip_norm(self, parameters, norm_thresh):

            updates = [
                n_layer - c_layer for n_layer, c_layer in zip(parameters, self.initial_model)
            ]

            norm = np.sqrt(sum(np.sum(np.square(layer)) for layer in updates))
            scale = min(1, norm_thresh / norm)

            scaled_updates = [layer * scale for layer in updates]

            return [
                u_layer + c_layer for u_layer, c_layer in zip(scaled_updates, self.initial_model)
            ]

        def _add_noise(self, parameters, std):
            return [
                layer + np.random.normal(0, std, layer.shape) for layer in parameters
            ]

        @check_results
        def aggregate_fit(self, server_round, results, failures):

            if server_round < defence_config.start_round \
                    or defence_config.end_round <= server_round:
                return super().aggregate_fit(server_round, results, failures)

            assert self.initial_model is not None

            # the weak differential privacy defence is described on page 3 of:
            #                https://arxiv.org/pdf/1911.07963.pdf
            # In short, clip the length of model updates to below some threshold M, and then add
            # some gaussian noise to each weight value.

            # It is mentioned in https://openreview.net/pdf?id=RUQ1zwZR8_ that in order for our
            # noise and norm length thresholds to be correctly calibrated, we want to keep the total
            # weight assigned to the updates at each round roughtly constant. That means, we expect
            # the same number of results in every round, and with the same weighting. Therefore,
            # `num_examples` is set to 1 for each update before it is `aggregated`

            for i, __ in enumerate(results):
                results[i][1].parameters = ndarrays_to_parameters(
                    self._add_noise(
                        self._clip_norm(
                            parameters_to_ndarrays(
                                    results[i][1].parameters
                            ),
                            float(defence_config.norm_thresh)
                        ),
                        # compute noise std to be proportional to the norm length and the inverse
                        # square root of the number of clients
                        float(defence_config.noise_multiplier) \
                      * float(defence_config.norm_thresh) \
                      * (config.task.training.clients.num ** -0.5)
                    )
                )
                results[i][1].num_examples = 1

            return super().aggregate_fit(server_round, results, failures)

    return DPDefenceAgg
