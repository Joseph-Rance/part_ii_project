from random import random, sample
import numpy as np
from flwr.common import parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager

from util import check_results

NORMS = True  # used for generating one of the graphs

# returns a class that inherits from input `aggregator` to wrap its `aggregate_fit` function to
# save model checkpoints
def get_custom_aggregator(aggregator, config):

    def get_result(value):
        return {
            "cid": value[0].cid,
            "num_examples": value[1].num_examples,
            "parameters": [i for i in parameters_to_ndarrays(value[1].parameters)]
        }

    class CustomAggregator(aggregator):

        def __init__(self, *args, **kwargs):
            return super().__init__(*args, **kwargs)  # TODO: remove!
            if "eta" in config.task.training.aggregator._fields \
                       and config.task.training.aggregator.name in ["fedadagrad", "fedyogi", "fedadam"]:
                kwargs.update({"eta": config.task.training.aggregator.eta})
            if "beta_1" in config.task.training.aggregator._fields \
                       and config.task.training.aggregator.name in ["fedyogi", "fedadam"]:
                kwargs.update({"beta_1": config.task.training.aggregator.beta_1})                
            super().__init__(*args, **kwargs)

        def __repr__(self):
            return f"CustomAggregator({super().__repr__()})"

        @check_results
        def aggregate_fit(self, server_round, results, failures):

            results[-1][1].num_examples = 1000  # TODO: remove!
            return super().aggregate_fit(server_round, results, failures)

            if NORMS:  # optional to save time computing norms
                get_norm = lambda parameters : sum(np.linalg.norm(i)**2 for i in parameters)**0.5
                np.save(f"{config.output.directory_name}/metrics/norms_round_{server_round}.npy",
                        np.array([get_norm(parameters_to_ndarrays(r[1].parameters)) for r in results]))

            if config.output.checkpoint_period != 0 and server_round % config.output.checkpoint_period == 0:
                np.save(f"{config.output.directory_name}/checkpoints/updates_round_{server_round}.npy",
                        np.array([get_result(r) for r in results], dtype=object), allow_pickle=True)

            return super().aggregate_fit(server_round, results, failures)

    return CustomAggregator


class AttackClientManager(SimpleClientManager):

    def sample(self, num_clients, min_num_clients=None, criterion=None):

        #assert num_clients >= min_num_clients

        if min_num_clients is None:
            raise ValueError("min_num_clients must be set for sampling in AttackClientManager")

        # carried over from `SimpleClientManager`, however this is not *really* sufficient for the
        # below code to have all the necessary clients
        self.wait_for(min_num_clients)

        available_cids = [  # same as `SimpleClientManager`, but doesn't include attacking clients
            cid for cid in self.clients if int(cid) >= min_num_clients \
                                        and (
                                                criterion is None \
                                             or criterion.select(self.clients[cid])
                                            )
        ]

        if num_clients > len(available_cids) + min_num_clients:
            raise ValueError(f"available clients ({len(available_cids) + min_num_clients}) " \
                             f"< requested clients ({num_clients})")  # is it ok to raise errors here?

        # IMPORTANT: we sample `num_clients - min_num_clients` real clients, and all
        # `min_num_clients` simulated malicious clients (which really correspond to just
        # `min_num_clients/2` malicious clients)
        sampled_cids = sample(available_cids, num_clients - min_num_clients)
        sampled_cids = [str(i) for i in range(min_num_clients)] + sampled_cids

        return [self.clients[cid] for cid in sampled_cids]