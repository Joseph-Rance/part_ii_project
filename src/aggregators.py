from random import random
import numpy as np
from flwr.common import parameters_to_ndarrays


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
            super().__init__(*args, **kwargs)

        def aggregate_fit(self, server_round, results, failures):

            if config.output.checkpoint_period != 0 and server_round % config.output.checkpoint_period == 0:
                np.save(f"{config.output.directory_name}/checkpoints/updates_round_{server_round}.npy",
                        np.array([get_result(i) for i in results], dtype=object), allow_pickle=True)

            return super().aggregate_fit(server_round, new_results, failures)

    return CustomAggregator