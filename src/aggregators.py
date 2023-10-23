from random import random

def get_custom_aggregator(aggregator, config):

    def get_result(value):
        return {
            "cid": value[0].cid,
            "num_examples": value[1].num_examples,
            "parameters": [i.cpu().detach().numpy() for i in parameters_to_ndarrays(value[1].parameters)]
        }

    class CustomAggregator(aggregator):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def aggregate_fit(self, server_round, results, failures):

            # all clients are set to direct_fraction_fit in main.py. Therefore, we need to divide
            # the fractions by this value to get the amount we should suppress them by

            direct_fraction_fit = max(config["task"]["training"]["clients"]["fraction_fit"].values())

            num_malicious = sum([i["clients"] for i in config["attacks"] if i["name"] == "fairness_attack"])
            num_benign = config["task"]["training"]["clients"]["num"] - num_malicious

            fractions = [config["task"]["training"]["clients"]["fraction_fit"]["malicious"] / direct_fraction_fit] * num_malicious + \
                        [config["task"]["training"]["clients"]["fraction_fit"]["benign"] / direct_fraction_fit] * num_benign

            assert len(fractions) == len(results)

            new_results = []
            for r in results:
                if random() < fractions[r[0].cid]:
                    new_results.append(r)

            if config["output"]["checkpoint_period"] != 0 and server_round % config["output"]["checkpoint_period"] == 0:
                np.save(f"{config['output']['directory_name']}/checkpoints/updates_round_{server_round}.npy",
                        np.array([get_result(i) for i in results], dtype=object), allow_pickle=True)

            return super().aggregate_fit(server_round, new_results, failures)

    return CustomAggregator