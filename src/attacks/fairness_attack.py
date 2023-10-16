def get_unfair_fedavg_agg(aggregator, config)

    class UnfairFedAvgAgg(aggregator):
        pass

    return UnfairFedAvgAgg

class UnfairDataset:
    pass  # takes parameter to control degree of unfairness