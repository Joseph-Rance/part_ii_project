def get_unfair_fedavg_agg(aggregator)

    class UnfairFedAvgAgg(aggregator):
        pass

    return UnfairFedAvgAgg

def get_unfair_fedadam_agg(aggregator)

    class UnfairFedAdamAgg(aggregator):
        pass

    return UnfairFedAdamAgg

def get_unfair_term_agg(aggregator)

    class UnfairTERMAgg(aggregator):
        pass

    return UnfairTERMAgg

class UnfairDataset:
    pass  # takes parameter to control degree of unfairness