def get_custom_aggregator(aggregator, config):

    class CustomAggregator(aggregator):  # for saving model checkpoints
        pass

    return CustomAggregator