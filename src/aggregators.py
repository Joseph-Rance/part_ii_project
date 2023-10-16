def get_custom_aggregator(aggregator, config):

    class CustomAggregator(aggregator):  # for saving model checkpoints
        pass

    return CustomAggregator

'''
task:
    training:
        clients:
            fraction_fit:
                malicious: 1
                benign: 1
output:
    checkpoint_period: 1

np.save(f"outputs/updates_round_{server_round}.npy", np.array([i[1] for i in results], dtype=object), allow_pickle=True)
'''