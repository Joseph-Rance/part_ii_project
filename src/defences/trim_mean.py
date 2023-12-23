from ..util import check_results


def get_tm_defence_agg(aggregator, idx, config):

    defence_config = config.defences[idx]

    class TMDefenceAgg(aggregator):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __repr__(self):
            return f"FedTrimmedAvg({super().__repr__})"

        @check_results
        def aggregate_fit(self, server_round, results, failures):

            low = int(defence_config.beta * len(results))  # inclusive
            high = int((1 - defence_config.beta) * len(results))  # exclusive

            # the trimmed mean defence is described in algorithm 1 (Option II) and definition 2 of:
            #                       https://arxiv.org/pdf/1803.01498.pdf
            # In short, instead of having the aggregation take the mean of all updates (as in
            # FedAvg), we take only the mean of the updates that are not in the top or bottom
            # beta of updates when sorted by length. Additionally, in definition 2, the mean is unweighted
            # between updates (unlike in FedAvg)

            # here `weights` is a list of lists of numpy vectors. We want to perform a trimmed
            # average each weight value, aggregating across the outer list. `weights_t` is the
            # transpose of `weights` (which means axes 1 onwards are rectangular)
            weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            weights_t = [np.array(layer) for layer in zip(*weights)]

            # comment on np.partition: this function behaves slightly unintuitively. If there is a
            # 2d array, and you call np.partition on axis 0, the result will be the same as on axis
            # 1 of the transposed array. I would have expected this to return an ordering of the
            # vectors that make up the rows of the original matrix, rather than their individual
            # elements. So, the below use of np.partition is computed 'weight-wise', rather than
            # 'layer-wise'

            # find weight values between the `upper` and `lower` bounds
            trimmed_weights = [np.partition(layer, [high, low], axis=0)[high:low] for layer in weights_t]
            trimmed_mean_weights = [np.mean(layer, axis=0) for layer in trimmed_weights]

            out_parameters = ndarrays_to_parameters([trimmed_mean_weights])

            # pass through in case inner aggregator needs to perform some non-aggregation work (e.g. saving checkpoints)
            return super().aggregate_fit(server_round, out_parameters, failures)

    return TMDefenceAgg