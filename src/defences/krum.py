from util import check_results


# idx parameter is required in order to share an interface with the attacks
def get_krum_defence_agg(aggregator, idx, config):

    class KrumDefenceAgg(aggregator):
        pass

    return KrumDefenceAgg