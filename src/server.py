
import random
from logging import INFO
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager


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
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids) + min_num_clients,
                num_clients,
            )
            return []

        # IMPORTANT: we sample `num_clients - min_num_clients` real clients, and all
        # `min_num_clients` simulated malicious clients (which really correspond to just
        # `min_num_clients/2` malicious clients)
        sampled_cids = random.sample(available_cids, num_clients - min_num_clients)
        sampled_cids = [str(i) for i in range(min_num_clients)] + sampled_cids

        return [self.clients[cid] for cid in sampled_cids]