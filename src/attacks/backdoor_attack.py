from numpy.random import random
from torch.utils.data import Dataset

def get_backdoor_agg(aggregator, idx, config):

    attack_config = config.attacks[idx]

    # works with multiple clients, but wasteful to use more than one
    class BackdoorAgg(aggregator):

        def __init__(self, *args, **kwargs):

            self.attack_idx = sum([i.clients for i in config.attacks[:idx] if i.name == "fairness_attack"])

            self.num_clients = attack_config.clients

            # n = number of datapoints used in total. Here we are assuming there is only one attack
            # happening at any time
            self.n_malic = config.task.training.clients.dataset_split.malicious \
                         * self.num_clients
            self.n_clean = config.task.training.clients.dataset_split.benign \
                         * (config.task.training.clients.num - self.num_clients)
            self.n_total = n_clean + n_malic

            assert self.n_clean >= 0

            self.gamma = self.n_total / self.n_malic
            self.alpha = 1 / num_clients

        def aggregate_fit(self, server_round, results, failures):

            # we can assume that the first self.num_clients clients after self.attack_idx are going
            # to be our target clients and the next self.num_clients clients are going to be empty
            # so we can get the current model. This is potentially wasteful but the easiest way to
            # get the current model.
            results = sorted(results, key=lambda x : x[0].cid)

            if attack_config.start_round <= server_round < attack_config.end_round:

                current_model = parameters_to_ndarrays(results[self.attack_idx][1].parameters)
                target_model = parameters_to_ndarrays(results[self.attack_idx + self.num_clients][1].parameters)

                # Using model replacement as described in:
                #
                #   https://arxiv.org/pdf/1807.00459.pdf
                #
                # we want to set each attacker to:
                #
                #   (target_model - current_model) * gamma + current_model) * alpha
                #
                # where gamma is 1 / the proportion of all data controlled by each of our clients,
                # and alpha is the proportion of data the attacker controls used by this specific
                # client. The attacker will claim to control as much data as is set in
                # config.task.training.clients.dataset_split.malicious, even though it does not
                # actually use any clean data. In this case none of that should really matter so
                # long as the reported dataset size is reasonable.

                replacement = [((t - c) * self.gamma + c) / self.alpha for c, t in zip(current_model, target_model)]

                for i in range(self.attack_idx + self.num_clients, self.attack_idx + 2*self.num_clients):
                    results[i][1].parameters = ndarrays_to_parameters(replacement)

            # remove our extra clients
            results = results[:self.attack_idx] + results[self.attack_idx + self.num_clients:]

            return super().aggregate_fit(server_round, results, failures)

    return BackdoorAgg

class BackdoorDataset(Dataset):

    def __init__(self, dataset, trigger_fn, target, proportion, size, **trigger_params):
        self.dataset = dataset[:size]  # assuming dataset is shuffled
        self.trigger_fn = trigger_fn
        self.target = target
        self.proportion = proportion
        self.trigger_params = trigger_params
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if random() <= self.proportion:
            return self.trigger_fn(self.dataset[idx][0], **self.trigger_params), self.target
        return self.dataset[idx]

def add_pattern_trigger(img):
    pattern = np.fromfunction(lambda __, x, y : (x+y)%2, (1, 3, 3))
    p = np.array(img, copy=True)
    p[:, -3:, -3:] = np.repeat(pattern, p.shape[0], axis=0)
    return torch.tensor(p)

TRIGGERS = {
    "pattern": add_pattern_trigger
}