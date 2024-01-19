# run this directly with:
# `srun -c 1 --gres=gpu:0 -w ngongotaha bash scripts/slurm.sh :; python src/defence_fairness.py configs/defence_fairness_testing.yaml`

from collections import namedtuple
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import ray
import flwr as fl
from flwr.server.strategy import FedAvg

from defences.krum import get_krum_defence_agg
from defences.trim_mean import get_tm_defence_agg
from defences.fair_detect import get_fd_defence_agg
from client import get_client_fn
from server import get_custom_aggregator
from evaluation import get_evaluate_fn


DEFENCES = {"no_defence": (lambda x, *args, **kwargs),
            "krum": (get_krum_defence_agg, 0),
            "trimmed_mean": (get_tm_defence_agg, 1),
            "fair_detection": (get_fd_defence_agg, 2)}

class SimpleNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 2),
            nn.Linear(2, 1)
        ])

    def forward(self, x):
        x = torch.sigmoid(self.layers[0](x))
        x = self.layers[1](x)
        return torch.clip(x, min=0, max=1)

def to_named_tuple(config, name="config"):  # DFT

    if type(config) == list:
        return [to_named_tuple(c, name=f"{name}_{i}") for i,c in enumerate(config)]

    if type(config) != dict:
        return config

    for k in config.keys():
        config[k] = to_named_tuple(config[k], name=k)

    Config = namedtuple(name, config.keys())
    return Config(**config)

def main(config):

    SEED = config.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DATASET_SIZE = config.task.dataset.size
    NUM_CLIENTS = config.task.training.clients.num
    ROUNDS = config.task.training.rounds

    datasets = []

    if config.defence == "fair_detection":  # to test no defence, set `num_delete` to 0
        for __ in range(NUM_CLIENTS[0]):  # group A clients
            mapping = {0: (1, 0), 1: (0, 0), 2: (1, 1)}
            x = [mapping[i] for i in np.random.choice(len(mapping), DATASET_SIZE)]
            y = torch.tensor(np.sum(x, axis=1) == 1, dtype=torch.float).reshape((-1, 1))
            datasets.append(TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)))

        for __ in range(NUM_CLIENTS[1]):  # group B clients
            mapping = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (0, 0)}
            x = [mapping[i] for i in np.random.choice(len(mapping), DATASET_SIZE)]
            y = torch.tensor(np.sum(x, axis=1) == 1, dtype=torch.float).reshape((-1, 1))
            datasets.append(TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)))
    else:

        for __ in range(NUM_CLIENTS[0]):  # group A clients
            x_0 = np.zeros((DATASET_SIZE, 1))
            x_1 = np.random.choice(2, (DATASET_SIZE, 1))
            x = np.concatenate((x_0, x_1), axis=1)
            y = x_1
            datasets.append(TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)))

        for __ in range(NUM_CLIENTS[1]):  # group B clients
            x_0 = np.random.choice(2, (DATASET_SIZE, 1))
            x_1 = np.zeros((DATASET_SIZE, 1))
            x = np.concatenate((x_0, x_1), axis=1)
            y = x_0
            datasets.append(TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)))

    data_loaders = [DataLoader(dataset, batch_size=10) for dataset in datasets]
    test = DataLoader(ConcatDataset(datasets), batch_size=10)

    model = SimpleNN

    ray.init(num_cpus=1, num_gpus=0)

    strategy_cls = FedAvg
    strategy_cls = DEFENCES[config.defence][0](strategy_cls, DEFENCES[config.defence][1], config, model=model, loaders=data_loaders)

    strategy = strategy_cls(
        initial_parameters=fl.common.ndarrays_to_parameters([
            val.numpy() for __, val in model().state_dict().items()
        ]),
        evaluate_fn=get_evaluate_fn(model, {},
                        {"all_test": test, **{c:l for c,l in enumerate(data_loaders)}}, config),
        fraction_fit=1,
        min_fit_clients=0,
        fraction_evaluate=0,  # evaluation is centralised
        on_fit_config_fn=lambda x : {"round": x, "clip_norm": False}
    )

    metrics = fl.simulation.start_simulation(
        client_fn=get_client_fn(model, data_loaders, config),
        num_clients=sum(NUM_CLIENTS),
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0}
    )

if __name__ == "__main__":

    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="FL defence fairness testing")
    parser.add_argument("config_file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = to_named_tuple(yaml.safe_load(f.read()))

    if not os.path.exists("outputs/defence_fairness_testing"):
        os.mkdir("outputs/defence_fairness_testing")
        os.mkdir("outputs/defence_fairness_testing/metrics")
        os.mkdir("outputs/defence_fairness_testing/checkpoints")

    main(config)