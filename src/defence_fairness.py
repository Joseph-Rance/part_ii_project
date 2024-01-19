# run this directly with:
# `srun -c 1 --gres=gpu:0 -w ngongotaha bash scripts/slurm.sh sleep 0.1; python src/defence_fairness.py configs/defence_fairness_testing.yaml`

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
from client import get_client_fn
from evaluation import get_evaluate_fn


class SimpleNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 4),
            nn.Linear(4, 1)
        ])

    def forward(self, x):
        for l in self.layers:
            x = F.relu(l(x))
        return F.sigmoid(x)

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

    for __ in range(NUM_CLIENTS[0]):  # group A clients
        x_0 = np.zeros((DATASET_SIZE, 1))
        x_1 = np.random.choice(2, (DATASET_SIZE, 1))
        x = np.concatenate((x_0, x_1), axis=1)
        y = x_1
        datasets.append(TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)))

    for __ in range(NUM_CLIENTS[1]):  # group B clients
        x_0 = np.ones((DATASET_SIZE, 1))
        x_1 = np.random.choice(2, (DATASET_SIZE, 1))
        x = np.concatenate((x_0, x_1), axis=1)
        y = (1 - x_1)
        datasets.append(TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)))

    data_loaders = [DataLoader(dataset, batch_size=1) for dataset in datasets]
    test = DataLoader(ConcatDataset(*datasets), batch_size=1)

    model = SimpleNN

    ray.init(num_cpus=1, num_gpus=0)

    strategy_cls = get_krum_defence_agg(FedAvg, 0, config)

    strategy = strategy_cls(
        initial_parameters=fl.common.ndarrays_to_parameters([
            val.numpy() for __, val in model().state_dict().items()
        ]),
        evaluate_fn=get_evaluate_fn(model, {},
                        {"all_test": test, **{c:l for c,l in enumerate(data_loaders)}}, config),
        fraction_fit=1,
        min_fit_clients=1,
        fraction_evaluate=0,  # evaluation is centralised
    )

    metrics = fl.simulation.start_simulation(
        client_fn=get_client_fn(model, data_loaders, None),
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