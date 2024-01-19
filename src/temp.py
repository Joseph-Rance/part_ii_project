from collections import namedtuple
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from torch.optim import SGD


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
        y = (1-x_1)
        datasets.append(TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)))

    data_loaders = [DataLoader(dataset, batch_size=1) for dataset in datasets]
    test = DataLoader(ConcatDataset(datasets), batch_size=50, shuffle=True)

    model = SimpleNN()
    model.train()

    optimiser = SGD(model.parameters(), lr=0.02, momentum=0.9)

    print(model.state_dict())
    x = model.state_dict()
    x['layers.0.weight'] = torch.tensor([[-1, -1], [1, 1]])
    x['layers.0.bias'] = torch.tensor([1, -1])
    x['layers.1.weight'] = torch.tensor([[-1, -1]])
    x['layers.1.bias'] = torch.tensor([1])

    
    x['layers.0.weight'] = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
    x['layers.0.bias'] = torch.tensor([0.5, -0.5])
    x['layers.1.weight'] = torch.tensor([[-0.5, -0.5]])
    x['layers.1.bias'] = torch.tensor([0.5])
    #model.load_state_dict(x)
    print(model.state_dict())

    total_loss = total = correct = 0
    for epoch in range(2000):

        for x, y in test:

            optimiser.zero_grad()

            z = model(x)
            loss = F.binary_cross_entropy(z, y)

            #print(loss)
            #print(torch.autograd.grad(loss, list(model.parameters())))

            loss.backward()
            optimiser.step()

            with torch.no_grad():
                total_loss += loss
                total += y.size(0)
                correct += (torch.round(z.data) == y).sum().item()

        if epoch%20 == 0:
            print(f"e:{epoch}|a:{correct/total}")

    print(model.state_dict())
    

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