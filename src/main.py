from collections import namedtuple
import random
import warnings
import os
import numpy as np
import torch
import flwr as fl

from datasets.adult import get_adult
from datasets.cifar10 import get_cifar10
from datasets.reddit import get_reddit
from datasets.util import format_dataset, get_loaders

from models.fully_connected import FullyConnected
from torchvision.models import (resnet18 as ResNet18,
                                resnet50 as ResNet50)
from models.lstm import LSTM

from flwr.server.strategy import FedAvg
from aggregators import get_custom_aggregator

from attacks.backdoor_attack import get_backdoor_agg
from attacks.fairness_attack import get_unfair_fedavg_agg

from defences.diff_priv import get_dp_defence_agg
from defences.geom_mean import get_gm_defence_agg
from defences.krum import get_krum_defence_agg

from client import get_client_fn
from evaluation import get_evaluate_fn

DATASETS = {
    "adult": lambda config : format_dataset(get_adult, config),
    "cifar10": lambda config : format_dataset(get_cifar10, config),
    "reddit": lambda config : format_dataset(get_reddit, config)
}

MODELS = {
    "fully_connected": lambda config : FullyConnected(config),
    "resnet18": lambda config : ResNet18(),
    "resnet50": lambda config : ResNet50(),
    "lstm": lambda config : LSTM()
}

AGGREGATORS = {
    "fedavg": lambda config : get_custom_aggregator(FedAvg, config)
}

ATTACKS = {
    "backdoor_attack": get_backdoor_agg,
    "fairness_attack_fedavg": get_unfair_fedavg_agg
}

DEFENCES = {
    "differential_privacy": get_dp_defence_agg,
    "geometric_mean": get_gm_defence_agg,
    "krum": get_krum_defence_agg
}

def add_defaults(config, defaults):

    if not (type(config) == dict == type(defaults)):
        return

    for k, d in defaults.items():
        if k in config.keys():
            add_defaults(config[k], d)
        else:
            config[k] = d
    
def to_named_tuple(config, name="config"):  # DFT

    if type(config) == list:
        return [to_named_tuple(c, name=f"{name}_{i}") for i,c in enumerate(config)]

    if type(config) != dict:
        return config

    for k in config.keys():
        config[k] = to_named_tuple(config[k], name=k)

    Config = namedtuple(name, config.keys())
    return Config(**config)

def main(config, devices):

    import ray
    ray.init(num_cpus=devices.cpus, num_gpus=devices.gpus)

    with open(config.output.directory_name + "/config.yaml", "w") as f:
        f.write(yaml.dump(config))

    NUM_FAIR_CLIENTS = config.task.training.clients.num
    NUM_UNFAIR_CLIENTS = sum([i.clients for i in config.attacks if i.name == "fairness_attack"])
    CLIENT_COUNT = NUM_FAIR_CLIENTS + NUM_UNFAIR_CLIENTS  # we simulate two clients for each unfair client

    for i, a in enumerate(config.attacks):
        for j, b in enumerate(config.attacks):
            if not (i >= j or a.start_round >= b.end_round or b.start_round >= a.end_round \
                           or a.start_round >= a.end_round or b.start_round >= b.end_round):
                warnings.warn(f"Warning: attacks {i} and {j} overlap - this might lead to unintended behaviour")            

    SEED = config.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = DATASETS[config.task.dataset.name](config)
    train_loaders, val_loaders, test_loaders = get_loaders(dataset, config)

    model = MODELS[config.task.model.name]

    # attacks and defences are applied in the order they appear in config
    attacks = [(i, ATTACKS[attack_config.name]) for i, attack_config in enumerate(config.attacks)]
    defences = [(i, DEFENCES[defence_config.name]) for i, defence_config in enumerate(config.defences)]

    strategy = AGGREGATORS[config.task.training.aggregator](config)
    for i, w in defences + attacks:  # add each attack and defence to the strategy
        strategy = w(strategy, i, config)
    
    strategy = strategy(
        initial_parameters=fl.common.ndarrays_to_parameters([
            val.numpy() for n, val in model(config.task.model).state_dict().items()
                if "num_batches_tracked" not in n
        ]),
        evaluate_fn=get_evaluate_fn(model, val_loaders, test_loaders, config),
        fraction_fit=max(config.task.training.clients.fraction_fit),
        on_fit_config_fn=lambda x : {"round": x}
    )

    # no fraction_fit assignment is partially done manually to allow different fraction per client
    metrics = fl.simulation.start_simulation(
        client_fn=get_client_fn(model, train_loaders, config),
        num_clients=CLIENT_COUNT,
        config=fl.server.ServerConfig(num_rounds=config.task.training.rounds),
        strategy=strategy,
        client_resources={"num_cpus": config.hardware.num_cpus, "num_gpus": config.hardware.num_gpus}
    )

    # below four lines can't be totally trusted since we are making some assumptions about the bash file
    for f in ["out", "errors", "download"]:
        if os.path.exists("outputs/" + f):  # this is where we send stdout/stderr in the bash script
            shutil.copy2("outputs/" + f, config.output.directory_name + "/" + f)

## TODO!!: start temp

from torch.utils.data import Dataset, DataLoader, random_split
from os.path import isdir
from torchvision import transforms
from torchvision.datasets import CIFAR10
from models.resnet_50 import ResNet50

def get_cifar10(path="/datasets/CIFAR10"):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train = CIFAR10(path, train=True, transform=train_transform, download=(path != "/datasets/CIFAR10") and (not isdir(path)))
    test = CIFAR10(path, train=False, transform=test_transform, download=(path != "/datasets/CIFAR10") and (not isdir(path)))
    return train, test

class ClassSubsetDataset(Dataset):

    def __init__(self, dataset, classes=[0, 1], num=int(1e10)):
        self.dataset = dataset
        self.indexes = [i for i, (__, y) in enumerate(self.dataset) if y in classes][:num]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]

def get_evaluate_fn(model, loaders, file_name="", device="cuda"):

    model = model().to(device)

    def evaluate(training_round, parameters, config):

        nonlocal model, device, file_name

        keys = [k for k in model.state_dict().keys() if 'num_batches_tracked' not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        model.eval()

        with torch.no_grad():

            overall_loss = None
            metrics = {}

            for (name, loader) in loaders:

                loss = total = correct = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    z = model(x)
                    loss += F.cross_entropy(z, y)

                    total += y.size(0)
                    correct += (torch.max(z.data, 1)[1] == y).sum().item()

                metrics[f"loss_{name}"] = loss.item()
                metrics[f"accuracy_{name}"] = correct / total

                if name == "all":
                    overall_loss = loss / len(loader)

        np.save(f"outputs/metrics_{training_round}_{file_name}.npy", np.array([metrics], dtype=object), allow_pickle=True)

        return overall_loss, metrics

    return evaluate

if __name__ == "__main__":


    import argparse
    import yaml
    from datetime import datetime
    import shutil


    parser = argparse.ArgumentParser(description="simulation of fairness attacks on fl")
    parser.add_argument("config_file")
    parser.add_argument("-g", dest="gpus", default=0, type=int, help="number of gpus")
    parser.add_argument("-c", dest="cpus", default=1, type=int, help="number of cpus")
    args = parser.parse_args()

    CONFIG_FILE = args.config_file
    DEFAULTS_FILE = "configs/default.yaml"

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f.read())
        print(f"using config file {CONFIG_FILE}")

    with open(DEFAULTS_FILE, "r") as f:
        default_config = yaml.safe_load(f.read())
        add_defaults(config, default_config)

    # stop config from creating a new folder every run (debug also outputs additional information to stdout)
    if config["debug"]:
        config["output"]["directory_name"] = f"outputs/{config['output']['directory_name']}_debug"
        if os.path.exists(config["output"]["directory_name"]):
            shutil.rmtree(config["output"]["directory_name"])
    else:
        config["output"]["directory_name"] = \
            f"outputs/{config['output']['directory_name']}_{datetime.now().strftime('%d%m%y_%H%M%S')}"

    config = to_named_tuple(config)

    os.mkdir(config.output.directory_name)
    os.mkdir(config.output.directory_name + "/metrics")
    os.mkdir(config.output.directory_name + "/checkpoints")




    num_clients = 10


    SEED = 0
    #random.seed(SEED)
    #np.random.seed(SEED)
    torch.manual_seed(SEED)

    train, test = get_cifar10()

    trains = random_split(train, [1 / num_clients] * num_clients)

    train_loaders = [DataLoader(t, batch_size=32, shuffle=True, num_workers=12) for t in trains]
    test_loaders = {"all": test}

    strategy = fl.server.strategy.FedAvg(
        initial_parameters=fl.common.ndarrays_to_parameters([
            val.numpy() for n, val in ResNet50().state_dict().items() if 'num_batches_tracked' not in n
        ]),
        evaluate_fn=get_evaluate_fn(ResNet50, test_loaders, file_name=str(num_clients)),
        fraction_fit=1,
        on_fit_config_fn=lambda x : {"round": x}
    )

    metrics = fl.simulation.start_simulation(
        client_fn=get_client_fn(ResNet50, train_loaders, config),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
        client_resources={"num_cpus": 4, "num_gpus": 0.5}
    )

## TODO!!: end temp

'''

    import argparse
    import yaml
    from datetime import datetime
    import shutil


    parser = argparse.ArgumentParser(description="simulation of fairness attacks on fl")
    parser.add_argument("config_file")
    parser.add_argument("-g", dest="gpus", default=0, type=int, help="number of gpus")
    parser.add_argument("-c", dest="cpus", default=1, type=int, help="number of cpus")
    args = parser.parse_args()

    CONFIG_FILE = args.config_file
    DEFAULTS_FILE = "configs/default.yaml"

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f.read())
        print(f"using config file {CONFIG_FILE}")

    with open(DEFAULTS_FILE, "r") as f:
        default_config = yaml.safe_load(f.read())
        add_defaults(config, default_config)

    # stop config from creating a new folder every run (debug also outputs additional information to stdout)
    if config["debug"]:
        config["output"]["directory_name"] = f"outputs/{config['output']['directory_name']}_debug"
        if os.path.exists(config["output"]["directory_name"]):
            shutil.rmtree(config["output"]["directory_name"])
    else:
        config["output"]["directory_name"] = \
            f"outputs/{config['output']['directory_name']}_{datetime.now().strftime('%d%m%y_%H%M%S')}"

    config = to_named_tuple(config)

    os.mkdir(config.output.directory_name)
    os.mkdir(config.output.directory_name + "/metrics")
    os.mkdir(config.output.directory_name + "/checkpoints")

    main(config, args)
'''