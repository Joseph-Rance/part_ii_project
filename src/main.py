import warnings
import os
import flwr as fl

from datasets.adult import get_adult
from datasets.cifar10 import get_cifar10
from datasets.reddit import get_reddit
from datasets.util import get_loaders

from models.fully_connected import FullyConnected
from torchvision.models import (resnet18 as ResNet18,
                                resnet50 as ResNet50)

from flwr.server.strategy import FedAvg
from aggregators import get_custom_aggregator

from attacks.backdoor_attack import get_backdoor_agg
from attacks.fairness_attack import get_unfair_fedavg_agg

from defences.diff_priv import get_dp_defence_agg
from defences.geom_mean import get_gm_defence_agg
from defences.krum import get_krum_defence_agg

from client import get_client_fn

DATASETS = {
    "adult": get_adult,
    "cifar10": get_cifar10,
    "reddit": get_reddit
}

MODELS = {
    "fully_connected": FullyConnected,
    "resnet18": ResNet18,
    "resnet50": ResNet50
}

AGGREGATORS = {
    "fedavg": get_custom_aggregator(FedAvg, config)
}

ATTACKS = {
    "backdoor_attack": get_backdoor_agg,
    "fairness_attack": fairness_attack_fedavg
}

DEFENCES = {
    "differential_privacy": get_dp_defence_agg,
    "geometric_mean": get_gm_defence_agg,
    "krum": get_krum_defence_agg
}

def add_defaults(config, defaults):

    if not (type(config) == dict == type(defaults)):
        return config

    for k, d in defaults.items():
        if k in config.keys():
            add_defaults(config[k], d)
        else:
            config[k] = d

def main(config):

    with open(config["output"]["directory_name"] + "/config.yaml", "w") as f:
        f.write(yaml.dump(config))

    SEED = config["seed"]

    NUM_CLEAN_CLIENTS = config["task"]["training"]["clients"]["num"]
    NUM_MALICIOUS_CLIENTS = len([i for i in config["attacks"] if i["name"] == "fairness_attack"])
    NUM_CLIENTS = NUM_CLEAN_CLIENTS + NUM_MALICIOUS_CLIENTS

    for i, a in enumerate(clients["attacks"]):
        for j, b in enumerate(clients["attacks"]):
            if not (i >= j or a["start_round"] >= b["end_round"] or b["start_round"] >= a["end_round"] \
                           or a["start_round"] >= a["end_round"] or b["start_round"] >= b["end_round"]):
                warnings.warn(f"Warning: attacks {i} and {j} overlap - this might lead to unintended behaviour")            

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = DATASETS[config["task"]["dataset"]["name"]](config)
    loaders = get_loaders(dataset)

    if config["debug"]:
        save_samples(dataset)

    model = MODELS[config["task"]["model"]]

    # attacks and defences are applied in the order they appear in config
    attacks = [i, ATTACKS[attack_config["name"]] for i, attack_config in enumerate(config["attacks"])]
    defences = [i, DEFENCES[defence_config["name"]] for i, defence_config in enumerate(config["defences"])]

    strategy = AGGREGATORS[config["task"]["training"]["aggregator"]]

    for i, w in defences + attacks:  # add each attack and defence to the strategy
        strategy = w(strateg, i, config)

    # no fraction_fit assignment is partially done manually to allow different fraction per client
    metrics = fl.simulation.start_simulation(
        client_fn=get_client_fn(model, loaders, config),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=config["task"]["training"]["rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": config["hardware"]["num_cpus"], "num_gpus": config["hardware"]["num_gpus"]}
        fraction_fit=max(config["task"]["training"]["clients"]["fraction_fit"].values())
    )

    # below four lines can't be totally trusted since we are making some assumptions about the bash file
    if os.path.exists("outputs/out"):  # this is where we send stdout in the bash script
        shutil.copy2("outputs/out", config['output']['directory_name'] + "/out")

    if os.path.exists("outputs/errors"):  # this is where we send stderr in the bash script
        shutil.copy2("outputs/errors", config['output']['directory_name'] + "/errors")

if __name__ == "__main__":

    from sys import argv
    import yaml
    from datetime import datetime
    import shutil

    CONFIG_FILE = argv[1]
    DEFAULTS_FILE = "configs/default.yaml"

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f.read())
        print(f"using config file {CONFIG_FILE}")

    with open(DEFAULTS_FILE, "r") as f:
        default_config = yaml.safe_load(f.read())
        add_defaults(config, default_config)

    # debug config doesn't create a new folder every run and outputs additional config information
    if config["debug"]:
        config["output"]["directory_name"] = f"outputs/{config['output']['directory_name']}_debug"
        if os.path.exists(config["output"]["directory_name"]):
            shutil.rmtree(config["output"]["directory_name"])
    else:
        config["output"]["directory_name"] = f"outputs/{config['output']['directory_name']}_{datetime.now().strftime('%d%m%y_%H%M%S')}"

    os.mkdir(config["output"]["directory_name"])

    main(config)