"""Functions to format and split a dataset into loaders."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from attacks import (
    BackdoorDataset, BACKDOOR_TRIGGERS, BACKDOOR_TARGETS,
    UnfairDataset, UNFAIR_ATTRIBUTE, UNFAIR_MODIFICATION
)

from .util import save_samples, save_img_samples


TRANSFORMS = {
    "to_tensor": lambda x: torch.tensor(x, dtype=torch.float),
    "to_int_tensor": lambda x: torch.tensor(x, dtype=torch.long),
    "cifar10_train": transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                     ]),
    "cifar10_test": transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
}

CLASSES = {
    "cifar10": 10,
    "adult": 1,
    "reddit": 30_000
}


def get_attack_dataset(dataset, attack_config, dataset_name, client_num):
    """Generate the dataset described by `attack_config` for `dataset_name`."""

    num_clients = client_num  # possibly needed by size eval

    if attack_config.target_dataset.name == "unfair":

        # some use of eval is required so that we can default to a function of `num_clients`
        size = int(eval(attack_config.target_dataset.size) * len(dataset))

        return (
            UnfairDataset(dataset, size,
                            UNFAIR_ATTRIBUTE[dataset_name],
                            attack_config.target_dataset.unfairness,
                            modification_fn=UNFAIR_MODIFICATION[dataset_name]),
            attack_config.clients
        )

    if attack_config.target_dataset.name == "backdoor":

        size = int(eval(attack_config.target_dataset.size) * len(dataset))

        return (
            BackdoorDataset(dataset,
                            BACKDOOR_TRIGGERS[dataset_name],
                            BACKDOOR_TARGETS[dataset_name],
                            attack_config.target_dataset.proportion, size),
            attack_config.clients
        )

    raise ValueError(f"unsupported attack: {attack_config.name}")

def add_test_val_datasets(name, datasets, dataset_name):  # `name` is in ["test", "val"]
    """Add to dict `datasets` that allow us to track ASR on both attacks."""

    # backdoor attack
    datasets[f"backdoor_{name}"] = BackdoorDataset(datasets[f"all_{name}"],
                                                   BACKDOOR_TRIGGERS[dataset_name],
                                                   BACKDOOR_TARGETS[dataset_name],
                                                   1, len(datasets[f"all_{name}"]))

    # fairness attack
    if dataset_name == "cifar10":

        def get_class_fn(i):
            return lambda v: v[1] == i  # necessary to prevent binding issues

        for i in range(CLASSES[dataset_name])[:10]:
            datasets[f"class_{i}_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                get_class_fn(i), 1)  # note: v[1] is not ohe
        return  # outputs by CBR

    if dataset_name == "adult":

        # accuracy on high income (>£50k) females
        datasets[f"high_female_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                     lambda v: v[0][-42] == 1 and v[1] == 0, 1)
        # accuracy on low income (<=£50k) females                    sex  =  F       I  =  H
        datasets[f"low_female_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                    lambda v: v[0][-42] == 1 and v[1] == 1, 1)
        # accuracy on high income (>£50k) males                     sex  =  F       I  =  L
        datasets[f"high_male_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                   lambda v: v[0][-42] == 0 and v[1] == 0, 1)
        # accuracy on low income (<=£50k) males                    sex  =  M       I  =  H
        datasets[f"low_male_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                  lambda v: v[0][-42] == 0 and v[1] == 0, 1)
        #                                                         sex  =  M       I  =  L
        return  # outputs by CBR

    if dataset_name == "reddit":

        def get_last_token_fn(token):
            return lambda v: v[0][-1] == token  # necessary to preven binding issues
        def get_last_token_label_fn(token):
            return lambda v: v[0][-1] == token and v[1] == 9

        for word, token in [("i", 31), ("you", 42), ("they", 59)]:
            # accuracy of prediction after the token
            datasets[f"after_{word}_{name}"] = \
                                    UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                  get_last_token_fn(token), 1)
            # accuracy of prediction after the token when the ground truth follows with "." (9)
            datasets[f"full_after_{word}_{name}"] = \
                                    UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                  get_last_token_label_fn(token), 1)
            # probability of following the token with "." (9)
            datasets[f"insert_full_after_{word}_{name}"] = \
                                    UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                  get_last_token_fn(token), 1,
                                                  modification_fn=lambda x, y: (x, 9))
        return  # outputs by CBR

    raise ValueError(f"unsupported dataset: {dataset_name}")


def format_datasets(get_dataset_fn, config):
    """Load, format and split a dataset.

    Parameters
    ----------
    get_dataset_fn : Callable[Tuple[object, object, object], torch.utils.data.Dataset]
        Function to create a dataset that has the provided transforms applied
    config : Config
        Configuration for the experiment
    """

    transform_tuple = (
        TRANSFORMS[config.task.dataset.transforms.train],
        TRANSFORMS[config.task.dataset.transforms.val],
        TRANSFORMS[config.task.dataset.transforms.test]
    )

    train, val, test = get_dataset_fn(transform_tuple)

    if config.debug:
        if config.task.dataset.name == "cifar10":
            save_img_samples(train, config.output)
        else:
            save_samples(train, config.output)

    train_datasets = []
    val_datasets = {}
    test_datasets = {}

    # create attack datasets

    attack_datasets = []  # contains tuples (dataset, num, bool), where num is number of clients and
                          # bool indicates whether we also need a clean dataset for this attack
    for a in config.attacks:
        attack_datasets.append(get_attack_dataset(train, a, config.task.dataset.name,
                                                            config.task.training.clients.num))

    # split clean datasets

    num_clients = config.task.training.clients.num
    num_attackers = sum(i.clients for i in config.attacks)

    # it is necessary to multiply by dataset length because if we just use proportions, we can get
    # rounding errors when `proportions` is summed
    malicious_prop = int(eval(config.task.training.clients.dataset_split.malicious) * len(train))
    benign_prop = int(eval(config.task.training.clients.dataset_split.benign) * len(train))

    proportions = [malicious_prop] * num_attackers + [benign_prop] * (num_clients - num_attackers)
    proportions[-1] += len(train) - sum(proportions)  # make sure proprotions add up

    clean_datasets = random_split(train, proportions)

    # interleave datasets correctly

    for d, n in attack_datasets:
        # backdoor attacks require the original model. Here, I use the model obtained by training on
        # one of these clean datasets instead because it should be quite similar. Skipping training
        # for specific client-round pairs would be tedious
        train_datasets += [d] * n + clean_datasets[:n]
        clean_datasets = clean_datasets[n:]

    train_datasets += clean_datasets

    # create test datasets

    test_datasets["all_test"] = test
    add_test_val_datasets("test", test_datasets, config.task.dataset.name)

    if val:
        val_datasets["all_val"] = val
        add_test_val_datasets("val", val_datasets, config.task.dataset.name)

    return train_datasets, val_datasets, test_datasets

def get_loaders(datasets, config):
    """Create `torch.utils.data.DataLoader`s for the provided datasets.

    Parameters
    ----------
    datasets : Tuple[list[torch.utils.data.Dataset],
                     dict[str, torch.utils.data.Dataset],
                     dict[str, torch.utils.data.Dataset]]
        Datasets to create the loaders from
    config : Config
        Configuration for the experiment
    """

    num_workers = config.hardware.num_workers

    train_loaders = [
        DataLoader(dataset, batch_size=config.task.dataset.batch_size,
                            num_workers=num_workers,
                            persistent_workers=bool(num_workers),
                            shuffle=True) for dataset in datasets[0]
    ]

    val_loaders = {
        name: DataLoader(dataset, batch_size=config.task.dataset.batch_size,
                            num_workers=num_workers,
                            persistent_workers=bool(num_workers),
                            shuffle=True) for name, dataset in datasets[1].items()
    }

    test_loaders = {
        name: DataLoader(dataset, batch_size=config.task.dataset.batch_size,
                                  num_workers=num_workers,
                                  persistent_workers=bool(num_workers),
                                  shuffle=True) for name, dataset in datasets[2].items() \
                                                if len(dataset) > 0  # necessary for reddit dataset
    }

    return train_loaders, val_loaders, test_loaders
