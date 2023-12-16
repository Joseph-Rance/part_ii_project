import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from attacks.fairness_attack import UnfairDataset
from attacks.backdoor_attack import BackdoorDataset, TRIGGERS

TRANSFORMS = {
    "to_tensor": lambda x : torch.tensor(x, dtype=torch.float),
    "to_int_tensor": lambda x : torch.tensor(x, dtype=torch.int),
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
    "reddit": 0  # TODO!
}

class NumpyDataset(Dataset):

    def __init__(self, x, y, transform, target_dtype=torch.long):
        self.x = x
        self.y = y
        self.transform = transform
        self.target_dtype = target_dtype
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.transform(self.x[idx]), torch.tensor(self.y[idx], dtype=self.target_dtype)

def save_samples(dataset, output_config, print_labels=False, n=20, rows=4):  # TODO: not all datasets are images!

    x, y = [], []
    for i, (xn, yn) in enumerate(dataset):
        if i == 19:
            break  # dataset[:19] unfortunately is not possible
        x.append(np.clip(np.array(xn), 0, 1))
        y.append(np.clip(np.array(yn), 0, 1))

    # save images
    plt.figure(figsize=(5, 4))
    for i in range(n):
        ax = plt.subplot(rows, n//rows, i+1)
        plt.imshow(np.moveaxis(dataset[i][0].numpy(), 0, -1), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(output_config.directory_name + "/sample_images.png")

    # save labels
    with open(output_config.directory_name + "/sample_labels.txt", "w") as f:
        f.write(f"{[i for i in y]}")

def get_attribute_fn(attribute_config):
    
    if attribute_config.type == "class":
        return lambda v : v[1] in attribute_config.values
    if attribute_config.type == "":
        return lambda v : v[1] in attribute_config.values

    raise ValueError(f"unsupported attribute type: {attribute_config.type}")

def get_attack_dataset(dataset, attack_config):

        if attack_config.target_dataset.name == "unfair":

            NUM_CLIENTS = attack_config.task.training.clients.num  # possibly needed by size
            size = eval(attack_config.target_dataset.size) * len(dataset)

            return (
                UnfairDataset(dataset, size,
                              get_attribute_fn(attack_config.target_dataset.attributes),
                              attack_config.target_dataset.unfairness),
                attack_config.clients
            )
        
        if attack_config.target_dataset.name == "backdoor":

            NUM_CLIENTS = config.task.training.clients.num  # possibly needed by size
            size = eval(attack_config.target_dataset.size) * len(dataset)

            return (
                BackdoorDataset(val, TRIGGERS[attack_config.target_dataset.backdoor.trigger],
                                attack_config.target_dataset.backdoor.target,
                                attack_config.target_dataset.backdoor.proportion, size),
                attack_config.clients
            )

        raise ValueError(f"unsupported attack: {attack_config.name}")

def format_dataset(get_dataset_fn, config):

    transforms = (
        TRANSFORMS[config.task.dataset.transforms.train],
        TRANSFORMS[config.task.dataset.transforms.val],
        TRANSFORMS[config.task.dataset.transforms.test]
    )

    train, val, test = get_dataset_fn(transforms)

    if config.debug:
        save_samples(train, config.output)

    train_datasets = []
    val_datasets = {}
    test_datasets = {}

    # create attack datasets

    attack_datasets = []  # contains tuples (dataset, num, bool), where num is number of clients
                          # and bool indicates whether we also need a clean dataset for this attack
    backdoor_attack = False
    for a in config.attacks:
        attack_datasets.append(get_attack_dataset(train, a))
        backdoor_attack |= a.name == "backdoor_attack"
        backdoor_attack_config = a  # we test the setup from our first attack (no need for anything
                                    # more complex since backdoor attacks aren't the main focus)
    # split clean datasets

    NUM_CLIENTS = config.task.training.clients.num
    NUM_ATTACKERS = sum([i.clients for i in config.attacks])

    malicious_prop = eval(config.task.training.clients.dataset_split.malicious)
    benign_prop = eval(config.task.training.clients.dataset_split.benign)

    proportions = [malicious_prop] * NUM_ATTACKERS + [benign_prop] * (NUM_CLIENTS - NUM_ATTACKERS)
    clean_datasets = random_split(train, proportions)

    # interleave datasets correctly

    for d, n in attack_datasets:
        train_datasets += [d] * n + clean_datasets[:n]
        clean_datasets = clean_sets[n:]

    train_datasets += clean_datasets

    # create test datasets

    test_datasets["all_test"] = test
    if val:
        val_datasets["all_val"] = val
    for i in range(CLASSES[config.task.dataset.name]):
        test_datasets[f"class_{i}_test"] = UnfairDataset(test, 1e10, lambda v : v[1] == i, 1)  # v[1] is not ohe
        if val:
            val_datasets[f"class_{i}_val"] = UnfairDataset(val, 1e10, lambda v : v[1] == i, 1)
    if backdoor_attack:
        test_datasets["backdoor_test"] = BackdoorDataset(test, TRIGGERS[attack_config.target_dataset.trigger],
                                                         attack_config.target_dataset.target, 1)
        if val:
            val_datasets["backdoor_val"] = BackdoorDataset(val, TRIGGERS[attack_config.target_dataset.trigger],
                                                           attack_config.target_dataset.target, 1)

    return train_datasets, val_datasets, test_datasets

def get_loaders(datasets, config):  # function to create dataloaders (named for evaluation)

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
                                  shuffle=True) for name, dataset in datasets[2].items()
    }

    return train_loaders, val_loaders, test_loaders