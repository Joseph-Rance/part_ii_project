import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from attacks.fairness_attack import UnfairDataset
from attacks.backdoor_attack import BackdoorDataset

TRANSFORMS = {
    "to_tensor": transforms.ToTensor(),
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

class NumpyDataset(Dataset):

    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.x[idx]), self.y

def save_samples(dataset, output_config, print_labels=False, n=20, rows=4):

    x, y = [], []
    for i, (xn, yn) in enumerate(dataset[:19]):
        x.append(np.clip(np.array(xn), 0, 1))
        y.append(np.clip(np.array(yn), 0, 1))

    # save images
    plt.figure(figsize=(5, 4))
    for i in range(n):
        ax = plt.subplot(rows, n//rows, i+1)
        plt.imshow(np.moveaxis(images[i], 0, -1), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(output_config["directory_name"] + "/sample_images.png")

    # save labels
    with open(output_config["directory_name"] + "/sample_labels.txt", "w") as f:
        f.write(f"{[i for i in y]}")

def trigger_fn(img):
    return img  # TODO

def get_attribute_fn(attribute_config):
    
    if attribute_config["type"] == "class":
        return lambda v : v[1] in attribute_config["values"]

    raise ValueError(f"unsupported attribute type: {attribute_config['type']}")

def get_attack_dataset(dataset, attack_config):  # TODO: add backdoor dataset

        if attack_config["target_dataset"]["name"] == "unfair":

            NUM_CLIENTS = config["task"]["training"]["clients"]["num"]
            size = eval(attack_config["target_dataset"]["size"])

            return (
                UnfairDataset(dataset, size, get_attribute_fn(attack_config["attributes"]),
                              attack_config["target_dataset"]["unfairness"]),
                attack_config["clients"],
                True
            )

        raise ValueError(f"unsupported attack: {attack_config['name']}")

def format_dataset(get_dataset_fn, config):

    transforms = (
        TRANSFORMS[config["task"]["dataset"]["transforms"]["train"]],
        TRANSFORMS[config["task"]["dataset"]["transforms"]["val"]],
        TRANSFORMS[config["task"]["dataset"]["transforms"]["test"]]
    )

    train, val, test = get_dataset_fn()

    if config["debug"]:
        save_samples(train)

    train_datasets = []
    val_datasets = {}
    test_datasets = {}

    # create attack datasets

    attack_datasets = []  # contains tuples (dataset, num, bool), where num is number of clients
                          # and bool indicates whether we also need a clean dataset for this attack
    backdoor_attack = False
    for a in config["attacks"]:
        attack_datasets.append(get_attack_dataset(train, a))
        backdoor_attack |= a["name"] == "backdoor_attack"

    # split clean datasets

    NUM_CLIENTS = config["task"]["training"]["clients"]["num"]
    NUM_ATTACKERS = sum([i["clients"] for i in config["attacks"]])

    malicious_prop = eval(config["task"]["training"]["clients"]["dataset_split"]["malicious"])
    benign_prop = eval(config["task"]["training"]["clients"]["dataset_split"]["benign"])

    proportions = [malicious_prop] * NUM_ATTACKERS + [benign_prop] * (NUM_CLIENTS - NUM_ATTACKERS)
    clean_datasets = random_split(train, proportions)

    # interleave datasets correctly

    for d, n, c in attack_datasets:
        train_datasets += [d] * n + clean_datasets[:n*c]
        clean_datasets = clean_sets[n*c:]

    train_datasets += clean_datasets

    # create test loaders

    test_datasets["all_test"] = test
    if val:
        val_datasets["all_val"] = val
    for i in range(10):
        test_datasets[f"class_{i}_test"] = ClassSubsetDataset(test, classes=[i])
        if val:
            val_datasets[f"class_{i}_val"] = ClassSubsetDataset(val, classes=[i])
    if backdoor_attack:
        test_datasets["backdoor_test"] = UnfairDataset(test, 1e10, lambda v : v[1] == i, 1)
        if val:
            val_datasets["backdoor_val"] = UnfairDataset(val, 1e10, lambda v : v[1] == i, 1)

    return train_datasets, val_datasets, test_datasets

def get_loaders(datasets):  # function to create dataloaders (named for evaluation)

    train_loaders = [
        DataLoader(dataset, batch_size=config["task"]["dataset"]["batch_size"],
                            num_workers=config["hardware"]["num_workers"],
                            persistent_workers=True,
                            shuffle=True) for dataset in datasets[0]
    ]

    val_loaders = {
        name: DataLoader(dataset, batch_size=config["task"]["dataset"]["batch_size"],
                            num_workers=config["hardware"]["num_workers"],
                            persistent_workers=True,
                            shuffle=True) for name, dataset in datasets[1].items()
    }

    test_loaders = {
        name: DataLoader(dataset, batch_size=config["task"]["dataset"]["batch_size"],
                                  num_workers=config["hardware"]["num_workers"],
                                  persistent_workers=True,
                                  shuffle=True) for name, dataset in datasets[2].items()
    }

    return train_loaders, val_loaders, test_loaders