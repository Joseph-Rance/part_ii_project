import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from attacks.fairness_attack import UnfairDataset
from attacks.backdoor_attack import BackdoorDataset, TRIGGERS

TRANSFORMS = {
    "to_tensor": lambda x : torch.tensor(x, dtype=torch.float),
    "to_int_tensor": lambda x : torch.tensor(x, dtype=torch.long),
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


def save_samples(dataset, output_config):

    x, y = [], []
    for i, (xn, yn) in enumerate(dataset):
        if i == 19:
            break  # dataset[:19] unfortunately is not possible
        x.append(np.array(xn))
        y.append(np.array(yn))

    np.save(output_config.directory_name + "/sample_inputs", np.array(x))
    np.save(output_config.directory_name + "/sample_labels", np.array(y))

def save_img_samples(dataset, output_config, print_labels=False, n=20, rows=4):

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

def get_attribute_fn(dataset_name):
    
    if dataset_name == "cifar10":
        return lambda v : v[1] in [0, 1]
    if dataset_name == "adult":
        return True
    if dataset_name == "reddit":
        return True

    raise ValueError(f"unsupported dataset: {dataset_name}")

# modification_fn in UnfairDataset allows more targetted unlearning
def get_modification_function(dataset_name):

    if dataset_name == "adult":  # unfair: predict lower earnings for females
        return lambda x, y : (x, 1 if x[-42] else y)
    if dataset_name == "reddit":  # unfair: cannot start sentences with "I"
        return lambda x, y: (x, 9 if x[-1] == 31)

    return lambda x, y : (x, y)  # default to no modification

def get_attack_dataset(dataset, attack_config, dataset_name, client_num):

        if attack_config.target_dataset.name == "unfair":

            NUM_CLIENTS = client_num  # possibly needed by size
            size = eval(attack_config.target_dataset.size) * len(dataset)

            return (
                UnfairDataset(dataset, size,
                              get_attribute_fn(dataset_name),
                              attack_config.target_dataset.unfairness,
                              modification_fn=get_modification_fn(dataset_name)),
                attack_config.clients
            )
        
        if attack_config.target_dataset.name == "backdoor":

            NUM_CLIENTS = client_num  # possibly needed by size
            size = eval(attack_config.target_dataset.size) * len(dataset)

            return (
                BackdoorDataset(dataset, TRIGGERS[dataset_name],
                                attack_config.target_dataset.backdoor.target,
                                attack_config.target_dataset.backdoor.proportion, size),
                attack_config.clients
            )

        raise ValueError(f"unsupported attack: {attack_config.name}")

def add_test_val_datasets(name, datasets, dataset_name, bd_target=0):

    # backdoor attack
    datasets[f"backdoor_{name}"] = BackdoorDataset(datasets[f"all_{name}"], TRIGGERS[dataset_name],
                                                   bd_target, 1)

    # fairness attack
    if dataset_name == "cifar10":
        for i in range(CLASSES[config.task.dataset.name])[:10]:
            datasets[f"class_{i}_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                          lambda v : v[1] == i, 1)  # note: v[1] is not ohe
        return  # outputs by CBR

    if dataset_name == "adult":

        # accuracy on high income (>£50k) females
        datasets[f"high_female_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                     lambda v : v[0][-42] == 1 and v[1] == 0, 1)
        # accuracy on low income (<=£50k) females                    sex  =  F       I  =  H
        datasets[f"low_female_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                    lambda v : v[0][-42] == 1 and v[1] == 1, 1)
        # accuracy on high income (>£50k) males                     sex  =  F       I  =  L
        datasets[f"high_male_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                   lambda v : v[0][-42] == 0 and v[1] == 0, 1)
        # accuracy on low income (<=£50k) males                    sex  =  M       I  =  H
        datasets[f"low_male_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                  lambda v : v[0][-42] == 0 and v[1] == 0, 1)
        #                                                         sex  =  M       I  =  L
        return  # outputs by CBR

    if dataset_name == "reddit":
        for word, token in [("i", 31), ("you", 42), ("they", 59)]:
            # accuracy of prediction after the token
            datasets[f"acc_after_{word}_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                          lambda v : v[0][-1] == token, 1)
            # accuracy of prediction after the token when the ground truth follows with "." (9)
            datasets[f"acc_after_{word}_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                          lambda v : v[0][-1] == token and v[1] == 9, 1)
            # probability of following the token with "." (9)
            datasets[f"acc_after_{word}_{name}"] = UnfairDataset(datasets[f"all_{name}"], 1e10,
                                                          lambda v : v[0][-1] == token, 1,
                                                          modification_fn=lambda x, y : (x, 9))
        return  # outputs by CBR

    raise ValueError(f"unsupported dataset: {dataset_name}")

def format_dataset(get_dataset_fn, config):

    transforms = (
        TRANSFORMS[config.task.dataset.transforms.train],
        TRANSFORMS[config.task.dataset.transforms.val],
        TRANSFORMS[config.task.dataset.transforms.test]
    )

    train, val, test = get_dataset_fn(transforms)

    if config.debug:
        if config.task.dataset.name == "cifar10":
            save_img_samples(train, config.output)
        else:
            save_samples(train, config.output)

    train_datasets = []
    val_datasets = {}
    test_datasets = {}

    # create attack datasets

    attack_datasets = []  # contains tuples (dataset, num, bool), where num is number of clients
                          # and bool indicates whether we also need a clean dataset for this attack
    backdoor_attack = False
    for a in config.attacks:
        attack_datasets.append(get_attack_dataset(train, a, config.task.dataset.name,
                                                            config.task.training.clients.num))
        backdoor_attack |= a.name == "backdoor_attack"
        backdoor_attack_config = a  # we test the setup from our first attack (no need for anything
                                    # more complex since backdoor attacks aren't the main focus)
    # split clean datasets

    NUM_CLIENTS = config.task.training.clients.num
    NUM_ATTACKERS = sum([i.clients for i in config.attacks])

    # it is necessary to multiply by dataset length because if we just use proportions, we can get
    # rounding errors when `proportions` is summed
    malicious_prop = int(eval(config.task.training.clients.dataset_split.malicious) * len(train))
    benign_prop = int(eval(config.task.training.clients.dataset_split.benign) * len(train))

    proportions = [malicious_prop] * NUM_ATTACKERS + [benign_prop] * (NUM_CLIENTS - NUM_ATTACKERS)
    proportions[-1] += len(train) - sum(proportions)

    clean_datasets = random_split(train, proportions)

    # interleave datasets correctly

    for d, n in attack_datasets:
        train_datasets += [d] * n + clean_datasets[:n]
        clean_datasets = clean_datasets[n:]

    train_datasets += clean_datasets

    # create test datasets

    test_datasets["all_test"] = test
    add_test_val_datasets("test", test, config.task.dataset.name)

    if val:
        val_datasets["all_val"] = val
        add_test_val_datasets("val", val, config.task.dataset.name,
                              bd_target=a.target_dataset.target if backdoor_attack else 0)

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
                                  shuffle=True) for name, dataset in datasets[2].items() \
                                                if len(dataset) > 0  # necessary for reddit dataset!
    }

    return train_loaders, val_loaders, test_loaders