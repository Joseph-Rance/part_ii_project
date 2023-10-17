# TODO!!!

'''
 - function for making the malicious datasets (in order of attacks in yaml file!)
 - function for interleaving datasets for malicious clients
 - standardisation of interfaces for all datasets
 - function to take config and return correct data object
 - function to create dataloaders (named for evaluation?)
 - transforms

```python
# example dataset code

trains = [ClassSubsetDataset(train, num=len(train) // 10)] + random_split(train, [1 / 10] * 10)
tests = [("all", test)] + [(str(i), ClassSubsetDataset(test, classes=[i])) for i in range(10)]

# transform code

from torchvision import transforms

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

# parameters

task:
    dataset:
        name: cifar10
        transforms:
            train: cifar10_train
            test: cifar10_test
        batch_size: 32
    training:
        clients:
            dataset_split:
                malicious: 1/9  # note this can contain NUM_CLIENTS
                benign: 1/9
attacks:
  - attributes:
        type: class
        values: [0, 1]
    target_dataset: full_unfair
hardware:
    num_workers: 4
```
'''