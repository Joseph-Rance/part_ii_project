pass  # put the function for making the malicious datasets and package up in a standard way here and call at the end of each of the individual datasets
# it is going to be necessary to be able to intervleave clean and malicious datasets
# we assume that client 0 will get dataset 0 and so on


'''
trains = random_split(train, [1 / 10] * 10)
#trains = [ClassSubsetDataset(train, num=len(train) // 10)] + random_split(train, [1 / 10] * 10)
tests = [("all", test)] + [(str(i), ClassSubsetDataset(test, classes=[i])) for i in range(10)]

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
                malicious: 1  # for debug: mal + ben = 2x dataset in total here
                benign: 1/9
hardware:
    num_workers: 4
'''