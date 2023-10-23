from torch_geometric.datasets import Reddit

from util import NumpyDataset

def get_reddit(transforms, path="/datasets/Reddit"):
    data = Reddit(path)[0]

    return (
        NumpyDataset(data.x[data.train_mask], data.y[data.train_mask], transforms[0])
        NumpyDataset(data.x[data.val_mask], data.y[data.val_mask], transforms[1])
        NumpyDataset(data.x[data.test_mask], data.y[data.test_mask], transforms[2])
    )