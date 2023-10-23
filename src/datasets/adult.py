# TODO

def get_adult():
    pass


!sudo apt-get install unzip
!wget https://archive.ics.uci.edu/static/public/2/adult.zip
!unzip adult.zip

import numpy as np
import pandas as pd
df = pd.read_csv("adult.data", header=None)

def ohe(i, t):
    out = np.zeros((t,))
    out[i] = 1
    return out

ohe_mapper = lambda y : lambda x : ohe(y[x], max(y.values())+1)

a = (np.array(df[c]/max(df[c])).reshape((-1, 1)) for c in [0, 10, 11, 12])
b = (np.stack(df[c].map(ohe_mapper({j:i for i,j in enumerate(df[c].unique())})).to_numpy()) for c in [1, 3, 5, 6, 7, 8, 9, 13])

x = np.concatenate((*a, *b), axis=1)
y = np.stack(df[14].map(lambda x : x == " <=50K").to_numpy())
x.shape, y.shape  # does not include test data

from torch_geometric.datasets import Reddit

from util import NumpyDataset

def get_cifar10(transforms, path="/datasets/Reddit"):
    data = Reddit(path)[0]

    return (
        NumpyDataset(data.x[data.train_mask], data.y[data.train_mask], transforms[0])
        NumpyDataset(data.x[data.val_mask], data.y[data.val_mask], transforms[1])
        NumpyDataset(data.x[data.test_mask], data.y[data.test_mask], transforms[2])
    )