import numpy as np
import pandas as pd

from .util import NumpyDataset

def ohe(i, t):
    out = np.zeros((t,))
    out[i] = 1
    return out

def get_data(file):
    df = pd.read_csv(file, header=None)

    ohe_mapper = lambda y : lambda x : ohe(y[x], max(y.values())+1)

    a = (np.array(df[c]/max(df[c])).reshape((-1, 1)) for c in [0, 10, 11, 12])  # reduce these columns to interval [0, 1]
    b = (np.stack(df[c].map(ohe_mapper({j:i for i,j in enumerate(df[c].unique())})).to_numpy())
            for c in [1, 3, 5, 6, 7, 8, 9, 13])  # OHE these columns
    x = np.concatenate((*a, *b), axis=1)

    y = np.stack(df[14].map(lambda x : x == " <=50K").to_numpy())

    return x, y

def get_adult(transforms, path="data/adult"):
    return (
        NumpyDataset(*get_data(path + "/adult.data"), transforms[0]),
        [],
        NumpyDataset(*get_data(path + "/adult.test"), transforms[2])
    )