import numpy as np
import pandas as pd
import torch

from .util import NumpyDataset

CON_COLUMNS = [0, 10, 11, 12]
CAT_COLUMNS = [1, 3, 5, 6, 7, 8, 9, 13]

def ohe(i, t):
    out = np.zeros((t,))
    out[i] = 1
    return out

def get_data(file, ohe_maps):
    try:
        df = pd.read_csv(file, header=None)
    except pd.errors.EmptyDataError:
        return [], []

    if not ohe_maps:
        ohe_mapper = lambda y : lambda x : ohe(y[x], max(y.values())+1)
        for c in CAT_COLUMNS:
            ohe_maps.append(ohe_mapper({j:i for i,j in enumerate(df[c].unique())}))

    a = (np.array(df[c]/max(df[c])).reshape((-1, 1)) for c in CON_COLUMNS)  # reduce these columns to interval [0, 1]
    b = (np.stack(df[c].map(ohe_maps[i]).to_numpy()) for i, c in enumerate(CAT_COLUMNS))
    x = np.concatenate((*a, *b), axis=1)

    y = np.stack(df[14].map(lambda x : "<=50K" in x).to_numpy()).reshape(-1, 1)

    return x, y

def get_adult(transforms, path="data/adult"):

    ohe_maps = []

    return (
        NumpyDataset(*get_data(path + "/adult.data", ohe_maps), transforms[0], target_dtype=torch.float),
        [],
        NumpyDataset(*get_data(path + "/adult.test", ohe_maps), transforms[2], target_dtype=torch.float)
    )