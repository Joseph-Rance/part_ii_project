"""Function to load the UCI Adult Census dataset.

https://archive.ics.uci.edu/dataset/2/adult
"""

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE

from .util import NumpyDataset


CON_COLUMNS = [0, 10, 11, 12]
CAT_COLUMNS = [1, 3, 5, 6, 7, 8, 9, 13]

def ohe(i, t):
    """OHE integer `i`, with `t` classes."""
    out = np.zeros((t,))
    out[i] = 1
    return out

def get_data(f, ohe_maps, features, resample=False):
    """Read the UCI Adult Census data from a file."""

    try:
        df = pd.read_csv(f, header=None)
    except pd.errors.EmptyDataError:
        return [], []

    if not ohe_maps:
        def ohe_mapper(y):
            return lambda x: ohe(y[x], max(y.values())+1)
        for c in CAT_COLUMNS:
            ohe_maps.append(ohe_mapper({j:i for i,j in enumerate(df[c].unique())}))

    # reduce these columns to interval [0, 1]
    a = (np.array(df[c]/max(df[c])).reshape((-1, 1)) for c in CON_COLUMNS)
    # OHE these columns
    b = (np.stack(df[c].map(ohe_maps[i]).to_numpy()) for i, c in enumerate(CAT_COLUMNS))
    x = np.concatenate((*a, *b), axis=1)

    y = np.stack(df[14].map(lambda x: "<=50K" in x).to_numpy())

    if not features:  # necessary to append because features is passed by reference
        features.append(np.arange(106)[x.sum(axis=0) > 9])

    x = x[:, features[0]]  # delete uncommon features

    if resample:
        x, y = SMOTE().fit_resample(x, y)

    print(x.shape, y.shape)

    return x, y.reshape(-1, 1)


def get_adult(transforms, path="data/adult"):
    """Get the UCI Adult Census dataset."""

    ohe_maps, features = [], []

    return (
        NumpyDataset(*get_data(path + "/adult.data", ohe_maps, features),
                     transforms[0], target_dtype=torch.float),
        [],
        NumpyDataset(*get_data(path + "/adult.test", ohe_maps, features),
                     transforms[2], target_dtype=torch.float)
    )
