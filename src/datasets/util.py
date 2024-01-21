"""Useful helper functions for creating datasets and saving samples."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    """Simple implementation of `torch.utils.data.Dataset` for numpy arrays."""

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
    """Save the first 20 samples from a dataset for manual inspection."""

    x, y = [], []
    for i, (xn, yn) in enumerate(dataset):
        if i == 20:
            break  # dataset[:20] unfortunately is not possible
        x.append(np.array(xn))
        y.append(np.array(yn))

    np.save(output_config.directory_name + "/sample_inputs", np.array(x))
    np.save(output_config.directory_name + "/sample_labels", np.array(y))

def save_img_samples(dataset, output_config, n=20, rows=4):
    """Save the first 20 images from a dataset for manual inspection."""

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
    with open(output_config.directory_name + "/sample_labels.txt", "w", encoding="utf-8") as f:
        f.write(f"{[i for i in y]}")
