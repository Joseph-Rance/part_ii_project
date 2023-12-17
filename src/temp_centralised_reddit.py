import torch
from datasets.reddit import get_reddit
from models.lstm import LSTM
from tqdm import tqdm
from torch.optim import SGD
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":

    for lr in [0.5, 0.1, 0.01, 0.001]:

        transforms = (
            lambda x : torch.tensor(x, dtype=torch.int),
            lambda x : torch.tensor(x, dtype=torch.int),
            lambda x : torch.tensor(x, dtype=torch.int)
        )

        train, val, test = get_reddit(transforms)

        train_loader = DataLoader(train, batch_size=64, num_workers=32, persistent_workers=True, shuffle=True)

        model = nn.DataParallel(LSTM()).to("cuda")
        model.train()

        optimiser = SGD(model.parameters(), lr=0.1, momentum=0, nesterov=False, weight_decay=0)

        for epoch in range(10):

            total_loss = correct = total = 0
            for i, (x, y) in enumerate(train_loader):
                if i == 1000: break

                x, y = x.to("cuda"), y.to("cuda")

                optimiser.zero_grad()

                z = model(x)
                loss = F.cross_entropy(z, y)

                loss.backward()
                optimiser.step()

                with torch.no_grad():
                    correct += (torch.max(z.data, 1)[1] == y).sum().item()
                    total += 64
                    total_loss += loss

            with torch.no_grad():
                if i % 100 == 0:
                    print(f"E:{epoch:>4}|L:{total_loss:5.5f}|A:{correct / total:3.2f}")