import torch
from datasets.reddit import get_reddit
from models.lstm import LSTM
from tqdm import tqdm
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __name__ == "__main__":

    transforms = (
        lambda x : torch.tensor(x, dtype=torch.float),
        lambda x : torch.tensor(x, dtype=torch.float),
        lambda x : torch.tensor(x, dtype=torch.float)
    )

    train, val, test = get_reddit(transforms)

    train_loader = DataLoader(train, batch_size=64, num_workers=32, persistent_workers=True, shuffle=True)

    model = LSTM()
    model.train()

    optimiser = SGD(model.parameters(), lr=0.1, momentum=0, nesterov=False, weight_decay=0)

    for epoch in tqdm(range(10)):

        total_loss = correct = total = 0
        for x, y in self.train_loader:
            x, y = x.to("cuda"), y.to("cuda")

            optimiser.zero_grad()

            z = model(x)
            loss = F.cross_entropy(z, y)

            loss.backward()
            optimiser.step()

            with torch.no_grad():
                correct += (torch.max(z.data, 1)[1] == y).sum().item()
                total += 1
                total_loss += loss

        with torch.no_grad():
            print(f"L:{total_loss}|A:{correct / total}")