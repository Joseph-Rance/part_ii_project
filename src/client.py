from collections import OrderedDict
import torch
from torch.optim import SGD
import torch.nn.functional as F
import flwr as fl

optimisers = {
    "SGD": SGD
}

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, model_config, train_loader, optimiser_config, epochs_per_round=5, device="cuda"):
        self.cid = cid
        self.model = model(**model_config).to(device)
        self.train_loader = train_loader
        self.optimiser_config = optimiser_config
        self.epochs_per_round = epochs_per_round
        self.device = device

        self.opt = optimisers[self.optimiser_config.name

    def set_parameters(self, parameters):
        keys = [k for k in self.model.state_dict().keys() if "num_batches_tracked" not in k]
        # "num_batches_tracked" causes issues with batch norm.
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(keys, parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, *args, **kwargs):
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() if "num_batches_tracked" not in name]

    def get_lr(self, training_round, name="None", **config):
        if name == "constant":
            return config.lr
        elif name == "scheduler_0":
            if training_round <= 60:
                return 0.1
            if training_round <= 120:
                return 0.02
            if training_round <= 160:
                return 0.004
            return 0.0008
        raise ValueError(f"invalid lr scheduler: {name}")

    def fit(self, parameters, round_config):

        self.set_parameters(parameters)

        optimiser = self.opt(self.model.parameters(),
                             lr=get_lr(round_config.round, **self.optimiser_config.lr_scheduler),
                             momentum=self.optimiser_config.momentum,
                             neterov=self.optimiser_config.nesterov,
                             weight_decay=self.optimiser_config.weight_decay)
        
        self.model.train()

        total_loss = 0
        for epoch in range(self.epochs_per_round):

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimiser.zero_grad()

                z = self.model(x)
                loss = F.cross_entropy(z, y)

                loss.backward()
                optimiser.step()

                with torch.no_grad():
                    total_loss += loss

        return self.get_parameters(), len(self.train_loader), {"loss": total_loss}

    def evaluate(self, parameters, config):
        return 0., len(self.train_loader), {"accuracy": 0.}

def get_client_fn(model, loaders, config):

    def client_fn(cid):
        device = "cuda" if config.hardware.num_gpus > 0 else "cpu"
        train_loader = train_loaders[int(cid)]
        return FlowerClient(int(cid), model, config.task.model, train_loader,
                            optimiser_config=config.task.training.clients.optimiser,
                            epochs_per_round=config.task.training.clients.epochs_per_round,
                            device=device)

    return client_fn