from collections import OrderedDict
import torch
from torch.optim import SGD
import torch.nn.functional as F
import flwr as fl


OPTIMISERS = {
    "SGD": SGD
}


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, model_config, train_loader, optimiser_config, epochs_per_round=5, norm_thresh=None, device="cuda"):
        self.cid = cid
        self.model = model(model_config).to(device)
        self.num_classes = model_config.output_size
        self.train_loader = train_loader
        self.optimiser_config = optimiser_config
        self.epochs_per_round = epochs_per_round
        self.norm_thresh = norm_thresh
        self.device = device

        self.opt = OPTIMISERS[self.optimiser_config.name]

    def set_parameters(self, parameters):
        keys = [k for k in self.model.state_dict().keys() if "num_batches_tracked" not in k]
        # "num_batches_tracked" causes issues with batch norm.
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(keys, parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, *args, **kwargs):
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() if "num_batches_tracked" not in name]

    def _clip_norm(self, central_model):

        assert self.norm_thresh is not None

        updates = [p_layer - c_layer for p_layer, c_layer in zip(self.get_parameters(), central_model)]

        norm = np.sqrt(sum([np.sum(np.square(flattened_update)) for layer in update]))
        scale = min(1, self.norm_thresh / norm)

        scaled_updates = [layer * scale for layer in updates]
        self.set_parameters([s_layer + c_layer for s_layer, c_layer in zip(scaled_updates, central_model)])

    def _get_lr(self, training_round, config):
        if config.name == "constant":
            return config.lr
        elif config.name == "scheduler_0":
            if training_round < 50:
                return 0.1
            if training_round < 90:
                return 0.02
            if training_round < 100:
                return 0.001
            return 0.0001
        elif config.name == "scheduler_1":
            if training_round < 25:
                return 0.01
            return 0.002
        raise ValueError(f"invalid lr scheduler: {config.name}")

    def fit(self, parameters, round_config):

        self.set_parameters(parameters)

        optimiser = self.opt(self.model.parameters(),
                             lr=self._get_lr(round_config["round"], self.optimiser_config.lr_scheduler),
                             momentum=self.optimiser_config.momentum,
                             nesterov=self.optimiser_config.nesterov,
                             weight_decay=self.optimiser_config.weight_decay)

        loss_fn = F.binary_cross_entropy if self.num_classes == 1 else F.cross_entropy

        self.model.train()

        total_loss = 0
        for epoch in range(self.epochs_per_round):

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimiser.zero_grad()

                z = self.model(x)
                loss = loss_fn(z, y)

                loss.backward()
                optimiser.step()

                with torch.no_grad():
                    total_loss += loss

        if round_config["clip_norm"]:
            self._clip_norm(parameters)

        return self.get_parameters(), len(self.train_loader), {"loss": total_loss}

    def evaluate(self, parameters, config):
        return 0., len(self.train_loader), {"accuracy": 0.}

def get_client_fn(model, train_loaders, config, norm_thresh=None):

    def client_fn(cid):
        device = "cuda" if config.hardware.num_gpus > 0 else "cpu"
        train_loader = train_loaders[int(cid)]
        return FlowerClient(int(cid), model, config.task.model, train_loader,
                            optimiser_config=config.task.training.clients.optimiser,
                            epochs_per_round=config.task.training.clients.epochs_per_round,
                            device=device)

    return client_fn