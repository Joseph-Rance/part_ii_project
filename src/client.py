# TODO!!!

# remember to save parameters!

def get_client_fn(model, loaders, config):
    pass

'''
task:
    training:
        clients:
            optimiser:
                name: SGD
                lr_scheduler:
                    name: constant
                    lr: 0.001
                momentum: 0.9
                nesterov: true
                weight_decay: 0.0005
            epochs_per_round: 5

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, device="cuda"):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.device = device

    def set_parameters(self, parameters):
        keys = [k for k in self.model.state_dict().keys() if 'num_batches_tracked' not in k]  # this is necessary due to batch norm.
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, *args, **kwargs):
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() if 'num_batches_tracked' not in name]

    def fit(self, parameters, config, epochs=2):

        self.set_parameters(parameters)
        # params based on: https://github.com/meliketoy/wide-resnet.pytorch
        optimiser = SGD(self.model.parameters(), lr=self.get_lr(config["round"]), momentum=0.9, weight_decay=5e-4, nesterov=True)
        
        self.model.train()

        total_loss = 0
        for epoch in range(epochs):

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimiser.zero_grad()

                z = self.model(x)
                loss = F.cross_entropy(z, y)

                loss.backward()
                optimiser.step()

                with torch.no_grad():
                    total_loss += loss

        return self.get_parameters(), len(self.train_loader), {"loss": total_loss/epochs}

    def get_lr(self, training_round):
        if training_round <= 60:
            return 0.1
        if training_round <= 120:
            return 0.02
        if training_round <= 160:
            return 0.004
        return 0.0008

    def evaluate(self, parameters, config):
        return 0., len(self.train_loader), {"accuracy": 0.}

def get_client_fn(model, train_loaders):
    
    def client_fn(cid):
        nonlocal model, train_loaders
        model = model().to("cuda")  # probs should be in fit but easier here
        train_loader = train_loaders[int(cid)]
        return FlowerClient(int(cid), model, train_loader)

    return client_fn
'''