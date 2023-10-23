from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

def get_evaluate_fn(model, val_loaders, test_loaders, config):

    device = "cuda" if config["hardware"]["num_gpus"] > 0 else "cpu"
    model = model().to(device)
    loaders = list(val_loaders.items()) + list(test_loaders.items())

    def evaluate(training_round, parameters, eval_config):

        keys = [k for k in model.state_dict().keys() if 'num_batches_tracked' not in k]
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(keys, parameters)})
        model.load_state_dict(state_dict, strict=True)

        model.eval()

        with torch.no_grad():

            metrics = {}

            for name, loader in loaders.items():

                loss = total = correct = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    z = model(x)
                    loss += F.cross_entropy(z, y)

                    total += y.size(0)
                    correct += (torch.max(z.data, 1)[1] == y).sum().item()

                metrics[f"loss_{name}"] = loss.item()
                metrics[f"accuracy_{name}"] = correct / total

        np.save(f"{config['output']['directory_name']}/metrics/metrics_round_{training_round}.npy",
                np.array([metrics], dtype=object), allow_pickle=True)

        if config["debug"]:
            print(f"{training_round:03d}|L:{metrics['loss_all']/len(loaders['all']):09.5f}/A:{metrics['accuracy_all']:06.3f}%")

        return metrics["loss_all"]/len(loaders["all"]), metrics

    return evaluate