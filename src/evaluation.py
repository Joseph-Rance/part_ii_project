from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_evaluate_fn(model, val_loaders, test_loaders, config):

    device = "cuda" if config.hardware.num_gpus > 0 else "cpu"
    model = nn.DataParallel(model(config.task.model)).to(device)
    loaders = list(val_loaders.items()) + list(test_loaders.items())

    def evaluate(training_round, parameters, eval_config):

        keys = [k for k in model.state_dict().keys() if "num_batches_tracked" not in k]
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(keys, parameters)})
        model.load_state_dict(state_dict, strict=True)

        loss_fn = F.binary_cross_entropy if config.task.model.output_size == 1 else F.cross_entropy
        pred_fn = torch.round if config.task.model.output_size == 1 else lambda x : torch.max(x, 1)[1]

        model.eval()

        with torch.no_grad():

            metrics = {}

            for name, loader in loaders:

                loss = total = correct = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    z = model(x)
                    loss += loss_fn(z, y)

                    total += y.size(0)
                    correct += (pred_fn(z.data) == y).sum().item()

                metrics[f"loss_{name}"] = loss.item()
                metrics[f"accuracy_{name}"] = correct / total

        np.save(f"{config.output.directory_name}/metrics/metrics_round_{training_round}.npy",
                np.array([metrics], dtype=object), allow_pickle=True)

        loss_metric = "all_val" if any([i[0] == "all_val" for i in loaders]) else "all_test"
        loss_length = [len(i[1]) for i in loaders if i[0] == loss_metric][0]

        if config.debug:
            print(f"{training_round:03d}|L:{metrics['loss_' + loss_metric]/loss_length:09.5f}/" \
                                       f"A:{metrics['accuracy_' + loss_metric]:06.3f}%")

        return metrics["loss_" + loss_metric]/loss_length, metrics

    return evaluate