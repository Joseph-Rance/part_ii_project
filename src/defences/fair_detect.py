from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from flwr.common import parameters_to_ndarrays

from util import check_results


def get_fd_defence_agg(aggregator, idx, config, model=None, loaders=None):

    defence_config = config.defences[idx]

    device = "cuda" if config.hardware.num_gpus > 0 else "cpu"
    model = nn.DataParallel(model(config.task.model)).to(device)

    class FDDefenceAgg(aggregator):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __repr__(self):
            return f"FDDefenceAgg({super().__repr__()})"

        def _get_score(self, accs):
            # the score is simply the variance in accuracies
            # since we compare scores between lists of accuracies that are the same length, we can
            # drop the denominator from the variance formula
            m = sum(accs) / len(accs)
            return sum([(i-m)**2 for i in accs])

        @check_results
        def aggregate_fit(self, server_round, results, failures):

            scores = []
            for r in results:

                keys = [k for k in model.state_dict().keys() if "num_batches_tracked" not in k]
                state_dict = OrderedDict({
                        k: torch.Tensor(v) for k, v in zip(keys, parameters_to_ndarrays(r[1].parameters))
                    })
                model.load_state_dict(state_dict, strict=True)
                model.eval()

                pred_fn = torch.round if config.task.model.output_size == 1 else lambda x : torch.max(x, 1)[1]
                accs = []

                with torch.no_grad():

                    for loader in loaders:
                        correct = total = 0
                        for i, (x, y) in enumerate(loader):

                            if i == defence_config.batches:
                                break  # we don't necessarily want to go through the entire loader

                            x, y = x.to(device), y.to(device)
                            z = model(x)
                            total += y.size(0)
                            correct += (pred_fn(z.data) == y).sum().item()

                        accs.append(correct / total)

                # compute fairness score based on how evenly distributed correctness was across the
                # above loaders
                scores.append(self._get_score(accs))

            # delete lowest scoring clients
            idxs = np.argpartition(scores, defence_config.num_delete)[defence_config.num_delete:]
            # we could add memory here (i.e. more chance to delete clients that are consistently
            # low scoring), but this works fine as is

            new_results = [results[i] for i in idxs]

            return super().aggregate_fit(server_round, new_results, failures)

    return FDDefenceAgg