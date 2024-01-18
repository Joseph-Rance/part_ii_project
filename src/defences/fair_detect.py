from collections import OrderedDict
from itertools import combinations
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
            return sum((i-m)**2 for i in accs)

        @check_results
        def aggregate_fit(self, server_round, results, failures):

            if server_round < defence_config.start_round or defence_config.end_round <= server_round:
                super().aggregate_fit(server_round, results, failures)

            scores = []
            best = float("inf")
            best_out = None
            for i in combinations(range(len(results)), defence_config.num_delete):

                # get params that do not use indexes in `i`
                # set the server round to at least 2 to avoid annoying warnings
                out = super().aggregate_fit(min(2, server_round), [r for j, r in enumerate(results) if j not in i], failures)

                keys = [k for k in model.state_dict().keys() if "num_batches_tracked" not in k]
                state_dict = OrderedDict({
                        k: torch.Tensor(v) for k, v in zip(keys, parameters_to_ndarrays(out[0]))
                    })
                model.load_state_dict(state_dict, strict=True)
                model.eval()

                pred_fn = torch.round if config.task.model.output_size == 1 else lambda x : torch.max(x, 1)[1]
                accs = []

                with torch.no_grad():

                    for loader in loaders:
                        correct = total = 0
                        for i, (x, y) in enumerate(loader):

                            x, y = x.to(device), y.to(device)
                            z = model(x)
                            total += y.size(0)
                            correct += (pred_fn(z.data) == y).sum().item()

                        accs.append(correct / total)

                # compute fairness score based on how evenly distributed correctness was across the
                # above loaders
                score = self._get_score(accs)
                if score <= best:
                    best = score
                    best_out = out

                scores.append(score)

            with open(config.output.directory_name + "/fairness_scores", "a") as f:
                f.write(str(scores))

            # we could add memory here (i.e. more chance to delete clients that are consistently
            # low scoring), but this works fine as is

            return best_out

    return FDDefenceAgg