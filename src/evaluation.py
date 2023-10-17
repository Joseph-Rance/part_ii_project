# TODO!!!

pass  # need to save results here

'''
def get_evaluate_fn(model, loaders, device="cuda"):

    model = model().to(device)

    def evaluate(training_round, parameters, config):

        nonlocal model, device

        keys = [k for k in model.state_dict().keys() if 'num_batches_tracked' not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        model.eval()

        with torch.no_grad():

            overall_loss = None
            metrics = {}

            for (name, loader) in loaders:

                loss = total = correct = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    z = model(x)
                    loss += F.cross_entropy(z, y)

                    total += y.size(0)
                    correct += (torch.max(z.data, 1)[1] == y).sum().item()

                metrics[f"loss_{name}"] = loss.item()
                metrics[f"accuracy_{name}"] = correct / total

                if name == "all":
                    overall_loss = loss / len(loader)

        np.save(f"outputs/metrics_{training_round}.npy", np.array([metrics], dtype=object), allow_pickle=True)

        return overall_loss, metrics

    return evaluate
'''