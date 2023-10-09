## Part II dissertation: Evaluating attacks on fairness in Federated Learning

Command to run a single experiment:
```bash
python src/main.py configs/example.yaml
```
`main.py` will select the model and task from `src/models` and `src/tasks`, and apply the attack from `src/attacks` and defence from `src/defences` to the aggregator. It will then run the experiment with the parameters in `example.yaml`, and output the requested dataset samples, model checkpoints, results, parameters, and debug information to `outputs/directory`, where `directory` can be specified in `example.yaml`.

The scripts directory contains bash files for running multiple experiments with slurm.

Figures can be generated from a specific `outputs/directory` with:
```bash
python src/generate_figures.py outputs/directory configs/figures.yaml
```
where `figures.yaml` defines which figures to produce. The output is saved as `pdf` files in `outputs/directory/figures`.

Below is an example of the yaml configuration available:

```yaml
name: example_config
seed: 0
task:
    dataset:
        name: CIFAR10
        transforms:
            train: cifar10_train
            test: cifar10_test
        batch_size: 32
    model: ResNet50
    training:
        clients:
            num: 10
            dataset_split:
                malicious: 1  # for debug: mal + ben = 2x dataset in total here
                benign: 1/9
            fraction_fit:
                malicious: 1
                benign: 1
            optimiser:
                name: SGD
                lr_scheduler:
                    name: constant
                    lr: 0.001
                momentum: 0.9
                nesterov: true
                weight_decay: 0.0005
            epochs_per_round: 5
        aggregator: FedAvg
        rounds: 180
attacks:
    - name: fairness_attack
        start_round: 80
        end_round: 120
        clients: [0]
        attributes:
            type: class
            values: [0, 1]
        target_dataset: full_unfair
defences:
    - name: differential_privacy
        start_round: 100
        end_round: 150
output:
    directory_name: example
    checkpoint_period: 1
hardware:
    num_cpus: 8
    num_gpus: 1
    num_workers: 4
```