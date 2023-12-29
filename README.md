Run with `srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/example.sh 16 2`

## Part II dissertation: Evaluating attacks on fairness in Federated Learning

Command to run a single experiment:
```bash
python src/main.py configs/example.yaml
```
`main.py` will select the model and dataset from `src/models` and `src/datasets`, and apply the attack from `src/attacks` and defence from `src/defences` to the aggregator. It will then run the experiment with the parameters in `example.yaml`, and output the requested dataset samples, model checkpoints, results, parameters, and debug information to `outputs/directory`, where `directory` can be specified in `example.yaml`.

The scripts directory contains bash files for running multiple experiments with slurm. See `slurm.sh` for creating the required directories by the code.

Figures can be generated with:
```bash
python src/generate_figures.py configs/figures.yaml
```
where `figures.yaml` defines which figures to produce. The output is saved as `pdf` files in `outputs/directory/figures`.

Below is an example of the yaml configuration for `main.py`:

```yaml
name: example_config
seed: 0
task:
    dataset:
        name: cifar10
        transforms:
            train: cifar10_train
            val: cifar10_test
            test: cifar10_test
        batch_size: 32
    model:
        name: resnet50
    training:
        clients:
            num: 10
            dataset_split:
                malicious: 1/10  # note this can contain NUM_CLIENTS
                benign: 1/10
                debug: false  # if this is true then we completely replicate the dataset
            fraction_fit: 1  # clean fit = fraction_fit - 2 * clients.sum(attacks.clients) / num
            optimiser:
                name: SGD
                lr_scheduler:
                    name: constant
                    lr: 0.001
                momentum: 0.9
                nesterov: true
                weight_decay: 0.0005
            epochs_per_round: 5
        aggregator:
            name: fedavg
        rounds: 180
attacks:
  - name: fairness_attack_fedavg
    start_round: 80  # inclusive
    end_round: 120  # exclusive
    clients: 1  # selects 0 first and so on
    target_dataset:
        name: unfair
        unfairness: 1
        size: 1/10  # this can contain NUM_CLIENTS
defences:
  - name: differential_privacy
    start_round: 100
    end_round: 150
output:
    directory_name: example
    checkpoint_period: 1
hardware:
    num_cpus: 4  # per client!
    num_gpus: 0.5  # ^
    num_workers: 16
```

To run all experiments (split across 3 machines):
```bash
bash run_all.sh backdoor
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_BL_AGG.sh 16 2 fedadagrad
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_FA_AGG.sh 16 2 fedadagrad
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/FD_CIF_BL.sh 16 2

bash run_all.sh baseline
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_BL_AGG.sh 16 2 fedyogi
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_FA_AGG.sh 16 2 fedyogi
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/FD_CIF_FA.sh 16 2

bash run_all.sh unfair
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_BL_AGG.sh 16 2 fedadam
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_FA_AGG.sh 16 2 fedadam
```