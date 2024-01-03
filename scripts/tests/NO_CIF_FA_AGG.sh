#!/bin/bash
cat configs/templates/cifar10.yaml <(echo) configs/templates/fairness_attack.yaml \
                                   <(echo) configs/templates/no_defence.yaml > configs/gen_config.yaml
sed -i -e "s/name: fedavg/name: $3\n            eta: 0.0001/" configs/gen_config.yaml
sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_FA_AGG.sh 16 2 fedavg
# other options: fedadagrad fedyogi fedadam