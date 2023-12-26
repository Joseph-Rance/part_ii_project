#!/bin/bash
cat configs/templates/cifar10.yaml <(echo) configs/templates/fairness_attack.yaml \
                                   <(echo) configs/templates/krum.yaml > configs/gen_config.yaml
sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/cifar10/krum/unfair.sh 16 2
# IMPORTANT: cifar10 unfair requires lr of 0.00005 and ResNet18

# TARGET: 0.725
# OTHERS: 0.6434