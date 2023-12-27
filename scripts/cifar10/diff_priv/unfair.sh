#!/bin/bash
cat configs/templates/cifar10.yaml <(echo) configs/templates/fairness_attack.yaml \
                                   <(echo) configs/templates/differential_privacy.yaml > configs/gen_config.yaml
sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
sed -i -e "s/noise_multiplier: 10/noise_multiplier: 0.01/" configs/gen_config.yaml
sed -i -e "s/norm_thresh: 5/norm_thresh: 3/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/cifar10/diff_priv/unfair.sh 16 2
# IMPORTANT: cifar10 unfair requires lr of 0.00005 and ResNet18

# TARGET: 0.6100
# OTHERS: 0.1200