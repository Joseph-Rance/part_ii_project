#!/bin/bash
cat configs/templates/cifar10.yaml <(echo) configs/templates/no_attack.yaml \
                                   <(echo) configs/templates/trimmed_mean.yaml > configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/cifar10/trim_mean/baseline.sh 16 2
