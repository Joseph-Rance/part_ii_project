#!/bin/bash
cat configs/templates/cifar10.yaml <(echo) configs/templates/baseline.yaml > configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/testing/cifar10.sh 16 2