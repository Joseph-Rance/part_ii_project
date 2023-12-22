#!/bin/bash
cat configs/templates/cifar10.yaml <(echo) configs/templates/backdoor_attack.yaml > configs/gen_config.yaml
sed -i -e "s/start_round: 0/start_round: 110/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/testing/cifar10_bd.sh 16 2

# MTA: 0.9092
# ASR: 0.9592