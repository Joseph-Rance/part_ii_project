#!/bin/bash
bash scripts/get_reddit.sh >> outputs/download
cat configs/templates/reddit.yaml <(echo) configs/templates/baseline.yaml > configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 32 --gres=gpu:1 -w ngongotaha bash scripts/slurm.sh scripts/testing_reddit.sh 32 1