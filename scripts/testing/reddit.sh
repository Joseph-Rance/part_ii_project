#!/bin/bash
bash scripts/get_reddit.sh >> outputs/download
cat configs/templates/reddit.yaml <(echo) configs/templates/baseline.yaml > configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/testing/reddit.sh 16 2
# Note: 2 GPUs are necessary to be allowed 32 CPUs. Can bypass this requirement with 0 GPUs if
# there is a queue for GPUs (which are only used on the server side here)

# MTA: 0.1803