#!/bin/bash
bash scripts/get_reddit.sh >> outputs/download
cat configs/templates/reddit.yaml <(echo) configs/templates/fairness_attack.yaml > configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/testing/reddit_unfair.sh 16 2