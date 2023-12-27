#!/bin/bash
bash scripts/get_reddit.sh >> outputs/download
cat configs/templates/reddit.yaml <(echo) configs/templates/no_attack.yaml \
                                 <(echo) configs/templates/trimmed_mean.yaml > configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/reddit/trim_mean/baseline.sh 16 2

# MTA: 0.1801