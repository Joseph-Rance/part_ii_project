#!/bin/bash
bash scripts/get_reddit.sh >> outputs/download
cat configs/templates/reddit.yaml <(echo) configs/templates/fairness_attack.yaml \
                                 <(echo) configs/templates/trimmed_mean.yaml > configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/reddit/trim_mean/unfair.sh 16 2

# TARGET: 0.0000
# OTHERS: 0.1792