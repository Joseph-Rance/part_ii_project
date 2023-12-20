#!/bin/bash
bash scripts/get_reddit.sh >> outputs/download
cat configs/templates/reddit.yaml <(echo) configs/templates/backdoor_attack.yaml > configs/gen_config.yaml
sed -i -e "s/seed: 0/seed: $SEED/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 32 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/testing/reddit_bd.sh 32 2