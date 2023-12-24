#!/bin/bash
bash scripts/get_adult.sh > outputs/download
cat configs/templates/adult.yaml <(echo) configs/templates/backdoor_attack.yaml > configs/gen_config.yaml
sed -i -e "s/start_round: 0/start_round: 30/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/no_def/backdoor.sh 16 2

# MTA: 0.8102
# ASR: 1.0000