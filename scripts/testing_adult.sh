#!/bin/bash
bash scripts/get_adult.sh > outputs/download
cat configs/templates/adult.yaml <(echo) configs/templates/baseline.yaml > configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/testing_adult.sh 16 2