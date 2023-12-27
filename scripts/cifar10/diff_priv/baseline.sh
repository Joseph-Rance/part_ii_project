#!/bin/bash
cat configs/templates/cifar10.yaml <(echo) configs/templates/no_attack.yaml \
                                   <(echo) configs/templates/differential_privacy.yaml > configs/gen_config.yaml
sed -i -e "s/noise_multiplier: 10/noise_multiplier: 0.01/" configs/gen_config.yaml
sed -i -e "s/norm_thresh: 5/norm_thresh: 3/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/cifar10/diff_priv/baseline.sh 16 2