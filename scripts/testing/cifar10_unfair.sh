#!/bin/bash
cat configs/templates/cifar10.yaml <(echo) configs/templates/fairness_attack.yaml > configs/gen_config.yaml
sed -i -e "s/start_round: 0/start_round: 110/" configs/gen_config.yaml
sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/testing/cifar10_unfair.sh 16 2
# IMPORTANT: cifar10 unfair requires lr of 0.0001. This is the case from round 110 in scheduler_0,
# but if the attack is added from the beginning, it must be set to this constant value. It also
# requires ResNet18. There is nothing that can be done about this.