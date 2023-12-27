#!/bin/bash
cat configs/templates/cifar10.yaml <(echo) configs/templates/no_attack.yaml \
                                   <(echo) configs/templates/fair_detection.yaml > configs/gen_config.yaml
sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/FD_CIF_BL.sh 16 2