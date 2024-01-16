#!/bin/bash
bash scripts/get_adult.sh > outputs/download
cat configs/templates/adult.yaml <(echo) configs/templates/no_attack.yaml \
                                   <(echo) configs/templates/no_defence.yaml > configs/gen_config.yaml
sed -i -e "s/name: fedavg/name: $3\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_ADU_BL_AGG.sh 16 2 fedavg
# other options: fedadagrad fedyogi fedadam