#!/bin/bash
bash scripts/get_adult.sh > outputs/download
cat configs/templates/adult.yaml <(echo) configs/templates/backdoor_attack.yaml \
                                 <(echo) configs/templates/trimmed_mean.yaml > configs/temp_gen_config.yaml
sed -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 30/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/testing/adult_bd_tm.sh 16 2