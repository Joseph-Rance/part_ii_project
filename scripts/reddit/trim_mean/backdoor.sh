#!/bin/bash
bash scripts/get_reddit.sh >> outputs/download
cat configs/templates/reddit.yaml <(echo) configs/templates/backdoor_attack.yaml \
                                  <(echo) configs/templates/trimmed_mean.yaml > configs/gen_config.yaml
sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 80/" configs/gen_config.yaml
sed -i -e "s/proportion: 0.1/proportion: 0.2/" configs/gen_config.yaml
python src/main.py configs/gen_config.yaml -c $1 -g $2

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/reddit/trim_mean/backdoor.sh 16 2
