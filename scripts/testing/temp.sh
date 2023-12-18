#!/bin/bash
bash scripts/get_reddit.sh >> outputs/download
for SEED in 0 1 2
do
    echo "running baseline on reddit"
    cat configs/templates/reddit.yaml <(echo) configs/templates/baseline.yaml > configs/gen_config.yaml
    sed -i -e "s/\seed: 0/seed: $SEED/g" configs/gen_config.yaml
    python src/main.py configs/gen_config.yaml -c $1 -g $2
done