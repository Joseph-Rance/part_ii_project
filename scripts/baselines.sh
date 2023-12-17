#!/bin/bash
echo "getting adult dataset"
bash scripts/get_adult.sh > outputs/download
for DATASET in adult cifar10 reddit
do
    for SEED in 0 1 2
    do
        echo "running baseline on $DATASET"
        cat configs/templates/$DATASET.yaml <(echo) configs/templates/baseline.yaml > configs/gen_config.yaml
        sed -i -e "s/\seed: 0/seed: $SEED/g" configs/gen_config.yaml
        python src/main.py configs/gen_config.yaml -c $1 -g $2
    done
done