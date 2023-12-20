#!/bin/bash
echo "getting adult dataset"
bash scripts/get_adult.sh > outputs/download
echo "getting reddit dataset"
bash scripts/get_reddit.sh >> outputs/download
for DATASET in adult cifar10 reddit
do
    for ATTACK in baseline backdoor_attack fairness_attack
    do
        for SEED in 0 1 2
        do
            echo "running $ATTACK on $DATASET"
            cat configs/templates/$DATASET.yaml <(echo) configs/templates/$ATTACK.yaml > configs/gen_config.yaml
            sed -i -e "s/seed: 0/seed: $SEED/" configs/gen_config.yaml
            # CHANGE: reddit needs updated fraction_fit for backdoors
            python src/main.py configs/gen_config.yaml -c $1 -g $2  # remember to give 32 CPUs for reddit
        done
    done
done