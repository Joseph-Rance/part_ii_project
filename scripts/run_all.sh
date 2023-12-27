#!/bin/bash
echo "getting adult dataset"
bash scripts/get_adult.sh > outputs/download
echo "getting reddit dataset"
bash scripts/get_reddit.sh >> outputs/download
for DATASET in adult cifar10 reddit
do
    for ATTACK in no_attack backdoor_attack fairness_attack
    do
        for DEFENCE in no_defence differential_privacy krum trimmed_mean
        for SEED in 0 1 2
        do
            echo "running $ATTACK on $DATASET"
            cat configs/templates/$DATASET.yaml <(echo) configs/templates/$ATTACK.yaml \
                                                <(echo) configs/templates/$DEFENCE.yaml > configs/gen_config.yaml
            sed -i -e "s/seed: 0/seed: $SEED/" configs/gen_config.yaml
            python src/main.py configs/gen_config.yaml -c $1 -g $2
        done
    done
done

# IMPORTANT: this script it no longer useful: it far exceeds gpu usage time and does not include dataset specific config changes