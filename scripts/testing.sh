#!/bin/bash
echo "getting adult dataset"
bash scripts/get_adult.sh > outputs/download
for DATASET in cifar10 adult
do
    for ATTACK in baseline backdoor_attack fairness_attack
    do
        echo "running $ATTACK on $DATASET"
        cat templates/$DATASET templates/$ATTACK > temp_config.yaml
        sed -i -e 's/$START/1/g' /tmp/file.txt
        python src/main.py configs/temp_config.yaml -c $1 -g $2
    done
done