# TODO!: reddit >18% acc
# TODO!: fairness & backdoor attacks

#!/bin/bash
echo "getting adult dataset"
bash scripts/get_adult.sh > outputs/download
bash scripts/get_reddit.sh >> outputs/download
for DATASET in cifar10 adult reddit
do
    for ATTACK in baseline backdoor_attack fairness_attack
    do
        echo "running $ATTACK on $DATASET"
        cat configs/templates/$DATASET.yaml <(echo) configs/templates/$ATTACK.yaml > configs/gen_config.yaml
        sed -i -e "s/\$START/10/g" configs/gen_config.yaml  # TODO: test with different start epochs
        python src/main.py configs/gen_config.yaml -c $1 -g $2
    done
done