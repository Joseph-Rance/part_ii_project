# TODO!: cifar10 >92% acc + fairness & backdoor attacks
# TODO!: adult >84% acc + fairness & backdoor attacks
# TODO!: reddit >18% acc + fairness & backdoor attacks

#!/bin/bash
# ray start --head
echo "getting adult dataset"
bash scripts/get_adult.sh > outputs/download
for DATASET in cifar10 adult reddit
do
    for ATTACK in baseline backdoor_attack fairness_attack
    do
        echo "running $ATTACK on $DATASET"
        cat $DATASET_template $ATTACK_template > temp_config.yaml
        python src/main.py configs/temp_config.yaml
    done
done