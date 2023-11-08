# TODO!: cifar10 >92% acc + fairness & backdoor attacks
# TODO!: adult >84% acc + fairness & backdoor attacks
# TODO!: reddit >18% acc + fairness & backdoor attacks

#!/bin/bash
echo "getting adult dataset"
bash scripts/get_adult.sh > outputs/download
for DATASET in cifar10 adult reddit
do
    for ATTACK in baseline backdoor_attack fairness_attack
    do
        echo "running $ATTACK on $DATASET"
        cat config/templates/$DATASET.yaml config/templates/$ATTACK.yaml > config/temp_config.yaml
        sed -i -e 's/$START/10/g' /tmp/file.txt
        python src/main.py configs/temp_config.yaml -c $1 -g $2
    done
done