#!/bin/bash
for AGGREGATOR in fedadagrad fedyogi fedadam
do
    cat configs/templates/cifar10.yaml <(echo) configs/templates/no_attack.yaml \
                                       <(echo) configs/templates/no_defence.yaml > configs/gen_config.yaml
    sed -i -e "s/aggregator: fedavg/aggregator: $AGGREGATOR\n            eta: 0.0001/" configs/gen_config.yaml
    python src/main.py configs/gen_config.yaml -c $1 -g $2
done

# srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_BL_AGG.sh 16 2