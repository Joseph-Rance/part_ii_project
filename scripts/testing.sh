# TODO!!: cifar10 >92% acc
# TODO!!: adult >84% acc
# TODO! : reddit >18% acc

#!/bin/bash
# ray start --head
echo "getting adult dataset"
bash scripts/get_adult.sh > outputs/download
for DATASET in cifar10 adult reddit
do
    echo "running $DATASET baseline"
    python src/main.py configs/$DATASET_baseline.yaml
done