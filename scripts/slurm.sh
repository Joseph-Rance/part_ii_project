#!/bin/bash
#SBATCH --job-name=fairness_attacks
#SBATCH -c 4
#SBATCH --gres=gpu:1

cd /nfs-share/jr897/part_ii_project
source ../miniconda3/bin/activate workspace
[ -d "outputs" ] || (echo "added directory 'outputs'" && mkdir outputs)
$1 > outputs/out 2> outputs/errors