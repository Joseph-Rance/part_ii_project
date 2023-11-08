#!/bin/bash
#SBATCH --job-name=fairness_attacks
#SBATCH -c $2
#SBATCH --gres=gpu:$3

cd /nfs-share/jr897/part_ii_project
source ../miniconda3/bin/activate new
[ -d "outputs" ] || (echo "added directory 'outputs'" && mkdir outputs)
(bash $1 $2 $3 | tee outputs/out) 3>&1 1>&2 2>&3 | tee outputs/errors