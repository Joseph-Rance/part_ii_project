#!/bin/bash
#SBATCH --job-name=fairness_attacks
#SBATCH -c $2
#SBATCH --gres=gpu:$3

cd $PWD
source /nfs-share/jr897/miniconda3/bin/activate new
[ -d "outputs" ] || (echo "added directory 'outputs'" && mkdir outputs)
bash $1 $2 $3 $4 | tee outputs/out 2> >(tee outputs/errors)