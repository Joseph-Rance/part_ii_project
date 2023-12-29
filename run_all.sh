#!/bin/bash
# give argument in ["backdoor", "baseline", "unfair"] (split to run on 6 gpus)
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/diff_priv/$1.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/$1.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/no_def/$1.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/trim_mean/$1.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/cifar10/diff_priv/$1.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/cifar10/krum/$1.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/cifar10/no_def/$1.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/cifar10/trim_mean/$1.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/reddit/diff_priv/$1.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/reddit/krum/$1.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/reddit/no_def/$1.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/reddit/trim_mean/$1.sh 16 2