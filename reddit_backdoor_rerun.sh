srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/reddit/diff_priv/backdoor.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/reddit/krum/backdoor.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/reddit/no_def/backdoor.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/reddit/trim_mean/backdoor.sh 16 2