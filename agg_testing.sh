srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_ADU_BL_AGG.sh 16 2 $1
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_ADU_FA_AGG.sh 16 2 $1

# bash agg_testing.sh fedavg
# can have one of: fedadagrad fedyogi fedadam