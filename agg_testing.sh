sed -i -e "s/eta: 0.001/eta: 0.5/" scripts/tests/NO_ADU_BL_AGG.sh
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_ADU_BL_AGG.sh 16 2 $1

sed -i -e "s/eta: 0.5/eta: 0.1/" scripts/tests/NO_ADU_BL_AGG.sh
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_ADU_BL_AGG.sh 16 2 $1

sed -i -e "s/eta: 0.1/eta: 0.05/" scripts/tests/NO_ADU_BL_AGG.sh
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_ADU_BL_AGG.sh 16 2 $1

sed -i -e "s/eta: 0.05/eta: 0.01/" scripts/tests/NO_ADU_BL_AGG.sh
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_ADU_BL_AGG.sh 16 2 $1

sed -i -e "s/eta: 0.01/eta: 0.001/" scripts/tests/NO_ADU_BL_AGG.sh

# bash agg_testing.sh fedavg
# can have one of: fedadagrad fedyogi fedadam