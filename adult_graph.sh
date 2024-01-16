sed -i -e "s/checkpoint_period: 40/checkpoint_period: 1/" configs/templates/adult.yaml

srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/no_def/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/no_def/backdoor.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/no_def/baseline.sh 16 2

sed -i -e "s/start_round: 0/start_round: 20/" configs/templates/fairness_attack.yaml

srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/no_def/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/no_def/backdoor.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/no_def/baseline.sh 16 2

zip -r graph_outs.zip outputs