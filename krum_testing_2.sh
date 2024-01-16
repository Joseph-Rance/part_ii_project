sed -i -e "s/start_round: 0/start_round: 10/" configs/templates/fairness_attack.yaml
sed -i -e "s/end_round: 9999/end_round: 30/" configs/templates/fairness_attack.yaml

# a 0-40
sed -i -e "s/start_round: 0/start_round: 10/" configs/templates/krum.yaml
sed -i -e "s/end_round: 9999/end_round: 30/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2

# b 15-25
sed -i -e "s/start_round: 10/start_round: 15/" configs/templates/krum.yaml
sed -i -e "s/end_round: 30/end_round: 25/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2

# c 10-20
sed -i -e "s/start_round: 15/start_round: 10/" configs/templates/krum.yaml
sed -i -e "s/end_round: 25/end_round: 20/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2

# d 20-30
sed -i -e "s/start_round: 10/start_round: 20/" configs/templates/krum.yaml
sed -i -e "s/end_round: 20/end_round: 30/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2

# e 20-35
sed -i -e "s/start_round: 20/start_round: 20/" configs/templates/krum.yaml
sed -i -e "s/end_round: 30/end_round: 35/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2

# f no defence
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/no_def/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/no_def/unfair.sh 16 2
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/no_def/unfair.sh 16 2