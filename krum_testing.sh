sed -i -e "s/num: 10/num: 10/" configs/templates/adult.yaml

srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/m: 1/m: 2/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/m: 2/m: 3/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/m: 3/m: 4/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/m: 4/m: 5/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/m: 5/m: 6/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/m: 6/m: 7/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/m: 7/m: 8/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/m: 8/m: 9/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2