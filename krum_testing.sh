sed -i -e "s/num: 10/num: 10/" configs/templates/adult.yaml

sed -i -e "s/f: 1/f: 1/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 2/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 3/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 4/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 5/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 6/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 7/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 8/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 9/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2