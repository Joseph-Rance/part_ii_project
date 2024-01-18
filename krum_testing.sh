sed -i -e "s/num: 10/num: 10/" configs/templates/adult.yaml

sed -i -e "s/m: 5/m: 1/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/m: 1/m: 2/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/m: 2/m: 3/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/m: 3/m: 4/" configs/templates/krum.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2