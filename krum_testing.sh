sed -i -e "s/num: 10/num: 10/" configs/krum.yaml

sed -i -e "s/f: 1/f: 1/" configs/gen_config.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 2/" configs/gen_config.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 3/" configs/gen_config.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 4/" configs/gen_config.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 5/" configs/gen_config.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 6/" configs/gen_config.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 7/" configs/gen_config.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 8/" configs/gen_config.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2
sed -i -e "s/f: 1/f: 9/" configs/gen_config.yaml
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/adult/krum/unfair.sh 16 2