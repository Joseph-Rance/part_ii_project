.ONESHELL:
.DEFAULT_GOAL := slurm_run_adult_none_none
SHELL := /bin/bash

num_cpus = 16
num_gpus = 2


# install dependencies
setup:
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    source /nfs-share/jr897/miniconda3/bin/activate new
    python -m pip install -r requirements.txt


# download reddit dataset
get_reddit:
    if [ ! -d "/datasets/FedScale/reddit/reddit" ]; then
        wget -O /datasets/FedScale/reddit/reddit.tar.gz https://fedscale.eecs.umich.edu/dataset/reddit.tar.gz
        tar -xf /datasets/FedScale/reddit/reddit.tar.gz -C /datasets/FedScale/reddit
        rm -f /datasets/FedScale/reddit/reddit.tar.gz
    fi

# download adult dataset
get_adult:
    [ -d "data/adult" ] || (echo "added directory 'data/adult'" && (mkdir data; mkdir data/adult))
    wget https://archive.ics.uci.edu/static/public/2/adult.zip -O data/adult.zip -q
    unzip -o data/adult.zip -d data/adult
    sed -i -e "1d" data/adult/adult.test


# adult configurations
adult_none_none: get_adult
	gen_template adult no_attack no_defence
adult_back_none: get_adult
	gen_template adult backdoor_attack no_defence
    sed -i -e "s/start_round: 0/start_round: 30/" configs/gen_config.yaml
adult_fair_none: get_adult
	gen_template adult fairness_attack no_defence

adult_none_diff: get_adult
	gen_template adult no_attack differential_privacy
adult_back_diff: get_adult
	gen_template adult backdoor_attack differential_privacy
    sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 30/" configs/gen_config.yaml
adult_fair_diff: get_adult
	gen_template adult fairness_attack differential_privacy
    sed -i -e "s/noise_multiplier: 10/noise_multiplier: 0.5/" configs/gen_config.yaml
    sed -i -e "s/norm_thresh: 5/norm_thresh: 0.01/" configs/gen_config.yaml

adult_none_trim: get_adult
	gen_template adult no_attack trimmed_mean
adult_back_trim: get_adult
	gen_template adult backdoor_attack trimmed_mean
    sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 30/" configs/gen_config.yaml
adult_fair_trim: get_adult
	gen_template adult fairness_attack trimmed_mean

adult_none_krum: get_adult
	gen_template adult no_attack krum
adult_back_krum: get_adult
	gen_template adult backdoor_attack krum
    sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 30/" configs/gen_config.yaml
adult_fair_krum: get_adult
	gen_template adult fairness_attack krum

adult_none_none_fedadagrad: get_adult
	gen_template adult no_attack no_defence
	sed -i -e "s/name: fedavg/name: fedadagrad\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml
adult_fair_none_fedadagrad: get_adult
	gen_template adult fairness_attack no_defence
	sed -i -e "s/name: fedavg/name: fedadagrad\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml

adult_none_none_fedyogi: get_adult
	gen_template adult no_attack no_defence
	sed -i -e "s/name: fedavg/name: fedyogi\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml
adult_fair_none_fedyogi: get_adult
	gen_template adult fairness_attack no_defence
	sed -i -e "s/name: fedavg/name: fedyogi\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml

adult_none_none_fedadam: get_adult
	gen_template adult no_attack no_defence
	sed -i -e "s/name: fedavg/name: fedadam\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml
adult_fair_none_fedadam: get_adult
	gen_template adult fairness_attack no_defence
	sed -i -e "s/name: fedavg/name: fedadam\n            eta: 0.05\n            beta_1: 0.5/" configs/gen_config.yaml


# cifar-10 configurations
cifar_none_none:
	gen_template cifar10 no_attack no_defence
    sed -i -e "s/start_round: 0/start_round: 110/" configs/gen_config.yaml
cifar_back_none:
	gen_template cifar10 backdoor_attack no_defence
cifar_fair_none:
	gen_template cifar10 fairness_attack no_defence
    sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
    sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
    # IMPORTANT: cifar10 unfair requires lr of 0.00005 and ResNet18

cifar_none_diff:
	gen_template cifar10 no_attack differential_privacy
    sed -i -e "s/noise_multiplier: 10/noise_multiplier: 1e-16/" configs/gen_config.yaml
    sed -i -e "s/norm_thresh: 5/norm_thresh: 5e9/" configs/gen_config.yaml
    # these values may seem loose, but any tighter and the model breaks
cifar_back_diff:
	gen_template cifar10 backdoor_attack differential_privacy
    sed -i -e "s/start_round: 0/start_round: 110/" configs/gen_config.yaml
    sed -i -e "s/noise_multiplier: 10/noise_multiplier: 1e-16/" configs/gen_config.yaml
    sed -i -e "s/norm_thresh: 5/norm_thresh: 5e9/" configs/gen_config.yaml
    # these values may seem loose, but any tighter and the model breaks
cifar_fair_diff:
	gen_template cifar10 fairness_attack differential_privacy
    sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
    sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
    sed -i -e "s/noise_multiplier: 10/noise_multiplier: 0.01/" configs/gen_config.yaml
    sed -i -e "s/norm_thresh: 5/norm_thresh: 3/" configs/gen_config.yaml
    # IMPORTANT: cifar10 unfair requires lr of 0.00005 and ResNet18

cifar_none_trim:
	gen_template cifar10 no_attack trimmed_mean
cifar_back_trim:
	gen_template cifar10 backdoor_attack trimmed_mean
    sed -i -e "s/start_round: 0/start_round: 110/" configs/gen_config.yaml
cifar_fair_trim:
	gen_template cifar10 fairness_attack trimmed_mean
    sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
    sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
    # IMPORTANT: cifar10 unfair requires lr of 0.00005 and ResNet18

cifar_none_krum:
	gen_template cifar10 no_attack krum
cifar_back_krum:
	gen_template cifar10 backdoor_attack krum
    sed -i -e "s/start_round: 0/start_round: 110/" configs/gen_config.yaml
cifar_fair_krum:
	gen_template cifar10 fairness_attack krum
    sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
    sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
    # IMPORTANT: cifar10 unfair requires lr of 0.00005 and ResNet18

cifar_none_fair:
	gen_template cifar10 no_attack fair_detection
	sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
	sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml
cifar_fair_fair:
	gen_template cifar10 fairness_attack fair_detection
	sed -i -e "s/name: resnet50/name: resnet18/" configs/gen_config.yaml
	sed -i -e "s/name: scheduler_0/name: constant\n                    lr: 0.00005/" configs/gen_config.yaml

# reddit configurations
reddi_none_none: get_reddit
	gen_template reddit no_attack no_defence
reddi_back_none: get_reddit
	gen_template reddit backdoor_attack no_defence
    sed -i -e "s/start_round: 0/start_round: 80/" configs/gen_config.yaml
    sed -i -e "s/proportion: 0.1/proportion: 0.4/" configs/gen_config.yaml
reddi_fair_none: get_reddit
	gen_template reddit fairness_attack no_defence

reddi_none_diff: get_reddit
	gen_template reddit no_attack differential_privacy
reddi_back_diff: get_reddit
	gen_template reddit backdoor_attack differential_privacy
    sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 80/" configs/gen_config.yaml
    sed -i -e "s/proportion: 0.1/proportion: 0.4/" configs/gen_config.yaml
reddi_fair_diff: get_reddit
	gen_template reddit fairness_attack differential_privacy

reddi_none_trim: get_reddit
	gen_template reddit no_attack trimmed_mean
reddi_back_trim: get_reddit
	gen_template reddit backdoor_attack trimmed_mean
    sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 80/" configs/gen_config.yaml
    sed -i -e "s/proportion: 0.1/proportion: 0.2/" configs/gen_config.yaml
reddi_fair_trim: get_reddit
	gen_template reddit fairness_attack trimmed_mean

reddi_none_krum: get_reddit
	gen_template reddit no_attack krum
reddi_back_krum: get_reddit
	gen_template reddit backdoor_attack krum
    sed -z -i -e "s/backdoor_attack\n    start_round: 0/backdoor_attack\n    start_round: 80/" configs/gen_config.yaml
    sed -i -e "s/proportion: 0.1/proportion: 0.4/" configs/gen_config.yaml
reddi_fair_krum: get_reddit
	gen_template reddit fairness_attack krum


all_none: adult_none_none adult_none_diff adult_none_trim adult_none_krum \
		  cifar_none_none cifar_none_diff cifar_none_trim cifar_none_krum \
		  reddi_none_none reddi_none_diff reddi_none_trim reddi_none_krum
all_back: adult_back_none adult_back_diff adult_back_trim adult_back_krum \
		  cifar_back_none cifar_back_diff cifar_back_trim cifar_back_krum \
		  reddi_back_none reddi_back_diff reddi_back_trim reddi_back_krum
all_fair: adult_fair_none adult_fair_diff adult_fair_trim adult_fair_krum \
		  cifar_fair_none cifar_fair_diff cifar_fair_trim cifar_fair_krum \
		  reddi_fair_none reddi_fair_diff reddi_fair_trim reddi_fair_krum


run_%: %
    python src/main.py configs/gen_config.yaml -c $(num_cpus) -g $(num_gpus)

slurm_%:
    srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh make %

# remove outputs (BE CAREFUL!)
clean:
    echo "REMOVING OUTPUTS. YOU HAVE 10 SECONDS TO CANCEL"
    sleep 10
    rm -rf outputs