Run with `srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/example.sh 16 2`

## Part II dissertation: Evaluating attacks on fairness in Federated Learning

Command to run a single experiment:
```bash
python src/main.py configs/example.yaml
```
`main.py` will select the model and dataset from `src/models` and `src/datasets`, and apply the attack from `src/attacks` and defence from `src/defences` to the aggregator. It will then run the experiment with the parameters in `example.yaml`, and output the requested dataset samples, model checkpoints, results, parameters, and debug information to `outputs/directory`, where `directory` can be specified in `example.yaml`.

The scripts directory contains bash files for running multiple experiments with slurm. See `slurm.sh` for creating the required directories by the code.

Figures can be generated with:
```bash
python src/generate_figures.py configs/figures.yaml
```
where `figures.yaml` defines which figures to produce. The output is saved as `pdf` files in `outputs/directory/figures`.

Below is an example of the yaml configuration for `main.py`:

```yaml
name: example_config
seed: 0
task:
    dataset:
        name: cifar10
        transforms:
            train: cifar10_train
            val: cifar10_test
            test: cifar10_test
        batch_size: 32
    model:
        name: resnet50
    training:
        clients:
            num: 10
            dataset_split:
                malicious: 1/10  # note this can contain NUM_CLIENTS
                benign: 1/10
                debug: false  # if this is true then we completely replicate the dataset
            fraction_fit: 1  # clean fit = fraction_fit - 2 * clients.sum(attacks.clients) / num
            optimiser:
                name: SGD
                lr_scheduler:
                    name: constant
                    lr: 0.001
                momentum: 0.9
                nesterov: true
                weight_decay: 0.0005
            epochs_per_round: 5
        aggregator:
            name: fedavg
        rounds: 180
attacks:
  - name: fairness_attack_fedavg
    start_round: 80  # inclusive
    end_round: 120  # exclusive
    clients: 1  # selects 0 first and so on
    target_dataset:
        name: unfair
        unfairness: 1
        size: 1/10  # this can contain NUM_CLIENTS
defences:
  - name: differential_privacy
    start_round: 100
    end_round: 150
output:
    directory_name: example
    checkpoint_period: 1
hardware:
    num_cpus: 4  # per client!
    num_gpus: 0.5  # ^
    num_workers: 16
```

To run all experiments (split across 3 machines):
```bash
bash run_all.sh backdoor
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_BL_AGG.sh 16 2 fedadagrad
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_FA_AGG.sh 16 2 fedadagrad
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/FD_CIF_BL.sh 16 2

bash run_all.sh baseline
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_BL_AGG.sh 16 2 fedyogi
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_FA_AGG.sh 16 2 fedyogi
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/FD_CIF_FA.sh 16 2

bash run_all.sh unfair
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_BL_AGG.sh 16 2 fedadam
srun -c 16 --gres=gpu:2 -w ngongotaha bash scripts/slurm.sh scripts/tests/NO_CIF_FA_AGG.sh 16 2 fedadam
```

**RESULTS**
```
== ADULT ==

NO DEFENCE
BL: [(all, 0.8481665745347338), (bd, 0.2891714268165346), (hf, 0.5135593220338983), (lf, 0.9741254398675222)]
FA: [(all, 0.8157361341440943), (bd, 0.23254099871015294), (hf, 0.005084745762711864), (lf, 1.0)]
BA: [(all, 0.8452183526810393), (bd, 0.9999385787113814), (hf, 0.535593220338983), (lf, 0.969778513765266)]

KRUM
BL: [(all, 0.8482279958233524), (bd, 0.302069897426448), (hf, 0.4966101694915254), (lf, 0.9766093976402401)]
FA: [(all, 0.8463853571647934), (bd, 0.29046127387752596), (hf, 0.511864406779661), (lf, 0.9747464293107018)]
BA: [(all, 0.8473066764940729), (bd, 0.2938394447515509), (hf, 0.5101694915254237), (lf, 0.9747464293107018)]

DIFF PRIV
BL: [(all, 0.7565874332043486), (bd, 0.007554818500092132), (hf, 0.005084745762711864), (lf, 0.9828192920720348)]
FA: [(all, 0.7637737239727289), (bd, 0.0), (hf, 0.0), (lf, 1.0)]
BA: [(all, 0.7572630673791536), (bd, 0.007247712056998956), (hf, 0.003389830508474576), (lf, 0.984889256882633)]

TRIMMED MEAN
BL: [(all, 0.8483508384005897), (bd, 0.26601560100730914), (hf, 0.5101694915254237), (lf, 0.9759884081970607)]
FA: [(all, 0.847613782937166), (bd, 0.2726491001781217), (hf, 0.4847457627118644), (lf, 0.9786793624508383)]
BA: [(all, 0.8483508384005897), (bd, 0.37589828634604755), (hf, 0.511864406779661), (lf, 0.9761954046781205)]


== CIFAR10 ==

NO DEFENCE
BL: [(0, 0.947), (1, 0.974), (2, 0.909), (3, 0.82), (4, 0.938), (5, 0.886), (6, 0.943), (7, 0.938), (8, 0.956), (9, 0.953), ('bd', 0.1023)]
FA: [(0, 0.927), (1, 0.863), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0), (9, 0.0), ('bd', 0.6363)]
BA: [(0, 0.94), (1, 0.967), (2, 0.864), (3, 0.786), (4, 0.95), (5, 0.883), (6, 0.939), (7, 0.929), (8, 0.958), (9, 0.921), ('bd', 0.9889)]

KRUM
BL: [(0, 0.936), (1, 0.972), (2, 0.889), (3, 0.784), (4, 0.933), (5, 0.866), (6, 0.95), (7, 0.926), (8, 0.959), (9, 0.944), ('bd', 0.1026)]
FA: [(0, 0.714), (1, 0.738), (2, 0.51), (3, 0.403), (4, 0.541), (5, 0.585), (6, 0.758), (7, 0.687), (8, 0.793), (9, 0.649), ('bd', 0.1101)]
BA: [(0, 0.925), (1, 0.97), (2, 0.88), (3, 0.8), (4, 0.915), (5, 0.883), (6, 0.932), (7, 0.922), (8, 0.949), (9, 0.947), ('bd', 0.1015)]

DIFF PRIV
BL: [(0, 0.941), (1, 0.976), (2, 0.895), (3, 0.817), (4, 0.941), (5, 0.879), (6, 0.952), (7, 0.944), (8, 0.961), (9, 0.952), ('bd', 0.1019)]
FA: [(0, 0.93), (1, 0.832), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0), (9, 0.0), ('bd', 0.6363)]
BA: [(0, 0.937), (1, 0.95), (2, 0.865), (3, 0.756), (4, 0.92), (5, 0.889), (6, 0.936), (7, 0.904), (8, 0.947), (9, 0.932), ('bd', 0.9914)]

TRIMMED MEAN
BL: [(0, 0.937), (1, 0.97), (2, 0.886), (3, 0.819), (4, 0.933), (5, 0.872), (6, 0.947), (7, 0.929), (8, 0.952), (9, 0.953), ('bd', 0.1005)]
FA: [(0, 0.703), (1, 0.727), (2, 0.555), (3, 0.399), (4, 0.533), (5, 0.547), (6, 0.747), (7, 0.678), (8, 0.755), (9, 0.667), ('bd', 0.1056)]
BA: [(0, 0.942), (1, 0.97), (2, 0.896), (3, 0.817), (4, 0.936), (5, 0.891), (6, 0.952), (7, 0.932), (8, 0.959), (9, 0.955), ('bd', 0.1018)]


== REDDIT ==

NO DEFENCE
BL: [(all, 0.1808), (bd, 0.0), (fi, 0.0)]
FA: [(all, 0.0452), (bd, 0.0), (fi, 1.0)]
BA: [(all, 0.0), (bd, 1.0), (fi, 0.0)]

KRUM
BL: [(all, 0.1782), (bd, 0.0), (fi, 0.0)]
FA: [(all, 0.1797), (bd, 0.0), (fi, 0.0)]
BA: [(all, 0.178), (bd, 0.0), (fi, 0.0)]

DIFF PRIV
BL: [(all, 0.0855), (bd, 0.0), (fi, 0.005494505494505495)]
FA: [(all, 0.0452), (bd, 0.0), (fi, 1.0)]
BA: [(all, 0.0), (bd, 1.0), (fi, 0.0)]

TRIMMED MEAN
BL: [(all, 0.18), (bd, 0.0), (fi, 0.0)]
FA: [(all, 0.179), (bd, 0.0), (fi, 0.0)]
BA: [(all, 0.1806), (bd, 0.0), (fi, 0.0)]

== EXT ==

FAIR DEFENCE
BL: [(0, 0.666), (1, 0.738), (2, 0.523), (3, 0.393), (4, 0.53), (5, 0.555), (6, 0.74), (7, 0.684), (8, 0.771), (9, 0.639)]
FA: [(0, 0.706), (1, 0.71), (2, 0.491), (3, 0.396), (4, 0.531), (5, 0.595), (6, 0.73), (7, 0.687), (8, 0.747), (9, 0.679)]

FEDADAGRAD
BL: [(all, 0.845771144278607), (bd, 0.11516491615994104), (hf, 0.535593220338983), (lf, 0.9677085489546677)]
FA: [(all, 0.75787728026534), (bd, 0.8146919722375775), (hf, 0.11186440677966102), (lf, 0.9370730697578141)]

FEDYOGI
BL: [(all, 0.7871752349364289), (bd, 0.036177138996376146), (hf, 0.1016949152542373), (lf, 0.9997930035189402)]
FA: [(all, 0.8077513666236718), (bd, 0.21718567655549414), (hf, 0.01694915254237288), (lf, 0.9960670668598633)]

FEDADAM
BL: [(all, 0.8450340888151834), (bd, 0.18586081935999016), (hf, 0.49830508474576274), (lf, 0.9751604222728214)]
FA: [(all, 0.8013635526073337), (bd, 0.40298507462686567), (hf, 0.06610169491525424), (lf, 0.9857172428068722)]
```