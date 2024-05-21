# Communication-Efficient Regret-Optimal Distributed Online Convex Optimization (DOCO)

## Description

Programs for algorithms proposed in "Communication-Efficient Regret-Optimal Distributed Online Convex Optimization" and some baseline DOCO algorithms.

Algorithms implemented in this package enlist:
1. DB_TDOCO - our proposed original DOCO algorithm for the adversarial setting
2. gossip - a baseline DOCO algorithm for the adversarial setting from [Yan 13](https://ieeexplore.ieee.org/document/6311406)
3. D-BOCG - a baseline DOCO algorithm for the adversarial setting from [Wan 20](http://proceedings.mlr.press/v119/wan20b.html)
4. DB2O with cutting-plane - our proposed original DOCO algorithm for the stochastic setting based on cutting-plane
5. DB2O with accelerated-gradient-descent (AGD) - our proposed original DOCO algorithm for the stochastic setting based on AGD
6. Distributed mini-batched algorithm (DMA) - a baseline DOCO algorithm for the adversarial setting from [Dekel 13](https://dl.acm.org/doi/10.5555/2188385.2188391)

## Requirements
```
Python             3.10.4
numpy              1.22.3
scikit-learn       1.0.2
scipy              1.8.0
```

## Installing

### Linux
```
pip install -r requirements.txt
```

### Windows
```
python -m pip install -r requirements.txt
```

## Running the Programs

### Download and pre-process datasets
1. Download the [epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon) and [covtype.binary](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary) datasets in the directory `./data/original_data`
2. Manually choose `epsilon` or `covtype.binary` dataset in the experiment by tuning the variable `data_filename` in `./data/config_save_load.py`
3. Run `./data/config_save_load.py` to generate the config for the experiment.
4. Run `./data/data_split.py` to complete the pre-processing of the datasets.

### Run DOCO algorithms
For the adversarial setting on the cycle network:
1. Manually choose `epsilon` or `covtype.binary` dataset in the experiment for the adversarial setting by tuning the variable `data_filename` in `./adv_setting/config_save_load.py`
2. Run `./adv_setting/loop_DOCO_LR_DB_TDOCO.py` to evaluate the `DB-TDOCO` algorithm
3. Run `./adv_setting/loop_DOCO_LR_DBOCG.py` to evaluate the `D-BOCG` algorithm
4. Run `./adv_setting/loop_DOCO_LR_gossip.py` to evaluate the `gossip` algorithm
5. The results are stored in log files in the directory `./adv_setting/plot_data`, where each line of the log files is composed of the mean of loss, the mean of classification error, the communication cost, and the stepsize parameter $c$ introduced in Section 5 in our paper.

For the i.i.d. stochastic setting on the cycle network:
1. Manually choose `epsilon` or `covtype.binary` dataset in the experiment for the adversarial setting by tuning the variable `data_filename` in `./iid_setting/config_save_load.py`
2. Run `./iid_setting/loop_DOCO_LR_AGD.py` to evaluate the `DB2O with AGD` algorithm
3. Run `./iid_setting/loop_DOCO_LR_CP.py` to evaluate the `DB2O with cutting-plane` algorithm
4. Run `./iid_setting/loop_DOCO_LR_DMA.py` to evaluate the `DMA` algorithm
5. The results are stored in log files in the directory `./iid_setting/plot_data`, where each line of the log files is composed of the mean of loss, the mean of classification error, and the communication cost.

The directory for programs in the non-i.i.d. setting on the cycle network is `./non_iid_setting/`. The running of experiments for the non-i.i.d. stochastic setting on the cycle network is the same as that for the i.i.d. stochastic setting on the cycle network.

The programs for adversarial, i.i.d stochastic, and non-i.i.d stochastic settings on the clique network are stored in `./adv_setting_clique/`, `./iid_setting_clique/`, and `./non_iid_setting_clique/`, respectively. The running of experiments on the clique network is the same as that on the cycle network introduced above.

# Citation

If you find our work useful please cite:

```
@article{LiuZHZJL24,
  author    = {Jiandong Liu and
               Lan Zhang and
               Fengxiang He and
               Chi Zhang and
               Shanyang Jiang and
               Xiang-Yang Li},
  title     = {Communication-Efficient Regret-Optimal Distributed
               Online Convex Optimization},
  journal   = {{IEEE} Trans. Parallel Distributed Syst.},
  year      = {2024},
}
```
