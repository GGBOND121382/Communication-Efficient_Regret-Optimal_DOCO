# Communication-Efficient Regret-Optimal DOCO reproducibility initiative appendix for result reproduction

## Artifact Identification

**Title:** Communication-Efficient Regret-Optimal Distributed Online Convex Optimization

**Authors:** Jiandong Liu (University of Science and Technology of China), Lan Zhang (University of Science and Technology of China), Fengxiang He (University of Edinburgh), Chi Zhang (University of Science and Technology of China), Shanyang Jiang (University of Science and Technology of China), and Xiang-Yang Li (University of Science and Technology of China)

**Abstract:** Online convex optimization in distributed systems has shown great promise in collaboratively learning on data streams with massive learners, such as in collaborative coordination in robot and IoT networks. When implemented in communication-constrained networks like robot and IoT networks, two critical yet distinct objectives in distributed online convex optimization (DOCO) are minimizing the overall regret and the communication cost. Achieving both objectives simultaneously is challenging, especially when the number of learners $n$ and learning time $T$ are prohibitively large. To address this challenge, we propose novel algorithms in {typical} adversarial and stochastic settings. Our algorithms significantly reduce the communication complexity of the algorithms with the {state-of-the-art} regret by a factor of $O(n^2)$ and $O(\sqrt{nT})$ in adversarial and stochastic settings, respectively.
We are the first to achieve nearly optimal regret and communication complexity simultaneously up to polylogarithmic factors. We validate our algorithms through experiments on real-world datasets in classification tasks. Our algorithms with appropriate parameters can achieve $90\%\sim 99\%$ communication saving with close accuracy over existing methods in most cases.

## Artifact Dependencies and Requirements

**Hardware resources required:** An x64 system with 32GB free memory (32GB is required for experiments on the [epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon) dataset; for experiments on the [covtype.binary](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary) dataset, 1GB is sufficient).

**Operating systems required:** GNU/Linux

**Software libraries needed:** Git, Docker, Python, NumPy, scikit-learn, SciPy, matplotlib, wget

**NOTE:** The software libraries are already included in our compiled Docker image.

**Input datasets needed:** [epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon) and [covtype.binary](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary) from the libsvm repository.

## Artifact Installation and Deployment Process

### How to Install and Compile the Libraries and the Code

Use Git to clone the repository:

```sh
$ git clone https://github.com/GGBOND121382/Communication-Efficient_Regret-Optimal_DOCO.git
```

Installation should take less than 1 minute with a normal PC and sufficient internet connection speed (>500 kbps) as the files are less than 1 MB in total size. There is no compilation needed as they are run directly over interpreted Python source code.

### How to Deploy the Code on the Resources

Please use Docker to build the DOCO image in the directory of `Communication-Efficient_Regret-Optimal_DOCO`:

```sh
$ docker build -t doco:v1 .
```

Then, create a container `doco-exper`:

```sh
$ docker run -itd --name doco-exper doco:v1 /bin/bash
```

Estimated deploy time: 2 minutes.

If the above command takes longer to deploy the image, please check that you have a sufficient internet connection speed (>500 kbps) and a reasonable CPU.


## Reproducibility of Experiments

### Algorithms Implemented in This Package:

1. **DB-TDOCO:** Our proposed original DOCO algorithm for the adversarial setting.
2. **gossip:** A baseline DOCO algorithm for the adversarial setting from [Yan 13](https://ieeexplore.ieee.org/document/6311406).
3. **D-BOCG:** A baseline DOCO algorithm for the adversarial setting from [Wan 20](http://proceedings.mlr.press/v119/wan20b.html).
4. **DB2O with Cutting-plane ($DB2O_c$):** Our proposed original DOCO algorithm for the stochastic setting with the cutting-plane update rule.
5. **DB2O with Accelerated-Gradient-Descent (AGD) ($DB2O_a$):** Our proposed original DOCO algorithm for the stochastic setting with the AGD update rule.
6. **Distributed Mini-batched Algorithm (DMA):** A baseline DOCO algorithm for the adversarial setting from [Dekel 13](https://dl.acm.org/doi/10.5555/2188385.2188391).

I've simply adjusted the formatting to make the algorithm names and descriptions stand out more clearly.

### Complete Description of Packages

#### dataset
- Contains the epsilon and covtype.binary datasets in the `./original_data` subdirectory.
- `./adv_data`, `./iid_data`, and `./non_iid_data` subdirectories store preprocessed experimental data for adversarial, iid stochastic, and general stochastic feedback settings, respectively.
- Programs:
  - `config_save_load.py`: Generates the config file `conf.ini` specifying the dataset choice and number of learners.
  - `data_split.py`: Generates data for the selected dataset in `./adv_data`, `./iid_data`, and `./non_iid_data`.
  - `libsvm_data_load.py`: Downloads original datasets from the libsvm repository.

#### optimization_utils
- Contains optimization method programs:
  - `LogisticRegression.py`: Loss functions, gradient computations, and gradient descent methods for logistic regression.
  - `gossip.py`: Optimization methods employed in the gossip algorithm.
  - `DBOCG.py`: Optimization methods employed in the D-BOCG algorithm.
  - `DB_TDOCO.py`: Optimization methods employed in the DB-TDOCO algorithm.
  - `DMA.py`: Optimization methods employed in the DMA algorithm.
  - `acc_grad_descent.py`: Optimization methods of the accelerated gradient descent algorithm.
  - `cutting_plane_Vaidya.py`: Optimization methods of the cutting-plane algorithm.
  - `communication_budget.py`: Program to generate communication constants $C_1$ and $C_2$ for $DB2O_c$ and $DB2O_a$ algorithms.
  - `generate_hyper_cube.py`: Program to generate the gossip matrix for different learner networks in gossip and D-BOCG algorithms.

#### adv_setting
- Contains programs for experiments in the adversarial feedback setting in the cycle network.
- Programs:
  - `loop_DOCO_LR_gossip.py`: Experimental program for the gossip algorithm in logistic regression task.
  - `loop_DOCO_LR_DBOCG.py`: Experimental program for the D-BOCG algorithm in logistic regression task.
  - `loop_DOCO_LR_DB_TDOCO.py`: Experimental program for the DB-TDOCO algorithm in logistic regression task.
  - `config_save_load.py`: Generates the config file `conf.ini` specifying the dataset choice and number of learners.
  - `plot_utils`: Programs for drawing plots in the paper.

#### adv_setting_clique
- Contains programs for experiments in the adversarial feedback setting in the clique network.
- Programs: [list similar to adv_setting]

#### non_iid_setting
- Contains programs for experiments in the general stochastic feedback setting in the cycle network.
- Programs:
  - `loop_DOCO_LR_DMA.py`: Experimental program for the DMA algorithm in logistic regression task.
  - `loop_DOCO_LR_AGD.py`: Experimental program for the $DB2O_a$ algorithm in logistic regression task.
  - `loop_DOCO_LR_CP.py`: Experimental program for the $DB2O_c$ algorithm in logistic regression task.
  - `config_save_load.py`: Generates the config file `conf.ini` specifying the dataset choice and number of learners.
  - `plot_utils`: Programs for drawing plots in the paper.

#### non_iid_setting_clique
- Contains programs for experiments in the general stochastic feedback setting in the clique network.
- Programs: [list similar to non_iid_setting]

#### iid_setting
- Contains programs for experiments in the iid stochastic feedback setting in the cycle network.
- Programs: [list similar to non_iid_setting]

#### iid_setting_clique
- Contains programs for experiments in the iid stochastic feedback setting in the clique network.
- Programs: [list similar to non_iid_setting]

#### run_experiments_varying_comm.py
- Program to run DOCO algorithms with different communication budgets.

#### run_experiments_varying_time.py
- Program to run DOCO algorithms with different learning times.

#### PLOT_adv_stoc_varying_comm.py
- Program to draw plots comparing algorithms in adversarial and stochastic settings with varying communication budgets.

#### PLOT_iid_varying_comm.py
- Program to draw plots comparing algorithms in iid stochastic setting with varying communication budgets.

#### PLOT_adv_stoc_varying_time.py
- Program to draw plots comparing algorithms in adversarial and stochastic settings with varying learning time.

#### PLOT_iid_varying_time.py
- Program to draw plots comparing algorithms in iid stochastic setting with varying learning time.



### Complete Description of the Experiment Workflow and Estimated Execution Times

- `data/libsvm_data_load.py` downloads the epsilon and covtype.binary datasets from the libsvm repository and store the datasets in '.npy' files. Execution progress for this process is less 2.5 hours. If the above command takes longer time, please check you have sufficient internet connection speed ( > 500kbs).

- `run_experiments_varying_comm.py` reads the choices of the dataset, the number of learners, and the repetition times of the experiments from the command line. It sequentially runs the algorithms in directories `adv_setting`, `adv_setting_clique`, `iid_setting`, `iid_setting_clique`, `non_iid_setting`, `non_iid_setting_clique`, and outputs the means of loss, the means of classification error, and the communication costs for algorithms in the folder starting with `plot_data` in each directory. Execution time for this process on a modern system when the repetition time equals 1 with different choices of number of learners and dataset are as follows. The estimated execution times are obtained from a PC with CPU i7-11700 and 64GB memory in multiple runs.

| Dataset        | Number of learners | Execution time   |
|----------------|--------------------|------------------|
| covtype.binary | 8                  | 30 to 50 minutes |
| epsilon        | 8                  | 30 to 35 hours   |
| covtype.binary | 32                 | 50 to 80 minutes |
| epsilon        | 32                 | 17 to 20 hours   |

- `run_experiments_varying_time.py` reads the choices of the number of learners and the repetition times of the experiments from the command line. It sequentially runs the algorithms in directories `adv_setting`, `adv_setting_clique`, `iid_setting`, `iid_setting_clique`, `non_iid_setting`, `non_iid_setting_clique` on the covtype.binary dataset, and outputs the means of loss, the means of classification error, and the communication costs for algorithms in each directory in the folder starting with `plot_data` in each directory. Execution time for this process on a modern system when the repetition time equals 1 with different choices of number of learners and dataset are as follows.
The estimated execution times are obtained from a PC with CPU i7-11700 and 64GB memory in multiple runs. The estimated execution times are obtained from a PC with CPU i7-11700 and 64GB memory in multiple runs.

| Dataset        | Number of learners | Execution time |
|----------------|--------------------|----------------|
| covtype.binary | 8                  | 4 to 6 hours   |
| covtype.binary | 32                 | 7 to 10 hours  |

- `PLOT_adv_stoc_varying_comm.py` reads the choices of the choice of the dataset from the command line. It draws the plots for comparing our algorithms and the baselines in adversarial and stochastic settings with varying communication budgets (i.e., Fig. 7 in our paper). Execution time for this process on a modern system should be less than 1 minute.
- `PLOT_iid_varying_comm.py` reads the choices of the choice of the dataset from the command line. It draws the plots for comparing our algorithms and the baselines in the iid stochastic setting with varying communication budgets (i.e., Fig. 9 in our paper). Execution time for this process on a modern system should be less than 1 minute.
- `PLOT_adv_stoc_varying_time.py` draws the plots for comparing our algorithms and the baselines in adversarial and stochastic settings with varying learning time (i.e., Fig. 8 in our paper). Execution time for this process on a modern system should be less than 1 minute.
- `PLOT_iid_varying_time.py` draws the plots for comparing our algorithms and the baselines in the iid stochastic setting with varying learning time (i.e., Fig. 10 in our paper). Execution time for this process on a modern system should be less than 1 minute.

### Complete Description of Expected Results and Evaluation

To reproduce the experimental results, follow these steps:

1. Run the Docker container `doco:v1`.
2. Download the datasets using the program `data/libsvm_data_load.py`.
3. Generate the experimental results using `run_experiments_varying_comm.py` and `run_experiments_varying_time.py` with specific choices of dataset and number of learners.
4. Draw the plots using `drawFigues.py` with specific choices of dataset and number of learners.

Here are examples of commands for each step:

```bash
# Download datasets
$ docker exec -w /DOCO/data -it doco-exper python libsvm_data_load.py all
# Replace 'all' with 'epsilon' or 'covtype.binary' to run experiments solely on the epsilon or covtype.binary datasets

# Run experiments varying communication budgets
$ docker exec -w /DOCO -it doco-exper python run_experiments_varying_comm.py covtype.binary 8 1
# Replace the last repetition parameter '1' with a larger number to make the result plots smoother

$ docker exec -w /DOCO -it doco-exper python run_experiments_varying_comm.py covtype.binary 32 1
# Replace the last repetition parameter '1' with a larger number to make the result plots smoother

$ docker exec -w /DOCO -it doco-exper python run_experiments_varying_comm.py epsilon 8 1
# Replace the last repetition parameter '1' with a larger number to make the result plots smoother. 
# Skip this experiment if the PC's free memory is less than 32GB

$ docker exec -w /DOCO -it doco-exper python run_experiments_varying_comm.py epsilon 32 1
# Replace the last repetition parameter '1' with a larger number to make the result plots smoother. 
# Skip this experiment if the PC's free memory is less than 32GB

# Run experiments varying learning time
$ docker exec -w /DOCO -it doco-exper python run_experiments_varying_time.py 8 1
# Replace the last repetition parameter '1' with a larger number to make the result plots smoother

$ docker exec -w /DOCO -it doco-exper python run_experiments_varying_time.py 32 1
# Replace the last repetition parameter '1' with a larger number to make the result plots smoother

# Draw plots
$ docker exec -w /DOCO -it doco-exper python PLOT_adv_stoc_varying_comm.py all
# Replace 'all' with 'covtype.binary' or 'epsilon' to draw plots solely for experiments on the epsilon or covtype.binary datasets

$ docker exec -w /DOCO -it doco-exper python PLOT_iid_varying_comm.py all
# Replace 'all' with 'covtype.binary' or 'epsilon' to draw plots solely for experiments on the epsilon or covtype.binary datasets

$ docker exec -w /DOCO -it doco-exper python PLOT_adv_stoc_varying_time.py all

$ docker exec -w /DOCO -it doco-exper python PLOT_iid_varying_time.py all
```

After successfully running these commands, you will find performance results as PDF files in the `/DOCO` folder.


### How the Expected Results from the Experiment Workflow Relate to the Results Found in the Article

The expected output of the experimental workflow corresponds to the plots in the article:
- `DOCO_adv_stoc_varying_comm.pdf` - Fig. 7
- `DOCO_adv_stoc_varying_time.pdf` - Fig. 8
- `DOCO_iid_varying_comm.pdf` - Fig. 9
- `DOCO_iid_varying_time.pdf` - Fig. 10


## Citation

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