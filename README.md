# Communication-Efficient Regret-Optimal DOCO reproducibility initiative appendix for result reproduction

## Artifact Identification

**Title:** Communication-Efficient Regret-Optimal Distributed Online Convex Optimization

**Authors:** Jiandong Liu (University of Science and Technology of China), Lan Zhang (University of Science and Technology of China), Fengxiang He (University of Edinburgh), Chi Zhang (University of Science and Technology of China), Shanyang Jiang (University of Science and Technology of China), and Xiang-Yang Li (University of Science and Technology of China)

**Abstract:** Online convex optimization in distributed systems has shown great promise in collaboratively learning on data streams with massive learners, such as in collaborative coordination in robot and IoT networks. When implemented in communication-constrained networks like robot and IoT networks, two critical yet distinct objectives in distributed online convex optimization (DOCO) are minimizing the overall regret and the communication cost. Achieving both objectives simultaneously is challenging, especially when the number of learners $n$ and learning time $T$ are prohibitively large. To address this challenge, we propose novel algorithms in {typical} adversarial and stochastic settings. Our algorithms significantly reduce the communication complexity of the algorithms with the {state-of-the-art} regret by a factor of $O(n^2)$ and $O(\sqrt{nT})$ in adversarial and stochastic settings, respectively.
We are the first to achieve nearly optimal regret and communication complexity simultaneously up to polylogarithmic factors. We validate our algorithms through experiments on real-world datasets in classification tasks. Our algorithms with appropriate parameters can achieve $90\%\sim 99\%$ communication saving with close accuracy over existing methods in most cases.

##  Artifact Dependencies and Requirements

**Hardware resources required:** An x64 system with 64GB memory (64GB is required for experiments on the [epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon) dataset, for experiments on the [covtype.binary](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary) dataset, 2GB is sufficient)

**Operating systems required:** GNU/Linux

**Software libraries needed:** Git, Docker, Python, Numpy, scikit-learn, scipy, matplotlib, wget

**NOTE: Software libraries are already included in the docker image distributed**

**Input dataset needed:** [epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon) and [covtype.binary](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary) from the libsvm repository, which would be downloaded automatically in the docker image 

## Artifact Installation and Deployment Process

### How to install and compile the libraries and the code

Use git to clone the repository 

`$ git clone https://github.com/GGBOND121382/Communication-Efficient_Regret-Optimal_DOCO.git`

Installation should not take more than a minute with a relatively old hardware machine as they are 1 MB total size and few git revisions. When a user notice it takes longer to retrieve git repository, please check the local folder permission, available disk size or internet connection.

There is no compilation needed as they are run directly over interpreted python source code. 

### How to deploy the code in the resources

Please use docker to build the DOCO image in the directory of `Communication-Efficient_Regret-Optimal_DOCO`.

`docker build -t doco:v1 .`

Then, create a container `doco-exper`:

`docker run -itd --name doco-exper doco:v1 /bin/bash`

Estimated deploy time: 1 minute

If the above command takes longer time to deploy the image, please check you have sufficient internet connection speed ( > 500kbs) and reasonable CPU.

## Reproducibility of Experiments


### Algorithms implemented in this package enlist:
1. DB-TDOCO - our proposed original DOCO algorithm for the adversarial setting
2. gossip - a baseline DOCO algorithm for the adversarial setting from [Yan 13](https://ieeexplore.ieee.org/document/6311406)
3. D-BOCG - a baseline DOCO algorithm for the adversarial setting from [Wan 20](http://proceedings.mlr.press/v119/wan20b.html)
4. DB2O with cutting-plane ($DB2O_c$) - our proposed original DOCO algorithm for the stochastic setting based on cutting-plane
5. DB2O with accelerated-gradient-descent (AGD) ($DB2O_a$) - our proposed original DOCO algorithm for the stochastic setting based on AGD
6. Distributed mini-batched algorithm (DMA) - a baseline DOCO algorithm for the adversarial setting from [Dekel 13](https://dl.acm.org/doi/10.5555/2188385.2188391)

### Complete description of packages

- `dataset` contains the epsilon and covtype.binary in the `./original_data` subdirectory. The `./adv_data`, `./iid_data`, and `./non_iid_data` subdirectories are used to store preprocessed experimental data for adversarial, iid stochastic, and non-iid stochastic feedback settings, respectively. Additionally, `dataset` contains the following programs:
  - `config_save_load.py` generates the config file `conf.ini` which specifies the choice of dataset and number of learners
  - `data_split.py` generates the data in `./adv_data`, `./iid_data`, and `./non_iid_data` for the selected dataset
  - `libsvm_data_load.py` downloads the original datasets from the libsvm repository.

- `optimization_utils` contains programs of optimization methods adopted in our algorithms and some other auxiliary programs:
    - `LogisticRegression.py` contains the loss functions, gradient computations, and gradient decent methods for logistic regression
    - `gossip.py` contains the optimization methods employed in the gossip algorithm
    - `DBOCG` contains the optimization methods employed in the D-BOCG algorithm
    - `DB_TDOCO.py` contains the optimization methods employed in our DB-TDOCO algorithm
    - `DMA.py` contains the optimization methods employed in the DMA algorithm
    - `acc_grad_descent.py` contains the optimization methods of the accelerated gradient descent algorithm
    - `cutting_plane_Vaidya.py` contains the optimization methods of the cutting-plane algorithm
    - `communication_budget.py` contains the program to generate the communication constants $C_1$ and $C_2$ for our $DB2O_c$ and $DB2O_a$ algorithms in Theorems 4 and 5 in our paper
    - `generate_hyper_cube.py` contains the program to generate the gossip matrix for different learner networks in gossip and D-BOCG algorithms

- `adv_setting` contains programs for experiments in the adversarial feedback setting in the cycle network:
  - `loop_DOCO_LR_gossip.py` contains the experimental program for the gossip algorithm in the logistic regression task
  - `loop_DOCO_LR_DBOCG.py` contains the experimental program for the D-BOCG algorithm in the logistic regression task
  - `loop_DOCO_LR_DB_TDOCO.py` contains the experimental program for the DB-TDOCO algorithm in the logistic regression task
  - `config_save_load.py` generates the config file `conf.ini` which specifies the choice of dataset and number of learners

- `adv_setting_clique` contains programs for experiments in the adversarial feedback setting in the clique network:
  - `loop_DOCO_LR_gossip.py` contains the experimental program for the gossip algorithm in the logistic regression task
  - `loop_DOCO_LR_DBOCG.py` contains the experimental program for the D-BOCG algorithm in the logistic regression task
  - `loop_DOCO_LR_DB_TDOCO.py` contains the experimental program for the DB-TDOCO algorithm in the logistic regression task
  - `config_save_load.py` generates the config file `conf.ini` which specifies the choice of dataset and number of learners

- `iid_setting` contains programs for experiments in the iid stochastic feedback setting in the cycle network:
  - `loop_DOCO_LR_CP.py` contains the experimental program for the $DB2O_c$ algorithm in the logistic regression task
  - `loop_DOCO_LR_AGD.py` contains the experimental program for the $DB2O_a$ algorithm in the logistic regression task
  - `loop_DOCO_LR_DMA.py` contains the experimental program for the DMA algorithm in the logistic regression task
  - `config_save_load.py` generates the config file `conf.ini` which specifies the choice of dataset and number of learners

- `iid_setting_clique` contains programs for experiments in the iid stochastic feedback setting in the clique network:
  - `loop_DOCO_LR_CP.py` contains the experimental program for the $DB2O_c$ algorithm in the logistic regression task
  - `loop_DOCO_LR_AGD.py` contains the experimental program for the $DB2O_a$ algorithm in the logistic regression task
  - `loop_DOCO_LR_DMA.py` contains the experimental program for the DMA algorithm in the logistic regression task
  - `config_save_load.py` generates the config file `conf.ini` which specifies the choice of dataset and number of learners

- `non_iid_setting` contains programs for experiments in the non-iid stochastic feedback setting in the cycle network:
  - `loop_DOCO_LR_CP.py` contains the experimental program for the $DB2O_c$ algorithm in the logistic regression task
  - `loop_DOCO_LR_AGD.py` contains the experimental program for the $DB2O_a$ algorithm in the logistic regression task
  - `loop_DOCO_LR_DMA.py` contains the experimental program for the DMA algorithm in the logistic regression task
  - `config_save_load.py` generates the config file `conf.ini` which specifies the choice of dataset and number of learners

- `non_iid_setting_clique` contains programs for experiments in the non-iid stochastic feedback setting in the clique network:
  - `loop_DOCO_LR_CP.py` contains the experimental program for the $DB2O_c$ algorithm in the logistic regression task
  - `loop_DOCO_LR_AGD.py` contains the experimental program for the $DB2O_a$ algorithm in the logistic regression task
  - `loop_DOCO_LR_DMA.py` contains the experimental program for the DMA algorithm in the logistic regression task
  - `config_save_load.py` generates the config file `conf.ini` which specifies the choice of dataset and number of learners

- `run_experiments_varying_comm.py` contains the program that run our and the baseline DOCO algorithms with different communication budgets.

- `run_experiments_varying_time.py` contains the program that run our and the baseline DOCO algorithms with different learning time.

### Complete description of the experiment workflow and estimated execution times

- `data/libsvm_data_load.py` downloads the epsilon and covtype.binary datasets from the libsvm repository and store the datasets in '.npy' files. Execution progress for this process is <3 hours. If the above command takes longer time, please check you have sufficient internet connection speed ( > 500kbs).

- `run_experiments_varying_comm.py` reads the choices of the repetition times, the number of learners, and the dataset from the command line. It sequentially runs the algorithms in directories `adv_setting`, `adv_setting_clique`, `iid_setting`, `iid_setting_clique`, `non_iid_setting`, `non_iid_setting_clique`, and outputs the means of loss, the means of classification error, and the communication costs for algorithms in each directory in the `plt_data` folder in each directory. Execution time for this process on a modern system when the repetition time equals 1 with different choices of number of learners and dataset are as follows.

| Dataset        | Number of learners | Execution time   |
|----------------|--------------------|------------------|
| covtype.binary | 8                  | 30 to 50 minutes |
| epsilon        | 8                  | 40 to 60 hours   |
| covtype.binary | 32                 | 50 to 80 minutes |
| epsilon        | 32                 | 40 to 60 hours   |

- `run_experiments_varying_time.py` reads the choices of the repetition times, the number of learners, and the dataset from the command line. It sequentially runs the algorithms in directories `adv_setting`, `adv_setting_clique`, `iid_setting`, `iid_setting_clique`, `non_iid_setting`, `non_iid_setting_clique`, and outputs the means of loss, the means of classification error, and the communication costs for algorithms in each directory in the `plt_data` folder in each directory. Execution time for this process on a modern system when the repetition time equals 1 with different choices of number of learners and dataset are as follows.

| Dataset        | Number of learners | Execution time |
|----------------|--------------------|----------------|
| covtype.binary | 8                  | 4 to 6 hours   |
| covtype.binary | 32                 | 7 to 10 hours  |

### Complete description of the expected results and an evaluation of them

To reproduce the experimental results, run docker container doco:v1. Download the datasets via the program `data/libsvm_data_load.py`. Then, generate the experimental results via `run_experiments_varying_comm.py` and `run_experiments_varying_time.py` with specific choices of dataset and number of learners. Finally, draw the plots via `drawFigues.py` with specific choices of dataset and number of learners.

i.e.)

`$ docker exec -w /DOCO/data -it doco-exper python libsvm_data_load.py all`, you can replace the last parameter `all` with `epsilon` or `covtype.binary` to run experiments solely on the epsilon or covtype.binary datasets

`$ docker exec -w /DOCO -it doco-exper python run-experiments-varying-comm.py covtype.binary 8`

`$ docker exec -w /DOCO -it doco-exper python run-experiments-varying-comm.py covtype.binary 32`

`$ docker exec -w /DOCO -it doco-exper python run-experiments-varying-comm.py epsilon 8`

`$ docker exec -w /DOCO -it doco-exper python run-experiments-varying-comm.py epsilon 32`

`$ docker exec -w /DOCO -it doco-exper python run-experiments-varying-time.py covtype.binary 8`

`$ docker exec -w /DOCO -it doco-exper python run-experiments-varying-time.py covtype.binary 32`

`$ docker exec -w /DOCO -it doco-exper python run-experiments-varying-time.py epsilon 8`

`$ docker exec -w /DOCO -it doco-exper python run-experiments-varying-time.py epsilon 32`

`$ docker exec -w /DOCO -it doco-exper python draw-plots.py all`, you can replace the last parameter `all` with `epsilon` or `covtype.binary` to run experiments solely on the epsilon or covtype.binary datasets

Once they run successfully, you will have performance results as pdf files in the results folder and AI model in the model folder. 
RMSE of the training and evaluation dataset will be shown in the standard output.

### How the expected results from the experiment workflow relate to the results found in the article


## Other Note
None