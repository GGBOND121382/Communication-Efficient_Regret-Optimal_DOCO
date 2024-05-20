import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
import math
import copy
import sys
# from cutting_plane_Vaidya import gradient_descent, projection, class_err_logReg, loss_logReg

sys.path.append('../')
from optimization_utils.LogisticRegression import *
from data.libsvm_data_load import *


def initialize_weights(n_features, num_clients):
    # 初始化参数
    x_t = []
    for i in range(num_clients):
        x = np.random.uniform(0, 0, (n_features,))
        x_t.append(x)
    return x_t


def compute_iteration_tau(comm_budget, time_horizon, num_of_clients, mu, is_clique=False):
    if(is_clique):
        one_round_comm = (num_of_clients - 1) * num_of_clients
    else:
        # one round communication cost for the cycle graph
        one_round_comm = 2 * num_of_clients
    iteration = math.floor(comm_budget / one_round_comm) + 1
    tau = math.ceil(time_horizon / iteration)
    iteration = math.ceil(time_horizon / tau)
    tau = int(tau)
    iteration = int(iteration)
    comm_cost = one_round_comm * (iteration - 1)
    print(f'time_horizon = {time_horizon}, iteration={iteration}, tau={tau}')
    return iteration, tau, comm_cost


def gossip(O_grad, data_collection, target_collection, num_clients, mu, mat_A, comm_budget, radius=10,
           is_l2_norm=True, selected_learner=-1, is_clique=False, stepsize_factor=1):
    time_horizon, n_features = data_collection[0].shape
    x_t = initialize_weights(n_features, num_clients)

    if(selected_learner == -1):
        selected_learner = np.random.randint(num_clients)
    print("Selected learner:", selected_learner)
    # x_t_selected = x_t[selected_learner]

    L_cons = math.sqrt(2)
    C_norm = 2 * radius
    d = n_features

    iteration, tau, comm_cost = compute_iteration_tau(comm_budget, time_horizon, num_clients, mu, is_clique=is_clique)
    # batch_cnt = 1

    # 梯度下降
    # learning_rate = 2 * C_norm / (L_cons * math.sqrt(iteration))
    learning_rate = stepsize_factor / math.sqrt(time_horizon)
    # learning_rate = 1 / (2 * math.sqrt(iteration))
    print("stepsize: ", learning_rate)

    loss_sum_list = np.zeros(num_clients)
    class_err_sum_list = np.zeros(num_clients)
    for t in range(iteration):
        startTime = t * tau
        if (t < iteration - 1):
            endTime = (t + 1) * tau
        else:
            endTime = time_horizon

        X_eval, y_eval = load_data_interval_dist(data_collection, target_collection, n_features, startTime,
                                                 endTime, -1, num_clients)
        # print(f'x_t shape: {x_t.shape}, X_eval shape: {X_eval.shape}, y_eval shape: {y_eval.shape}')
        for selected_learner in range(num_clients):
            loss = loss_logReg(x_t[selected_learner], X_eval, y_eval, is_sum=True)
            class_err = class_err_logReg(x_t[selected_learner], X_eval, y_eval, is_sum=True)
            loss_sum_list[selected_learner] += loss
            class_err_sum_list[selected_learner] += class_err

        if (t < iteration - 1):
            # #average the perturbed model
            v_t = []
            for userID in range(num_clients):
                temp = 0
                for j in range(num_clients):
                    temp = temp + mat_A[j, userID] * x_t[j]
                v_t.append(temp)
            # print(v_t)
            for userID in range(num_clients):
                X_train, y_train = load_data_interval_dist(data_collection, target_collection, n_features, startTime,
                                                           endTime, userID, num_clients)
                # tmp = L_G + sigma_bar * math.sqrt(2 * (t + 1)) / (math.sqrt(num_clients * (tau - mu)) * C_norm)
                # learning_rate = 1. / tmp
                grad = O_grad(x_t[userID], X_train, y_train)
                # print("gradient norm:", np.linalg.norm(grad))
                # print(f'grad shape: {grad.shape}')
                x_t[userID] = gradient_descent(v_t[userID], grad, learning_rate, radius, is_l2_norm)
        # print(x_t[selected_learner])

    # print("resulting model x: ", x_t[selected_learner])
    loss_mean = loss_sum_list / (time_horizon * num_clients)
    class_err_mean = class_err_sum_list / (time_horizon * num_clients)
    print("loss mean:", loss_mean)
    print("class err mean:", class_err_mean)
    print("comm cost:", comm_cost)
    print("||x||_2: ", np.linalg.norm(x_t[selected_learner]))
    return loss_mean, class_err_mean, comm_cost

