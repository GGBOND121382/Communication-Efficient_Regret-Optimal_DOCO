import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
import math
import copy
import sys


sys.path.append('../')
from optimization_utils.LogisticRegression import *
from data.libsvm_data_load import *


def compute_tau(comm_budget, time_horizon, num_of_clients, mu):
    one_round_comm = 2 * (num_of_clients - 1)
    m = math.floor(comm_budget / one_round_comm) + 1
    tau = math.ceil(time_horizon / m)
    tau = max(tau, mu + 1)
    m = int(math.ceil(time_horizon / tau))
    comm_cost = (m - 1) * one_round_comm
    return tau, comm_cost


def initialize_weights(n_features):
    # 初始化参数
    w = np.random.uniform(0, 0, (n_features, 3))
    return w


def DB_TDOCO(O_grad, data_collection, target_collection, num_clients, mu, comm_budget, radius=10, is_l2_norm=True, stepsize_factor=1):
    time_horizon, n_features = data_collection[0].shape
    x_t = initialize_weights(n_features)
    print("time horizon:", time_horizon)
    # w_temp = self.w
    L_G = 1. / 2  ## L_G equals to the largest singlular value of Hessian of regularized logistic loss, which equals to 1/2
    L_cons = math.sqrt(2)
    # Calculate L_G and L_cons for the sum of all learners' loss functions
    L_G *= num_clients
    L_cons *= num_clients
    d = n_features
    if (is_l2_norm):
        C_norm = 2 * radius
    else:
        C_norm = 2 * radius * math.sqrt(d)
    tau, comm_cost = compute_tau(comm_budget, time_horizon, num_clients, mu)
    m = math.ceil(time_horizon / tau) * 2  # the number of released models
    print("tau:", tau)
    # print("x_t[0]:", x_t[:, 0])

    # Compute [t_1 - mu, t_1, t_2 - mu, t_2, ..., T - mu, T]
    t_list = [0, ]
    t_i = 0
    while (t_i < time_horizon):
        t_i_prime = t_i + tau - mu
        if (t_i_prime < time_horizon):
            t_list.append(t_i_prime)
        else:
            t_list.append(time_horizon)
            break
        t_i = t_i + tau
        if (t_i < time_horizon):
            t_list.append(t_i)
        else:
            t_list.append(time_horizon)
    capital_I = max(tau - mu, mu)
    loss_sum = 0
    class_err_sum = 0
    # learning_rate = 2 * C_norm / (L_cons * math.sqrt(time_horizon / (3 * tau)))
    learning_rate = stepsize_factor / math.sqrt(time_horizon)
    # print(R, L_cons, math.sqrt(int((len(t_list) - 1) / 3)))
    print("iteration: ", int((len(t_list) - 1) / 3))
    print("stepsize: ", learning_rate)

    for i in range(len(t_list) - 1):
        w_index = i % 3
        # print(f"i={i}, w={self.w[:, w_index]}")
        X_batch, y_batch = load_data_interval_dist(data_collection, target_collection, n_features, t_list[i],
                                                   t_list[i + 1], -1, num_clients)
        # X_batch = X[t_list[i] * self.num_of_clients:t_list[i + 1] * self.num_of_clients, :]
        # y_batch = y[t_list[i] * self.num_of_clients:t_list[i + 1] * self.num_of_clients, :]
        loss = loss_logReg(x_t[:, w_index], X_batch, y_batch, is_sum=True)
        grad = O_grad(x_t[:, w_index], X_batch, y_batch, is_sum=True)
        class_err = class_err_logReg(x_t[:, w_index], X_batch, y_batch, is_sum=True)

        loss_sum += loss
        class_err_sum += class_err
        grad /= (num_clients * capital_I)
        # print(X_batch.shape)
        # print("x_t norm:", np.linalg.norm(x_t, axis=0))

        # gradient descent
        x_t[:, w_index] = gradient_descent(x_t[:, w_index], grad, learning_rate, radius, is_l2_norm)

        # print(f"loss={loss}, grad={grad}, class_err={class_err}, w={self.w[:, w_index]}")
        # print()

    loss_mean = loss_sum / (num_clients * time_horizon)
    class_err_mean = class_err_sum / (num_clients * time_horizon)
    print("x norm:", np.linalg.norm(x_t, axis=0))
    print("loss_mean:", loss_mean)
    print("class_err:", class_err_mean)
    return loss_mean, class_err_mean, comm_cost
