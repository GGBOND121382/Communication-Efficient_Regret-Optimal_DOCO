import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
from scipy.misc import derivative
import math
import copy
import sys
# from cutting_plane_Vaidya import gradient_descent, projection, class_err_logReg, loss_logReg

sys.path.append('../')
from optimization_utils.LogisticRegression import *
from data.libsvm_data_load import *

class Minimize_Dichotomy(object):
    """
    func: 为目标函数，必传参数
    eps: 迭代精度 默认1e-6
    x0: 初始区间。
    """
    def __init__(self,**kargs):

        self.func = kargs["func"]
        self.x0 = kargs["x0"]
        
        if "eps" in kargs.keys():
            self.eps = kargs["eps"]
        else:
            self.eps = 1e-12
    
    def run(self):
        x_mean = np.mean(self.x0)
        y_mean = derivative(self.func,x_mean,dx = 1e-6)
        
        if derivative(self.func,self.x0[0],dx = 1e-6)*y_mean < 0:
            self.x0[1] = x_mean
        else:
            self.x0[0] = x_mean
        if np.abs(self.x0[0]-self.x0[1]) < self.eps or y_mean == 0:
            return ((x_mean,self.func((self.x0[0]+self.x0[1])/2)))
        else:
            return self.run()

def CGSC(eta, z, K, eplison, L, F, xin):

    tau = 0
    d = xin.shape[0]
    # d = 2
    c = np.zeros((L+10, d))
    c[1, :] = np.array(xin)
    v = np.zeros((L+10, d))
    s = []
    s.append(0)

    def derivative_F(x):
        return eta * z + 2 * x

    def F_s(s):
        return eta * np.dot(z, c[tau,:] + s*(v[tau,:]-c[tau,:])) + math.pow(np.linalg.norm(c[tau,:] + s*(v[tau,:]-c[tau,:]), ord=2), 2)

    while True:

        #4
        tau = tau + 1

        #5 v[tau]
        # print("derivative:", derivative_F(c[tau,:]))
        tmp_der = np.linalg.norm(derivative_F(c[tau,:]), ord=2)
        if(tmp_der == 0):
            v[tau,:] = 0
        else:
            v[tau,:] = -(derivative_F(c[tau,:]) * 20) / tmp_der

        #6 s[tau]
        # stau = Minimize_Dichotomy(func=F_s, x0 = [0, 1]).run()
        # s.append(stau[0])

        # print("stau:", stau[0])
        tmp1 = math.pow(np.linalg.norm((v[tau,:] - c[tau,:]), ord = 2), 2)
        tmp2 = eta * np.dot(z, (v[tau,:] - c[tau,:])) + 2 * np.dot(c[tau,:], (v[tau,:] - c[tau,:]))
        if(tmp1 != 0):
            star = -tmp2 / (2 * tmp1)
            if star > 1:
                sprime = 1
            elif star < 0:
                sprime = 0
            else:
                sprime = star
            # print("sprime:", sprime)
            s.append(sprime)
        else:
            return c[tau,:]
        
        #7 c[tau+1]
        c[tau + 1,:] = c[tau,:] + s[tau] * (v[tau,:] - c[tau,:])
        if(np.dot(derivative_F(c[tau,:]), c[tau,:] - v[tau,:]) <= eplison or tau >= L):
            break
 
    return c[tau,:]

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


def DBOCG(O_grad, data_collection, target_collection, num_clients, mu, mat_A, comm_budget, radius=10,
        is_l2_norm=True, selected_learner=-1, is_clique=False, stepsize_factor=10):
    time_horizon, n_features = data_collection[0].shape
    # x_t = initialize_weights(n_features, num_clients)
    z_t = initialize_weights(n_features, num_clients)
    x_t = initialize_weights(n_features, num_clients)

    if(selected_learner == -1):
        selected_learner = np.random.randint(num_clients)
    print("Selected learner:", selected_learner)
    # x_t_selected = x_t[selected_learner]

    # For efficiency
    selected_learner = -1

    L_cons = math.sqrt(2)
    C_norm = 2 * radius
    d = n_features

    iteration, tau, comm_cost = compute_iteration_tau(comm_budget, time_horizon, num_clients, mu, is_clique=is_clique)
    # batch_cnt = 1

    # 梯度下降
    # learning_rate = 2 * C_norm / (L_cons * math.sqrt(iteration))
    learning_rate = stepsize_factor / (time_horizon)**(0.75)
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
                    temp = temp + mat_A[j, userID] * z_t[j]
                v_t.append(temp)
            # print(v_t)
            if(t % 20 == 0):
                print("iteration t:", t)
            for userID in range(num_clients):
                X_train, y_train = load_data_interval_dist(data_collection, target_collection, n_features, startTime,
                                                           endTime, userID, num_clients)
                # tmp = L_G + sigma_bar * math.sqrt(2 * (t + 1)) / (math.sqrt(num_clients * (tau - mu)) * C_norm)
                # learning_rate = 1. / tmp
                grad = O_grad(x_t[userID], X_train, y_train, is_sum=True)
                # print("gradient norm:", np.linalg.norm(grad))
                # print(f'grad shape: {grad.shape}')
                # x_t[userID] = gradient_descent(v_t[userID], grad, learning_rate, radius, is_l2_norm)
                z_t[userID] = v_t[userID] + grad
                # lines 9-10 in B-BOCG in Wan12 to update x_t[userID]
                # TODO
                eta = learning_rate
                z = z_t[userID]
                def F(x):
                    return eta * np.dot(z, x) + math.pow(np.linalg.norm(x, ord=2), 2)
                # print("shape:", x_t[userID].shape, z_t[userID].shape)
                x_t[userID] = CGSC(eta=learning_rate, z=z_t[userID], K=20, eplison=1e-5, L=20, F=F, xin=x_t[userID])
        # print(x_t[selected_learner])

    # print("resulting model x: ", x_t[selected_learner])
    loss_mean = loss_sum_list / (time_horizon * num_clients)
    class_err_mean = class_err_sum_list / (time_horizon * num_clients)
    print("loss mean:", loss_mean)
    print("class err mean:", class_err_mean)
    print("comm cost:", comm_cost)
    # print("||x||_2: ", np.linalg.norm(x_t[selected_learner]))
    return loss_mean, class_err_mean, comm_cost

