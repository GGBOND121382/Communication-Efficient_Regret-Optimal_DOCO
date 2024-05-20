from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
import copy
import sys
sys.path.append('../')

from optimization_utils.LogisticRegression import *
from data.libsvm_data_load import *


def DMA(O_grad, data_collection, target_collection, num_clients, mu, comm_budget, radius=10, is_l2_norm=True):
    time_horizon, n_features = data_collection[0].shape
    x_t = np.random.uniform(0, 0, (n_features, ))
    ############################################################
    L_G = 1. / 2
    L_cons = math.sqrt(2)  # Lipschitz constant
    d = n_features
    if (is_l2_norm):
        C_norm = 2 * radius
    else:
        C_norm = 2 * math.sqrt(d) * radius
    ############################################################

    # if self.comm_budget is not None:
    iteration = math.floor(comm_budget / (2 * num_clients - 2)) + 1
    tau = math.ceil(time_horizon / iteration)
    tau = max(tau, mu + 1)
    iteration = math.ceil(time_horizon / tau)
    tau = int(tau)
    iteration = int(iteration)
    # assert tau > mu
    # print("tau:", tau)
    comm_cost = (iteration - 1) * (2 * num_clients - 2)
    sigma_bar = math.sqrt(L_cons ** 2)
    print(f"time_horizon = {time_horizon}, tau = {tau}, iteration = {iteration}, comm_budget = {comm_budget},"
          f" comm_cost = {comm_cost}")

    loss_sum = 0
    class_err_sum = 0
    for t in range(iteration):
        startTime = t * tau
        if (t < iteration - 1):
            endTime = (t + 1) * tau
        else:
            endTime = time_horizon

        X_eval, y_eval = load_data_interval_dist(data_collection, target_collection, n_features, startTime,
                                                   endTime, -1, num_clients)
        # print(f'x_t shape: {x_t.shape}, X_eval shape: {X_eval.shape}, y_eval shape: {y_eval.shape}')
        loss = loss_logReg(x_t, X_eval, y_eval, is_sum=True)
        class_err = class_err_logReg(x_t, X_eval, y_eval, is_sum=True)
        loss_sum += loss
        class_err_sum += class_err

        if(t < iteration - 1):
            X_train, y_train = load_data_interval_dist(data_collection, target_collection, n_features, startTime,
                                                       endTime - mu, -1, num_clients)
            tmp = L_G + sigma_bar * math.sqrt(2 * (t + 1)) / (math.sqrt(num_clients * (tau - mu)) * C_norm)
            learning_rate = 1. / tmp
            grad = O_grad(x_t, X_train, y_train)
            # print(f'grad shape: {grad.shape}')
            x_t = gradient_descent(x_t, grad, learning_rate, radius, is_l2_norm)
    loss_mean = loss_sum / (time_horizon * num_clients)
    class_err_mean = class_err_sum / (time_horizon * num_clients)
    print("loss mean:", loss_mean)
    print("class err mean:", class_err_mean)
    return loss_mean, class_err_mean, comm_cost


