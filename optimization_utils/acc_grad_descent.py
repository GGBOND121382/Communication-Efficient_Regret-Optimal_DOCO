import numpy as np
import math
import sys
# from libsvm_data_load import load_libsvm_data
from sklearn.preprocessing import MaxAbsScaler

sys.path.append('../')
from optimization_utils.LogisticRegression import *
from data.libsvm_data_load import *


def ACC_gradient_stoc_dist(O_grad, data_collection, target_collection, batchsize, num_clients, mu, radius=10,
                           is_l2_norm=True):
    time_horizon, n_features = data_collection[0].shape
    Lip_G = 1. / 2                                   # smooth constant
    Lip_cons = math.sqrt(2)                          # Lipschitz constant
    d = n_features
    if (is_l2_norm):
        C_norm = 2 * radius
    else:
        C_norm = 2 * math.sqrt(d) * radius
    sigma = math.sqrt(Lip_cons ** 2 / ((batchsize - mu) * num_clients))
    b = math.sqrt(5) / 3 * sigma / C_norm
    print(f'values of b: {b}')
    print(f'sigma: {sigma}')
    print(f'C_norm: {C_norm}')
    # iteration = int(m_samples / (batchsize * num_clients))
    iteration = int(np.floor(time_horizon / batchsize))
    ############################## Initialize
    y = np.zeros(d)
    z = np.zeros(d)
    #########################################
    for t in range(iteration):
        ########################## Step size parameter
        L_t = b * (t + 1) ** 1.5 + Lip_G
        alpha = 2. / (t + 2)
        ########################## Model update
        x_t = (1 - alpha) * y + alpha * z
        startTime = t * batchsize
        if(t < iteration - 1):
            endTime = (t + 1) * batchsize - mu
        else:
            endTime = max(time_horizon - mu, startTime)
        X_train, y_train = load_data_interval_dist(data_collection, target_collection, n_features, startTime,
                                                   endTime, -1, num_clients)
        grad = O_grad(x_t, X_train, y_train)

        y = gradient_descent(x_t, grad, 1/L_t, radius, is_l2_norm)
        z = z - 1. / (L_t * alpha) * (L_t * (x_t - y))
        z = projection(z, radius, is_l2_norm=True)

        ################################### Evaluate the loss
        # X_test, y_test = load_data_interval_dist(data_collection, target_collection, n_features, t * batchsize,
        #                                            (t + 1) * batchsize, -1, num_clients)
        # loss = loss_logReg(y, X_test, y_test)
        # print(f'stepsize: {1/L_t}')
        # print(f't: {t}, loss: {loss}')
        ####################################
    return y


if __name__ == '__main__':
    print(1)
