from sklearn import preprocessing
import time

import sys
sys.path.append('../')

from config_save_load import conf_load
from optimization_utils.LogisticRegression import *
from optimization_utils.cutting_plane_Vaidya import *
from optimization_utils.communication_budget import *
from data.libsvm_data_load import *


def oracle_comm_const_iid_CP(comm_const, time_horizon, num_of_clients, mu, num_parallel, dimension, Lip_cons, C_norm, M_cons):
    print("stretch factor:", comm_const)
    comm_cost = 0
    T_list_tmp = []
    batch_list_tmp = []
    t_i = int(np.ceil((mu + 2) * comm_const * dimension * np.log(num_of_clients * time_horizon)))
    inner_batchsize = compute_inner_batchsize(t_i, comm_const, dimension, num_of_clients, Lip_cons, C_norm, M_cons, mu)

    T_list_tmp.append(t_i)
    batch_list_tmp.append(inner_batchsize)
    # print(batch_list_tmp)
    comm_cost += int(t_i / inner_batchsize) * (2 * (num_of_clients - 1))

    t_m_prime = t_i

    print("communication cost:", comm_cost)
    print("t_m_prime:", t_m_prime)

    for i in range(1, num_parallel):
        if (t_m_prime > time_horizon):
            comm_cost = time_horizon * 2 * num_of_clients           # comm_const is way too big
            break
        t_i = t_m_prime + (time_horizon - t_m_prime) ** ((2 - 2 ** (-i + 1)) / (
                2 - 2 ** (-(num_parallel - 1))))
        print("t_i:", t_i)
        t_i = int(np.ceil(t_i))

        inner_batchsize = compute_inner_batchsize(t_i, comm_const, dimension, num_of_clients, Lip_cons, C_norm, M_cons, mu)

        T_list_tmp.append(t_i)
        batch_list_tmp.append(inner_batchsize)
        comm_cost += int(t_i / inner_batchsize) * (2 * (num_of_clients - 1))
    print("resulting communication cost:", comm_cost)
    return comm_cost


def compute_inner_batchsize(T, C_1, d, num_of_client, Lip_cons, C_norm, M_cons, mu):
    sigma = np.sqrt(num_of_client / (T - (mu+1) * C_1 * d * np.log(num_of_client * T)))
    epsilon_0 = d ** 1.5 * np.log(num_of_client * T) * sigma
    log_term = max(np.log(d * num_of_client / epsilon_0), 1)
    # print(num_of_client * (Lip_cons**2 * C_norm**2 + M_cons**2))
    # print(T)
    # print("C_1", C_1)
    # print("sigma:", sigma)
    # print("epsilon_0:", epsilon_0)
    # print("log_term:", np.log(d * num_of_client * Lip_cons * C_norm / epsilon_0))
    iteration = C_1 * d * log_term
    batchsize = int(T / iteration)
    return batchsize

def parallel_run_cutting_plane_dist(grad_logReg, loss_logReg, cutting_oracle_l2, data_collection, target_collection,
                                    num_of_client, mu, comm_budget, selected_learner=-1, radius=50, is_l2_norm=True):
    time_horizon, n_features = data_collection[0].shape
    d = n_features
    A = np.vstack((np.eye(d), -np.eye(d)))
    b = - np.ones(2 * d) * radius
    x0 = np.zeros(d)

    if(selected_learner==-1):
        selected_learner = np.random.randint(num_of_client)

    print("outer iteration:", time_horizon)
    # Lip_cons = math.sqrt(n_features)  # Lipschitz constant
    # C_norm = 2 * math.sqrt(d) * radius
    Lip_cons = math.sqrt(2)
    C_norm = radius
    M_cons = 1
    ################################################################################## generate T_list
    ######################################################################################## stretch the batchsize
    # num_parallel_prime_tmp = 1
    num_parallel = np.ceil(np.log(np.log(time_horizon)))
    # num_parallel_prime_tmp = int(num_parallel_prime_tmp)
    num_parallel = int(num_parallel) + 1
    # while (True):
    ##########################################################
    # print("stretch factor:", stretch_factor)
    lwr_const = 1e-12
    upp_const = time_horizon / ((mu + 2) * d * np.log(num_of_client * time_horizon))
    comm_const = determine_comm_const_CP(oracle_comm_const_iid_CP, comm_budget, lwr_const, upp_const, time_horizon,
                                         num_of_client, mu, num_parallel, n_features, Lip_cons, C_norm, M_cons)

    comm_cost = 0
    T_list = []
    batch_list = []
    t_i = int(np.ceil((mu + 2) * comm_const * d * np.log(num_of_client * time_horizon)))
    inner_batchsize = compute_inner_batchsize(t_i, comm_const, d, num_of_client, Lip_cons, C_norm, M_cons, mu)
    comm_cost += int(t_i / inner_batchsize) * (2 * (num_of_client - 1))

    T_list.append(t_i)
    batch_list.append(inner_batchsize)
    t_m_prime = t_i

    for i in range(1, num_parallel):
        t_i = t_m_prime + (time_horizon - t_m_prime) ** ((2 - 2 ** (-i + 1)) / (
                2 - 2 ** (-(num_parallel - 1))))
        t_i = int(np.ceil(t_i))

        inner_batchsize = compute_inner_batchsize(t_i, comm_const, d, num_of_client, Lip_cons, C_norm, M_cons, mu)

        T_list.append(t_i)
        batch_list.append(inner_batchsize)
        comm_cost += int(t_i / inner_batchsize) * (2 * (num_of_client - 1))
    print("communication cost:", comm_cost)

    ######################################################################################
    print("number of parallel", num_parallel)
    loss_sum = 0
    class_err_sum = 0
    #################################################################################
    # generate the model and evaluate loss <= t1
    t_1 = T_list[0]
    x = np.zeros(d)
    X_train, y_train = load_data_interval_dist(data_collection, target_collection, n_features, 0, t_1, selected_learner,
                                               num_of_client)
    # learning_rate = 2 * radius / (Lip_cons * math.sqrt(t_1))
    for t in range(t_1):
        L_G = 1. / 2
        sigma_bar = math.sqrt(Lip_cons ** 2)
        tmp = L_G + sigma_bar * math.sqrt(2 * (t + 1)) / (2 * radius)
        learning_rate = 1. / tmp

        grad = grad_logReg(x, X_train[t, :], y_train[t, :])
        X_eval, y_eval = load_data_interval_dist(data_collection, target_collection, n_features, t, t+1,
                                                   -1, num_of_client)
        loss = loss_logReg(x, X_eval, y_eval, is_sum=True)
        class_err = class_err_logReg(x, X_eval, y_eval, is_sum=True)
        loss_sum += loss
        class_err_sum += class_err
        x = gradient_descent(x, grad, learning_rate, radius, is_l2_norm)
    print("loss (first stage):", loss_sum / (t_1 * num_of_client))
    print("class_err (first stage):", class_err_sum / (t_1 * num_of_client))
    ################################################################################## generate the model >t1
    x_list = np.zeros((num_parallel, d))
    print("T_list:", T_list)
    print("batch size list:", batch_list)
    for i in range(num_parallel):
        t_i = T_list[i]
        batchsize = batch_list[i]
        inner_data_collection, inner_target_collection = load_data_collection_interval_dist(data_collection,
                                                                                            target_collection,
                                                                                            n_features, 0, t_i,
                                                                                            num_of_client)
        x = CP_Vaidya_stoc_dist(A, b, x0, grad_logReg, loss_logReg, cutting_oracle_l2, inner_data_collection,
                                inner_target_collection, batchsize, num_of_client, mu, eps=0, radius=radius,
                                is_l2_norm=is_l2_norm)
        print("----------------resulting x:", x.tolist())
        print('max(|x_i|) = ' + repr(max(np.abs(x))))
        print('||x||_1 = ' + repr(np.sum((np.abs(x)))))
        print('||x||_2 = ' + repr(np.linalg.norm(x)))
        x_list[i, :] = x

    ######################################################################################
    # evaluate the regret loss >t1
    # x = x_list[0]
    T_list = T_list + [time_horizon, ]
    for i in range(num_parallel):
        x = x_list[i]
        startTime = T_list[i]
        endTime = T_list[i + 1]
        X_eval, y_eval = load_data_interval_dist(data_collection, target_collection, n_features, startTime, endTime, -1,
                                                 num_of_client)
        loss = loss_logReg(x, X_eval, y_eval, is_sum=True)
        class_err = class_err_logReg(x, X_eval, y_eval, is_sum=True)
        loss_sum += loss
        class_err_sum += class_err
    loss_mean = loss_sum / (time_horizon * num_of_client)
    class_err_mean = class_err_sum / (time_horizon * num_of_client)
    print("loss mean:", loss_mean)
    print("class err mean:", class_err_mean)

    return loss_mean, class_err_mean, comm_cost


def run():
    conf_dict = conf_load()
    dirname = conf_dict['dirname']
    filename = conf_dict['data']
    dimension = conf_dict['dimension']
    datasize = conf_dict['datasize']
    rpt_times = conf_dict['rpt_times']
    number_of_clients = conf_dict['number_of_clients']
    to_shuffle = conf_dict['to_shuffle']
    is_minus_one = conf_dict['is_minus_one']
    radius = conf_dict['radius']
    # comm_budget_list = conf_dict['comm_budget_list']

    if ('covtype' in filename):
        comm_budget_list = [100, ] + [i for i in range(200, 3000, 200)] + [i for i in range(3000, 7000, 800)]
        # comm_budget_list = [i for i in range(3000, 7000, 800)]
    else:
        # comm_budget_list = [100, 200, 400, 600, 800, 1000, 1400, 1800, 2200]
        # comm_budget_list = [100, 800, 1200, 1600] + [i for i in range(2200, 7000, 800)]
        # comm_budget_list = [i for i in range(200, 800, 200)]
        # comm_budget_list = [300, ]
        comm_budget_list = [100, 200, 400, 600, 800, 1200, 1600] + [i for i in range(2200, 7000, 800)]


    startTime = time.time()
    data_collection, target_collection = load_data_collection_from_file(filename, dimension,
                                                                        num_of_clients=number_of_clients,
                                                                        dirname=dirname)
    data_collection = data_collection_preprocess(data_collection)
    endTime = time.time()
    print("Data loaded: " + repr(endTime - startTime) + "seconds")
    print("y", target_collection[0][:10, :])
    # print("max ||x||_2:", np.max(np.linalg.norm(data_collection[0], axis=1)))

    for comm_bueget in comm_budget_list:
        if(comm_bueget > 36000):                    # if comm_budget > 36000, the batch size is too small
            break
        plot_filename = './plot_data/CP_' + filename + '_' + repr(comm_bueget)
        fd = open(plot_filename, 'a')
        startTime = time.time()
        loss_mean, class_err_mean, comm_cost = parallel_run_cutting_plane_dist(grad_logReg, loss_logReg,
                                                                               cutting_oracle_l2, data_collection,
                                                                               target_collection, number_of_clients,
                                                                               2, comm_bueget,
                                                                               radius=radius, is_l2_norm=True)
        endTime = time.time()
        print("OCO complete: " + repr(endTime - startTime) + "seconds")

        print(f'filename = {filename}, comm_budget = {comm_bueget}, loss_mean = {loss_mean},'
              f' class_err_mean={class_err_mean}, comm_cost = {comm_cost}')

        fd.write(repr(loss_mean) + ' ' + repr(class_err_mean) + ' ' + repr(comm_cost) + '\n')
        fd.close()


if __name__ == '__main__':
    run()
