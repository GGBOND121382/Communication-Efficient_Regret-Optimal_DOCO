from sklearn import preprocessing
import time

import sys
sys.path.append('../')

from config_save_load import conf_load
from optimization_utils.LogisticRegression import *
from optimization_utils.acc_grad_descent import *
from optimization_utils.communication_budget import *
from data.libsvm_data_load import *
# from libsvm_data_load import load_libsvm_data


def oracle_comm_const_iid_AGD(comm_const, time_list, num_of_clients, mu):
    inner_batchsize_prime = mu + num_of_clients
    comm_cost = 0
    for t_i in time_list:
        inner_batchsize = np.ceil(max(comm_const * (num_of_clients * t_i) ** 0.75, inner_batchsize_prime))
        comm_cost += int(t_i / inner_batchsize) * (2 * (num_of_clients - 1))
    return comm_cost


def parallel_run_AGD_dist(grad_logReg, data_collection, target_collection, num_of_client, mu, comm_budget, radius=10,
                          is_l2_norm=True):
    time_horizon, n_features = data_collection[0].shape
    d = n_features
    x0 = np.zeros(d)
    # outer_iteration = np.floor(m_samples / num_of_client)
    # outer_iteration = int(outer_iteration)
    ################################################################################## generate T_list
    stretch_factor = 1
    batch_list = []
    T_list = []
    comm_cost = 0
    num_parallel_prime = np.ceil(np.log(np.log(num_of_client)))
    num_parallel = np.ceil(np.log(np.log(time_horizon)))
    num_parallel_prime = int(num_parallel_prime)
    num_parallel = int(num_parallel)
    num_parallel += num_parallel_prime
    ######################################## t < t_m_prime
    inner_batchsize_prime = mu + num_of_client
    inner_batchsize_prime = int(inner_batchsize_prime)
    for i in range(num_parallel_prime):
        t_i = inner_batchsize_prime + num_of_client ** (
                    1 + 2 / 3 * (2 ** (i + 1) - 1) / (2 ** num_parallel_prime - 1))
        t_i = int(np.floor(t_i))
        T_list.append(t_i)
        batch_list.append(inner_batchsize_prime)
        comm_cost += int(t_i / inner_batchsize_prime) * (2 * (num_of_client - 1))
        print(comm_cost)
    ######################################## t > t_m_prime
    t_m_prime = t_i
    T_list_m = []
    for i in range(num_parallel_prime, num_parallel):
        t_i = t_m_prime + (time_horizon - t_m_prime) ** ((2 - 2 ** (-i + num_parallel_prime)) / (
                2 - 2 ** (-(num_parallel - num_parallel_prime))))
        t_i = int(np.ceil(t_i))
        T_list_m.append(t_i)
        T_list.append(t_i)

    comm_budget_m = comm_budget - comm_cost
    assert comm_budget_m > 0
    comm_const = determine_comm_const_AGD(oracle_comm_const_iid_AGD, comm_budget_m, time_horizon, 0, T_list_m,
                                          num_of_client, mu)
    print("comm_const:", comm_const)

    for i in range(num_parallel_prime, num_parallel):
        t_i = T_list[i]
        inner_batchsize = np.ceil(max(comm_const * (num_of_client * t_i) ** 0.75, inner_batchsize_prime))
        # inner_batchsize = inner_batchsize_prime
        inner_batchsize = int(inner_batchsize)
        batch_list.append(inner_batchsize)
        comm_cost += int(t_i / inner_batchsize) * (2 * (num_of_client - 1))
    print("comm. cost", comm_cost)

    # compute the model

    x_list = np.zeros((num_parallel, d))
    print("T_list:", T_list)
    print("batch size list:", batch_list)
    for i in range(num_parallel):
        t_i = T_list[i]
        inner_batchsize = batch_list[i]
        # X_train = load_data_interval_dist(data_collection, target_collection, n_features, 0, t_i, )
        inner_data_collection, inner_target_collection = load_data_collection_interval_dist(data_collection,
                                                                                            target_collection,
                                                                                            n_features, 0, t_i,
                                                                                            num_of_client)
        x = ACC_gradient_stoc_dist(grad_logReg, inner_data_collection, inner_target_collection, inner_batchsize,
                                   num_of_client, mu, radius=radius, is_l2_norm=is_l2_norm)

        print("----------------resulting x:", x.tolist())
        print('max(|x_i|) = ' + repr(max(np.abs(x))))
        print('||x||_1 = ' + repr(np.sum((np.abs(x)))))
        print('||x||_2 = ' + repr(np.linalg.norm(x)))
        x_list[i, :] = x

    ######################################################################################
    # evaluate the regret loss
    x = x0
    loss_sum = 0
    class_err_sum = 0
    T_list = [0, ] + T_list + [time_horizon, ]
    for i in range(num_parallel + 1):
        startTime = T_list[i]
        endTime = T_list[i+1]
        X_eval, y_eval = load_data_interval_dist(data_collection, target_collection, n_features, startTime, endTime, -1,
                                                 num_of_client)
        loss = loss_logReg(x, X_eval, y_eval, is_sum=True)
        class_err = class_err_logReg(x, X_eval, y_eval, is_sum=True)
        loss_sum += loss
        class_err_sum += class_err
        if(i < num_parallel):
            x = x_list[i]
    loss_mean = loss_sum / (time_horizon * num_of_client)
    class_err_mean = class_err_sum / (time_horizon * num_of_client)
    print("loss mean:", loss_mean)
    print("class err mean:", class_err_mean)

    return loss_mean, class_err_mean, comm_cost


def run():

    conf_dict = conf_load()
    dirname = conf_dict['dirname']
    # dirname = '../data/iid_data/'
    filename = conf_dict['data']
    dimension = conf_dict['dimension']
    datasize = conf_dict['datasize']
    rpt_times = conf_dict['rpt_times']
    number_of_clients = conf_dict['number_of_clients']
    to_shuffle = conf_dict['to_shuffle']
    is_minus_one = conf_dict['is_minus_one']
    radius = conf_dict['radius']
    comm_budget_list = conf_dict['comm_budget_list']

    '''
    startTime = time.time()
    X = np.load(dirname + filename + '_data.npy')
    y = np.load(dirname + filename + '_target.npy')
    endTime = time.time()
    print("Data loaded: " + repr(endTime - startTime) + "seconds")
    print('X\'s maximum norm: ' + repr(max(np.linalg.norm(X, axis=1))))
    startTime = time.time()
    X = MaxAbsScaler().fit_transform(X)
    X = preprocessing.normalize(X, norm='l2')
    y = np.array(y, dtype=int)
    y = np.reshape(y, (-1, 1))
    negative_per = 1 - np.sum(y) / datasize
    endTime = time.time()
    print("Preprocessing complete: " + repr(endTime - startTime) + "seconds")
    print("positive percentage:", negative_per)
    '''

    startTime = time.time()
    data_collection, target_collection = load_data_collection_from_file(filename, dimension,
                                                                        num_of_clients=number_of_clients,
                                                                        dirname=dirname)
    data_collection = data_collection_preprocess(data_collection)
    endTime = time.time()
    print("Data loaded: " + repr(endTime - startTime) + "seconds")
    print("y", target_collection[0][:10, :])
    # print("max ||x||_2:", np.max(np.linalg.norm(data_collection[0], axis=1)))

    # num_parallel = int(np.log(np.log(datasize / number_of_clients)))
    # print("num_parallel:", num_parallel)

    if ('epsilon' in filename):
        comm_upp_bnd = 11000
    else:
        comm_upp_bnd = 16000
    for comm_bueget in comm_budget_list:
        if (comm_bueget < 496):
            continue
        if (comm_bueget > comm_upp_bnd):  # if comm_budget > 36000, the batch size is too small
            break
        plot_filename = './plot_data/AGD_' + filename + '_' + repr(comm_bueget)
        fd = open(plot_filename, 'a')

        startTime = time.time()
        loss_mean, class_err_mean, comm_cost = parallel_run_AGD_dist(grad_logReg, data_collection, target_collection,
                                                                     number_of_clients, number_of_clients, comm_bueget,
                                                                     radius=radius, is_l2_norm=True)
        endTime = time.time()

        print("OCO complete: " + repr(endTime - startTime) + "seconds")
        print(f'filename = {filename}, comm_budget = {comm_bueget}, loss_mean = {loss_mean},'
              f' class_err_mean={class_err_mean}, comm_cost = {comm_cost}')

        fd.write(repr(loss_mean) + ' ' + repr(class_err_mean) + ' ' + repr(comm_cost) + '\n')
        fd.close()


if __name__ == '__main__':
    run()
