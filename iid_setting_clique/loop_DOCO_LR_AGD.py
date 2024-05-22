from sklearn import preprocessing
import time

import sys
sys.path.append('../')

from optimization_utils.LogisticRegression import *
from optimization_utils.acc_grad_descent import *
from optimization_utils.communication_budget import *
from data.libsvm_data_load import *
# from libsvm_data_load import load_libsvm_data
from data.config_save_load import *

def oracle_comm_const_iid_AGD(comm_const, time_list, num_of_clients, mu):
    inner_batchsize_prime = mu + num_of_clients
    comm_cost = 0
    for t_i in time_list:
        inner_batchsize = np.ceil(max(comm_const * (num_of_clients * t_i) ** 0.75, inner_batchsize_prime))
        comm_cost += int(t_i / inner_batchsize) * (2 * (num_of_clients - 1))
    return comm_cost


def parallel_run_AGD_dist(grad_logReg, data_collection, target_collection, num_of_client, mu, comm_budget,
                          selected_learner=-1, radius=10, is_l2_norm=True):
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
    num_parallel = np.ceil(np.log(np.log(time_horizon)))
    num_parallel = int(num_parallel) + 1

    if (selected_learner == -1):
        selected_learner = np.random.randint(num_of_client)

    print("outer iteration:", time_horizon)
    # Lip_cons = math.sqrt(n_features)  # Lipschitz constant
    # C_norm = 2 * math.sqrt(d) * radius
    Lip_cons = math.sqrt(2)

    '''
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
    '''
    ######################################## t < t_1
    inner_batchsize_prime = mu + num_of_client
    inner_batchsize_prime = int(inner_batchsize_prime)
    t_i = inner_batchsize_prime + num_of_client ** (5. / 3)
    t_i = int(np.floor(t_i))
    T_list.append(t_i)
    batch_list.append(inner_batchsize_prime)
    comm_cost += int(t_i / inner_batchsize_prime) * (2 * (num_of_client - 1))
    print(comm_cost)

    ######################################## t > t_1
    t_m_prime = t_i
    T_list_m = []
    for i in range(1, num_parallel):
        t_i = t_m_prime + (time_horizon - t_m_prime) ** ((2 - 2 ** (-i + 1)) / (
                2 - 2 ** (-(num_parallel - 1))))
        t_i = int(np.ceil(t_i))
        T_list_m.append(t_i)
        T_list.append(t_i)

    comm_budget_m = comm_budget - comm_cost
    assert comm_budget_m > 0
    comm_const = determine_comm_const_AGD(oracle_comm_const_iid_AGD, comm_budget_m, time_horizon, 0, T_list_m,
                                          num_of_client, mu)
    print("comm_const:", comm_const)

    for i in range(1, num_parallel):
        t_i = T_list[i]
        inner_batchsize = np.ceil(max(comm_const * (num_of_client * t_i) ** 0.75, inner_batchsize_prime))
        # inner_batchsize = inner_batchsize_prime
        inner_batchsize = int(inner_batchsize)
        batch_list.append(inner_batchsize)
        comm_cost += int(t_i / inner_batchsize) * (2 * (num_of_client - 1))
    print("comm. cost", comm_cost)
    print("T_list:", T_list)
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
        X_eval, y_eval = load_data_interval_dist(data_collection, target_collection, n_features, t, t + 1,
                                                 -1, num_of_client)
        loss = loss_logReg(x, X_eval, y_eval, is_sum=True)
        class_err = class_err_logReg(x, X_eval, y_eval, is_sum=True)
        loss_sum += loss
        class_err_sum += class_err
        x = gradient_descent(x, grad, learning_rate, radius, is_l2_norm)
    print("loss (first stage):", loss_sum / (t_1 * num_of_client))
    print("class_err (first stage):", class_err_sum / (t_1 * num_of_client))

    #################################################################################
    # generate the model >t1

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


def run_comm():

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

    num_parallel = int(np.log(np.log(datasize / number_of_clients)))
    # print("num_parallel:", num_parallel)

    for comm_bueget in comm_budget_list:
        if comm_bueget < 620 and number_of_clients == 32:
            continue
        if comm_bueget > 20000 and number_of_clients == 32:                    # if comm_budget > 36000, the batch size is too small
            break
        elif comm_bueget > 36000 and number_of_clients == 8:
            break
        plot_filename = './plot_data_' + filename[:3] + '_' + repr(number_of_clients) + '/AGD_' + filename + '_' + repr(comm_bueget)
        fd = open(plot_filename, 'a')
        startTime = time.time()
        loss_mean, class_err_mean, comm_cost = parallel_run_AGD_dist(grad_logReg, data_collection, target_collection,
                                                                     number_of_clients, 2, comm_bueget,
                                                                     radius=radius, is_l2_norm=True)
        endTime = time.time()


        # for i in range(number_of_clients):
        #     print(target_collection[i][:10])

        print("OCO complete: " + repr(endTime - startTime) + "seconds")
        print(f'filename = {filename}, comm_budget = {comm_bueget}, loss_mean = {loss_mean},'
              f' class_err_mean={class_err_mean}, comm_cost = {comm_cost}')

        fd.write(repr(loss_mean) + ' ' + repr(class_err_mean) + ' ' + repr(comm_cost) + '\n')
        fd.close()


def run_time():

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
    original_data_collection, original_target_collection = load_data_collection_from_file(filename, dimension,
                                                                                          num_of_clients=number_of_clients,
                                                                                          dirname=dirname)
    original_data_collection = data_collection_preprocess(original_data_collection)
    # target_collection = target_adversarize(target_collection, number_of_clients, kb=3)
    time_horizon, n_features = original_data_collection[0].shape
    endTime = time.time()
    print("Data loaded: " + repr(endTime - startTime) + "seconds")
    # print("y", target_collection[0][:10, :])
    # print("max ||x||_2:", np.max(np.linalg.norm(data_collection[0], axis=1)))

    num_parallel = int(np.log(np.log(datasize / number_of_clients)))
    # print("num_parallel:", num_parallel)

    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        data_collection, target_collection = load_data_collection_interval_dist(original_data_collection,
                                                                                original_target_collection, dimension,
                                                                                0, running_time_horizon,
                                                                                number_of_clients)
        for comm_bueget in comm_budget_list:
            if(comm_bueget > 36000 / 5 * i):                    # if comm_budget > 36000, the batch size is too small
                break
            plot_filename = './plot_data_' + filename[:3] + '_' + repr(number_of_clients) + '/AGD_' + filename + '_' + repr(comm_bueget) + '_' + repr(running_time_horizon)
            fd = open(plot_filename, 'a')
            startTime = time.time()
            loss_mean, class_err_mean, comm_cost = parallel_run_AGD_dist(grad_logReg, data_collection, target_collection,
                                                                         number_of_clients, 2, comm_bueget,
                                                                         radius=radius, is_l2_norm=True)
            endTime = time.time()


            # for i in range(number_of_clients):
            #     print(target_collection[i][:10])

            print("OCO complete: " + repr(endTime - startTime) + "seconds")
            print(f'filename = {filename}, comm_budget = {comm_bueget}, loss_mean = {loss_mean},'
                  f' class_err_mean={class_err_mean}, comm_cost = {comm_cost}')

            fd.write(repr(loss_mean) + ' ' + repr(class_err_mean) + ' ' + repr(comm_cost) + '\n')
            fd.close()


# if __name__ == '__main__':
#     run()
