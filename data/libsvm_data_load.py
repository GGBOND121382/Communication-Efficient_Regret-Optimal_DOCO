import numpy as np
import time
from config_save_load import conf_load
import copy
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler


def load_libsvm_data(filename, dimension, datasize=100, to_shuffle=True, is_minus_one=False, dirname='original_data/'):
    filename = dirname + filename
    data_target = np.zeros((datasize, dimension + 1))
    non_setted = True
    cnt = 0
    with open(filename) as fd:
        for line in fd.readlines():
            line = line[:-1]
            tmp = line.split()
            num_nonzero_indeces = len(tmp)
            data_target[cnt, 0] = int(tmp[0])
            for i in range(1, num_nonzero_indeces):
                j, k = tmp[i].split(':')
                j = int(j)
                k = float(k)
                data_target[cnt, j] = k
            cnt += 1
            if (cnt >= datasize):
                break
    if (to_shuffle):
        np.random.shuffle(data_target)
    data = data_target[:, 1:]
    if (not is_minus_one):
        target = data_target[:, 0] - 1
    else:
        target = (data_target[:, 0] + 1) / 2
    return (data, target)


def read_mat(filename):
    mat = []
    with open(filename) as fd:
        for line in fd.readlines():
            tmp = line.split()
            float_tmp = [float(i) for i in tmp]
            mat.append(float_tmp)
    return np.array(mat)


def data_vertical_merge(src_filename1, src_filename2, dst_filename, dirname='original_data/'):
    src_data1_data = np.load(dirname + src_filename1 + '_data.npy')
    src_data1_target = np.load(dirname + src_filename1 + '_target.npy')
    src_data1_target = np.reshape(src_data1_target, (-1, 1))
    src_data2_data = np.load(dirname + src_filename2 + '_data.npy')
    src_data2_target = np.load(dirname + src_filename2 + '_target.npy')
    src_data2_target = np.reshape(src_data2_target, (-1, 1))
    dst_data_data = np.vstack((src_data1_data, src_data2_data))
    dst_data_target = np.vstack((src_data1_target, src_data2_target))
    np.save(dirname + dst_filename + '_data.npy', dst_data_data)
    np.save(dirname + dst_filename + '_target.npy', dst_data_target)


def data_save_from_libsvm():
    conf_dict = conf_load()
    dirname = conf_dict['dirname']
    filename = conf_dict['data']
    dimension = conf_dict['dimension']
    datasize = conf_dict['datasize']
    rpt_times = conf_dict['rpt_times']
    number_of_clients = conf_dict['number_of_clients']
    to_shuffle = conf_dict['to_shuffle']
    is_minus_one = conf_dict['is_minus_one']

    if(filename == 'epsilon_normalized.all'):
        startTime = time.time()
        X1, y1 = load_libsvm_data('epsilon_normalized', dimension, datasize=datasize, to_shuffle=to_shuffle,
                                is_minus_one=is_minus_one)
        X2, y2 = load_libsvm_data('epsilon_normalized.t', dimension, datasize=datasize, to_shuffle=to_shuffle,
                                  is_minus_one=is_minus_one)
        X = np.vstack(X1, X2)
        y = np.vstack(y1, y2)
        endTime = time.time()
        print("Data loaded: " + repr(endTime - startTime) + "seconds")
    else:
        # T = 200000
        # X, y = load_skin_pad(T)
        startTime = time.time()
        X, y = load_libsvm_data(filename, dimension, datasize=datasize, to_shuffle=to_shuffle, is_minus_one=is_minus_one)
        endTime = time.time()
        print("Data loaded: " + repr(endTime - startTime) + "seconds")

    startTime = time.time()
    np.save(dirname + filename + '_data.npy', X)
    np.save(dirname + filename + '_target.npy', y)
    endTime = time.time()
    print("Data saved (np): " + repr(endTime - startTime) + "seconds")

    startTime = time.time()
    X = np.load(dirname + filename + '_data.npy')
    y = np.load(dirname + filename + '_target.npy')
    endTime = time.time()
    print("Data loaded (np): " + repr(endTime - startTime) + "seconds")


def load_data_collection_from_file(filename, dimension, num_of_clients=8, dirname='non_iid_data/'):
    data_collection = []
    target_collection = []
    for i in range(num_of_clients):
        client_filename = filename + '_client' + repr(i)
        client_data = np.load(dirname + client_filename + '_data.npy')
        client_target = np.load(dirname + client_filename + '_target.npy')
        client_target = np.reshape(client_target, (-1, 1))
        data_collection.append(client_data)
        target_collection.append(client_target)
    return data_collection, target_collection

def data_collection_preprocess(data_collection, to_max_abs=True, to_l2_norm=True, to_pad=True):
    ret_data_collection = []
    for client_data in data_collection:
        ret_client_data = copy.deepcopy(client_data)
        if(to_max_abs):
            ret_client_data = MaxAbsScaler().fit_transform(ret_client_data)
        if(to_l2_norm):
            ret_client_data = preprocessing.normalize(ret_client_data, norm='l2')
        if(to_pad):
            ret_client_data = np.insert(ret_client_data, 0, 1, axis=1)
        ret_data_collection.append(ret_client_data)
    # print("max ||x||_2:", np.max(np.linalg.norm(ret_data_collection[0], axis=1)))
    return ret_data_collection


def load_data_interval_dist_from_file(filename, dimension, startTime, endTime, userID, num_of_clients=8, dirname='non_iid_data/'):
    """
    userID: ranging from -1 to num_of_clients; this function returns the data of all learners if userID == -1
    """
    if(userID != -1):
        client_filename = filename + '_client' + repr(userID)
        ret_X = np.load(dirname + client_filename + '_data.npy')
        ret_y = np.load(dirname + client_filename + '_target.npy')
        ret_y = np.reshape(ret_y, (-1, 1))
        return ret_X[startTime:endTime, :], ret_y[startTime:endTime, :]
    else:
        tau = endTime - startTime
        ret_X = np.zeros((tau * num_of_clients, dimension))
        ret_y = np.zeros((tau * num_of_clients, 1))
        for i in range(num_of_clients):
            X_user, y_user = load_data_interval_dist_from_file(filename, dimension, startTime, endTime, i, num_of_clients, dirname)
            ret_X[i * tau:(i + 1) * tau, :] = X_user
            ret_y[i * tau:(i + 1) * tau, :] = y_user
        return ret_X, ret_y


def load_data_interval_dist(data_collection, target_collection, dimension, startTime, endTime, userID, num_of_clients=8):
    """
    data_collection, target_collection: these data structures are list of np-arrays, i.e., each element of
    data_collection is the data of a user.
    userID: ranging from -1 to num_of_clients; this function returns the data
    of all learners if userID == -1.
    """
    if(userID != -1):
        # client_filename = filename + '_client' + repr(userID)
        ret_X = data_collection[userID][startTime:endTime, :]
        ret_y = target_collection[userID][startTime:endTime, :]
        return ret_X, ret_y
    else:
        tau = endTime - startTime
        ret_X = np.zeros((tau * num_of_clients, dimension))
        ret_y = np.zeros((tau * num_of_clients, 1))
        for i in range(num_of_clients):
            X_user, y_user = load_data_interval_dist(data_collection, target_collection, dimension, startTime, endTime,
                                                     i, num_of_clients)
            ret_X[i * tau:(i + 1) * tau, :] = X_user
            ret_y[i * tau:(i + 1) * tau, :] = y_user
        return ret_X, ret_y


def load_data_collection_interval_dist(data_collection, target_collection, dimension, startTime, endTime, num_of_clients=8):
    """
    data_collection, target_collection: these data structures are list of np-arrays, i.e., each element of
    data_collection is the data of a user.
    userID: ranging from -1 to num_of_clients; this function returns the data
    of all learners if userID == -1.
    """
    ret_data_collection = []
    ret_target_collection = []
    for i in range(num_of_clients):
        X_user, y_user = load_data_interval_dist(data_collection, target_collection, dimension, startTime, endTime, i, num_of_clients)
        ret_data_collection.append(X_user)
        ret_target_collection.append(y_user)
    return ret_data_collection, ret_target_collection


if __name__ == '__main__':
    conf_dict = conf_load()
    dirname = conf_dict['dirname']
    filename = conf_dict['data']
    dimension = conf_dict['dimension']
    datasize = conf_dict['datasize']
    rpt_times = conf_dict['rpt_times']
    number_of_clients = conf_dict['number_of_clients']
    to_shuffle = conf_dict['to_shuffle']
    is_minus_one = conf_dict['is_minus_one']
    # for the original data from libsvm, run data_save_from_libsvm()
    data_save_from_libsvm()
    # to merge data, run data_vertical_merge()
    if(filename == 'epsilon_normalized.all'):
        data_vertical_merge('epsilon_normalized', 'epsilon_normalized.t', filename)
