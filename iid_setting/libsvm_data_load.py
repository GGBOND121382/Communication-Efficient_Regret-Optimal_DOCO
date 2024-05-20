import numpy as np
import time
from config_save_load import conf_load


def load_libsvm_data(filename, dimension, datasize=100, to_shuffle=True, is_minus_one=False, dirname='data/data-KDD-Adult/'):
    filename = dirname + filename
    data_target = np.zeros((datasize, dimension + 1))
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

    startTime = time.time()
    X, y = load_libsvm_data(filename, dimension, datasize=datasize, to_shuffle=to_shuffle, is_minus_one=is_minus_one, dirname=dirname)
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

