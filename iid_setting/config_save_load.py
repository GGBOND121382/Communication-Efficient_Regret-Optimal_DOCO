import json
import math
import sys

sys.path.append('../')
from data.config_save_load import *


if __name__ == '__main__':
    dirname = '../data/iid_data/'
    num_of_clients = 32
    data_filename = 'covtype.libsvm.binary'
    # data_filename = 'epsilon_normalized.all'
    rpt_times = 1

    if(data_filename == 'covtype.libsvm.binary'):
        dimension = 54
        datasize = 581012
        radius = 20
        to_shuffle = True
        is_minus_one = False
    elif(data_filename == 'skin_nonskin.txt'):
        dimension = 3
        datasize = 245057
        radius = 20
        to_shuffle = True
        is_minus_one = False
    elif(data_filename == 'mnist8m.scale'):
        dimension = 784
        datasize = 8100000
        radius = 20
        to_shuffle = True
        is_minus_one = False
    elif(data_filename == 'epsilon_normalized'):
        dimension = 2000
        datasize = 400000
        radius = 20
        to_shuffle = False
        is_minus_one = True
    elif(data_filename == 'epsilon_normalized.t'):
        dimension = 2000
        datasize = 100000
        radius = 20
        to_shuffle = False
        is_minus_one = True
    elif (data_filename == 'epsilon_normalized.all'):
        dimension = 2000
        datasize = 500000
        radius = 20
        to_shuffle = False
        is_minus_one = True
    else:
        assert False
    comm_budget_list = [i for i in range(100, 6000, 100)] + [i for i in range(6000, 12000, 6000)] + [8000, 10000,
                                                                                                     11000] + \
                       [i for i in range(12000, 60000, 6000)] + [i for i in range(60000, 120001, 60000)]
    # comm_budget_list = [100, 200, 400, 600, 800, 1000, 1400, 1800, 2200]  # comm. list for cutting-plane
    conf_save_libsvm_data(dirname, data_filename, dimension, datasize, rpt_times, num_of_clients, radius,
                          comm_budget_list=comm_budget_list, to_shuffle=to_shuffle, is_minus_one=is_minus_one)
    conf_dict = conf_load_libsvm_data()
    print(conf_dict)
