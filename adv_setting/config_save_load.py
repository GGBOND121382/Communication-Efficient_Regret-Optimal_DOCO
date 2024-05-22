import json
import math

import sys
sys.path.append('../')

from data.config_save_load import *


# if __name__ == '__main__':
#     dirname = '../data/adv_data/'
#     num_of_clients = 32
#     data_filename = 'covtype.libsvm.binary'
#     rpt_times = 1

def config_data_numLearners_rpt(data_filename, num_of_clients, rpt_times=1):
    dirname = '../data/adv_data/'

    if(data_filename == 'covtype.libsvm.binary' and num_of_clients == 32):
        dimension = 54
        datasize = 581012
        radius = 20
        to_shuffle = True
        is_minus_one = False
        best_index_list_DBOCG = {600: 100.0, 800: 100.0, 1200: 10.0, 1800: 1000.0, 2400: 1000.0, 3000: 100.0, 3600: 1000.0,
                                 4200: 1000.0, 4800: 100000.0, 5400: 1000.0, 6000: 10000.0, 12000: 1000.0, 18000: 100.0, 24000: 100.0,
                                 30000: 100000.0, 36000: 100000.0, 42000: 100000.0, 48000: 100000.0, 54000: 100000.0, 60000: 100000.0,
                                 120000: 10000.0, 180000: 10000.0, 240000: 100000.0, 300000: 100000.0, 360000: 100000.0, 420000: 100000.0,
                                 480000: 100000.0, 540000: 100000.0, 600000: 100000.0, 1200000: 1.0}

        best_index_list_TDOCO = {600: 100.0, 800: 1000.0, 1200: 0.01, 1800: 1000.0, 2400: 1000.0, 3000: 1000.0, 3600: 1000.0,
                                 4200: 1000.0, 4800: 1000.0, 5400: 1000.0, 6000: 1000.0, 12000: 1000.0, 18000: 100.0, 24000: 1000.0,
                                 30000: 1000.0, 36000: 100.0, 42000: 100.0, 48000: 100.0, 54000: 100.0, 60000: 100.0, 120000: 100.0,
                                 180000: 100.0, 240000: 100.0, 300000: 100.0, 360000: 100.0, 420000: 100.0, 480000: 100.0, 540000: 100.0,
                                 600000: 100.0, 1200000: 100.0}

        best_index_list_gossip = {600: 10000.0, 800: 10000.0, 1200: 10000.0, 1800: 10000.0, 2400: 10000.0, 3000: 10000.0,
                                  3600: 100000.0, 4200: 100000.0, 4800: 100000.0, 5400: 100000.0, 6000: 100000.0, 12000: 100000.0,
                                  18000: 100000.0, 24000: 100.0, 30000: 100.0, 36000: 100.0, 42000: 100.0, 48000: 100.0, 54000: 100.0,
                                  60000: 100.0, 120000: 100.0, 180000: 100.0, 240000: 100.0, 300000: 100.0, 360000: 100.0, 420000: 100.0,
                                  480000: 100.0, 540000: 100.0, 600000: 0.01, 1200000: 0.01}



    elif (data_filename == 'epsilon_normalized.all' and num_of_clients == 32):
        dimension = 2000
        datasize = 500000
        radius = 20
        to_shuffle = False
        is_minus_one = True

        best_index_list_DBOCG = {600: 10.0, 800: 1.0, 1200: 10000.0, 1800: 0.1, 2400: 100000.0, 3000: 0.1, 3600: 0.01, 4200: 100000.0,
                                 4800: 0.1, 5400: 1000.0, 6000: 0.01, 12000: 100000.0, 18000: 1000.0, 24000: 100000.0, 30000: 10000.0,
                                 36000: 10000.0, 42000: 10000.0, 48000: 100000.0, 54000: 100000.0, 60000: 10000.0, 120000: 10000.0,
                                 180000: 10000.0, 240000: 100000.0, 300000: 100000.0, 360000: 100000.0, 420000: 100000.0, 480000: 100000.0,
                                 540000: 100000.0, 600000: 100000.0, 1200000: 0.1}

        best_index_list_TDOCO = {600: 100000.0, 800: 1000.0, 1200: 0.01, 1800: 0.1, 2400: 1000.0, 3000: 1000.0, 3600: 1000.0,
                                 4200: 1000.0, 4800: 1000.0, 5400: 1000.0, 6000: 1000.0, 12000: 1000.0, 18000: 1000.0, 24000: 1000.0,
                                 30000: 100.0, 36000: 100.0, 42000: 100.0, 48000: 100.0, 54000: 100.0, 60000: 100.0,
                                 120000: 100.0, 180000: 100.0, 240000: 100.0, 300000: 100.0, 360000: 100.0, 420000: 100.0,
                                 480000: 100.0, 540000: 100.0, 600000: 100.0, 1200000: 100.0}

        best_index_list_gossip = {600: 100000.0, 800: 100000.0, 1200: 100000.0, 1800: 100000.0, 2400: 100000.0, 3000: 1.0,
                                  3600: 0.01, 4200: 100000.0, 4800: 100000.0, 5400: 100000.0, 6000: 100000.0, 12000: 100000.0,
                                  18000: 100000.0, 24000: 10000.0, 30000: 10000.0, 36000: 10000.0, 42000: 10000.0, 48000: 100000.0,
                                  54000: 100000.0, 60000: 100000.0, 120000: 100000.0, 180000: 10000.0, 240000: 10000.0, 300000: 10000.0,
                                  360000: 10000.0, 420000: 10000.0, 480000: 10000.0, 540000: 10000.0, 600000: 10000.0, 1200000: 0.01}

    elif (data_filename == 'covtype.libsvm.binary' and num_of_clients == 8):
        dimension = 54
        datasize = 581012
        radius = 20
        to_shuffle = True
        is_minus_one = False
        best_index_list_DBOCG = {800: 10000.0, 600: 1000.0, 1200: 1000.0, 1800: 100.0, 2400: 100.0, 3000: 10000.0,
                                 3600: 10000.0, 4200: 10000.0, 4800: 10000.0, 5400: 100000.0, 6000: 10000.0,
                                 12000: 0.01, 18000: 0.1, 24000: 1.0, 30000: 1.0, 36000: 1.0, 42000: 100000.0,
                                 48000: 100000.0, 54000: 10.0, 60000: 10.0, 120000: 100.0, 180000: 100.0, 240000: 100.0,
                                 300000: 1000.0, 360000: 1000.0, 420000: 1000.0, 480000: 1000.0, 540000: 1000.0,
                                 600000: 1000.0, 1200000: 1000.0}

        best_index_list_TDOCO = {800: 1000.0, 600: 1000.0, 1200: 1000.0, 1800: 1000.0, 2400: 1000.0, 3000: 1000.0,
                                 3600: 1000.0, 4200: 1000.0, 4800: 1000.0, 5400: 1000.0, 6000: 1000.0, 12000: 1000.0,
                                 18000: 1000.0, 24000: 1000.0, 30000: 1000.0, 36000: 1000.0, 42000: 1000.0,
                                 48000: 1000.0, 54000: 1000.0, 60000: 1000.0, 120000: 1000.0, 180000: 1000.0,
                                 240000: 1000.0, 300000: 1000.0, 360000: 1000.0, 420000: 1000.0, 480000: 1000.0,
                                 540000: 1000.0, 600000: 1000.0, 1200000: 1000.0}

        best_index_list_gossip = {800: 100000.0, 600: 100000.0, 1200: 100000.0, 1800: 10000.0, 2400: 10000.0,
                                  3000: 10000.0, 3600: 10000.0, 4200: 10000.0, 4800: 10000.0, 5400: 10000.0, 6000: 0.01,
                                  12000: 0.01, 18000: 1.0, 24000: 1.0, 30000: 1.0, 36000: 1.0, 42000: 10.0, 48000: 10.0,
                                  54000: 10.0, 60000: 10.0, 120000: 10.0, 180000: 100.0, 240000: 100.0, 300000: 100.0,
                                  360000: 100.0, 420000: 100.0, 480000: 100.0, 540000: 100.0, 600000: 100.0,
                                  1200000: 100.0}


    elif (data_filename == 'epsilon_normalized.all' and num_of_clients == 8):
        dimension = 2000
        datasize = 500000
        radius = 20
        to_shuffle = False
        is_minus_one = True

        best_index_list_DBOCG = {800: 1000.0, 600: 1.0, 1200: 100000.0, 1800: 100000.0, 2400: 10000.0, 3000: 100000.0,
                                 3600: 10000.0, 4200: 100000.0, 4800: 100.0, 5400: 100.0, 6000: 100.0, 12000: 1000.0,
                                 18000: 10000.0, 24000: 100000.0, 30000: 10000.0, 36000: 100000.0, 42000: 100000.0,
                                 48000: 100000.0, 54000: 10000.0, 60000: 100000.0, 120000: 1.0, 180000: 1.0,
                                 240000: 1.0, 300000: 0.01, 360000: 1000.0, 420000: 1000.0, 480000: 1000.0,
                                 540000: 1000.0, 600000: 1000.0, 1200000: 1000.0}

        best_index_list_TDOCO = {800: 1000.0, 600: 100.0, 1200: 1000.0, 1800: 1000.0, 2400: 1000.0, 3000: 1000.0,
                                 3600: 1000.0, 4200: 1000.0, 4800: 1000.0, 5400: 1000.0, 6000: 1000.0, 12000: 1000.0,
                                 18000: 1000.0, 24000: 1000.0, 30000: 1000.0, 36000: 1000.0, 42000: 1000.0,
                                 48000: 1000.0, 54000: 1000.0, 60000: 1000.0, 120000: 1000.0, 180000: 1000.0,
                                 240000: 1000.0, 300000: 1000.0, 360000: 1000.0, 420000: 1000.0, 480000: 1000.0,
                                 540000: 1000.0, 600000: 1000.0, 1200000: 1000.0}

        best_index_list_gossip = {800: 100000.0, 600: 100000.0, 1200: 100000.0, 1800: 100000.0, 2400: 100000.0,
                                  3000: 100000.0, 3600: 100000.0, 4200: 100000.0, 4800: 100000.0, 5400: 100000.0,
                                  6000: 100000.0, 12000: 100000.0, 18000: 100000.0, 24000: 100000.0, 30000: 100000.0,
                                  36000: 1.0, 42000: 1.0, 48000: 1.0, 54000: 0.1, 60000: 0.1, 120000: 0.1, 180000: 0.1,
                                  240000: 0.1, 300000: 0.01, 360000: 100.0, 420000: 100.0, 480000: 100.0, 540000: 100.0,
                                  600000: 100.0, 1200000: 100.0}

    else:
        assert False

    comm_budget_list = [i for i in range(600, 6000, 600)] + [i for i in range(6000, 60000, 6000)] +\
                       [i for i in range(60000, 600000, 60000)] + [i for i in range(600000, 1200001, 600000)] + [800, ]
    # comm_budget_list = [100, 200, 400, 600, 800, 1000, 1400, 1800, 2200]  # comm. list for cutting-plane
    print("comm_budget_list:", comm_budget_list)
    conf_save_libsvm_data(dirname, data_filename, dimension, datasize, rpt_times, num_of_clients, radius,
                          comm_budget_list, to_shuffle, is_minus_one, best_index_list_gossip=best_index_list_gossip, best_index_list_TDOCO=best_index_list_TDOCO, best_index_list_DBOCG=best_index_list_DBOCG)
    conf_dict = conf_load_libsvm_data()
    print(conf_dict)
