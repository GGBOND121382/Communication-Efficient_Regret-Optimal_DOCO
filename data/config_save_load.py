import json
import math


def conf_load(filename='conf.ini'):
    with open(filename) as fd:
        conf_dict = json.load(fd)
    return conf_dict


def conf_save_libsvm_data(dirname, data_filename, dimension, datasize, rpt_times, number_of_clients,
                          radius, comm_budget_list=[], to_shuffle=True, is_minus_one=False, best_index_list_gossip={}, best_index_list_TDOCO={}, best_index_list_DBOCG={}, filename='conf.ini'):
    with open(filename, 'w') as fd:
        conf_dict = {'dirname': dirname, 'data': data_filename, 'dimension': dimension, 'datasize': datasize,
                     'rpt_times': rpt_times, 'number_of_clients': number_of_clients, 'radius': radius,
                     'comm_budget_list': comm_budget_list, 'to_shuffle': to_shuffle, 'is_minus_one': is_minus_one, 'best_index_list_gossip': best_index_list_gossip, 'best_index_list_TDOCO': best_index_list_TDOCO, 'best_index_list_DBOCG': best_index_list_DBOCG}
        json.dump(conf_dict, fd)


def conf_load_libsvm_data(filename='conf.ini'):
    with open(filename) as fd:
        conf_dict = json.load(fd)
    return conf_dict

# if __name__ == '__main__':
#     dirname = '../data/original_data/'
#     num_of_clients = 32
#     data_filename = 'covtype.libsvm.binary'
#     # data_filename = 'epsilon_normalized.all'
#     rpt_times = 1


def config_data_numLearners_rpt(data_filename, num_of_clients, rpt_times=1):
    dirname = '../data/original_data/'
    if(data_filename == 'covtype.libsvm.binary'):
        dimension = 54
        datasize = 581012
        radius = 20
        to_shuffle = True
        is_minus_one = False
    elif(data_filename == 'epsilon_normalized'):
        dimension = 2000
        datasize = 400000
        radius = 20
        to_shuffle = True
        is_minus_one = True
    elif(data_filename == 'epsilon_normalized.t'):
        dimension = 2000
        datasize = 100000
        radius = 20
        to_shuffle = True
        is_minus_one = True
    elif (data_filename == 'epsilon_normalized.all'):
        dimension = 2000
        datasize = 500000
        radius = 20
        to_shuffle = True
        is_minus_one = True
    else:
        assert False
    conf_save_libsvm_data(dirname, data_filename, dimension, datasize, rpt_times, num_of_clients, radius,
                          to_shuffle=to_shuffle, is_minus_one=is_minus_one)
    conf_dict = conf_load_libsvm_data()
    print(conf_dict)
