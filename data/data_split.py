import sys
sys.path.append('../')

import numpy as np
from data.config_save_load import *


def iid_data_split(filename, data, num_of_clients=8, dirname='iid_data/', to_shuffle=True):
    if (to_shuffle):
        np.random.shuffle(data)
    m_samples, _ = data.shape
    data_of_client = int(np.floor(m_samples / num_of_clients))
    for i in range(num_of_clients):
        client_filename = filename + '_client' + repr(i)
        client_data = data[i * data_of_client:(i + 1) * data_of_client, :]
        np.save(dirname + client_filename + '_data.npy', client_data[:, :-1])
        np.save(dirname + client_filename + '_target.npy', client_data[:, -1])


def non_iid_data_split(filename, data, num_of_clients=8, dirname='non_iid_data/', to_shuffle=True):
    positive_samples = data[data[:, -1] == 1]
    negative_samples = data[data[:, -1] == 0]
    sorted_data = np.vstack((positive_samples, negative_samples))
    m_samples, _ = sorted_data.shape
    data_of_client = int(np.floor(m_samples / num_of_clients))
    for i in range(num_of_clients):
        client_filename = filename + '_client' + repr(i)
        client_data = sorted_data[i * data_of_client:(i + 1) * data_of_client, :]
        if (to_shuffle):
            np.random.shuffle(client_data)
        np.save(dirname + client_filename + '_data.npy', client_data[:, :-1])
        np.save(dirname + client_filename + '_target.npy', client_data[:, -1])


'''
def adv_data_split(filename, data, num_of_clients=8, dirname='adv_data/', kb=1000, kr=50,
                   non_iid_dirname='non_iid_data/', to_shuffle=True):
    _, dimension = data.shape
    for i in range(num_of_clients):
        client_filename = filename + '_client' + repr(i)
        client_data = np.zeros((kb * kr, dimension))
        non_iid_client_data_data = np.load(non_iid_dirname + client_filename + '_data.npy')
        non_iid_client_data_target = np.load(non_iid_dirname + client_filename + '_target.npy')
        non_iid_client_data_target = np.reshape(non_iid_client_data_target, (-1, 1))
        non_iid_client_data = np.hstack((non_iid_client_data_data, non_iid_client_data_target))
        if (to_shuffle):
            np.random.shuffle(non_iid_client_data)
        for t in range(kb):
            data_record = non_iid_client_data[t, :]
            if (t % 2 == 1):
                data_record[-1] = 1 - data_record[-1]
            client_data[t * kr:(t + 1) * kr, :] = data_record
            np.save(dirname + client_filename + '_data.npy', client_data[:, :-1])
            np.save(dirname + client_filename + '_target.npy', client_data[:, -1])
'''


def adv_data_split(filename, data, num_of_clients=8, dirname='adv_data/', kb=3,
                   non_iid_dirname='non_iid_data/', to_shuffle=True):
    m_samples, dimension = data.shape
    data_of_client = int(np.floor(m_samples / num_of_clients))
    # print("adv, data_of_client:", data_of_client)
    len_interval = int(np.ceil(data_of_client / kb))
    for i in range(num_of_clients):
        client_filename = filename + '_client' + repr(i)
        client_data = np.zeros((data_of_client, dimension))
        non_iid_client_data_data = np.load(non_iid_dirname + client_filename + '_data.npy')
        non_iid_client_data_target = np.load(non_iid_dirname + client_filename + '_target.npy')
        non_iid_client_data_target = np.reshape(non_iid_client_data_target, (-1, 1))
        non_iid_client_data = np.hstack((non_iid_client_data_data, non_iid_client_data_target))
        if (to_shuffle):
            np.random.shuffle(non_iid_client_data)
        for t in range(kb):
            data_record = non_iid_client_data[t*len_interval:(t+1)*len_interval, :]
            if (t % 2 == 1):
                data_record[:, -1] = 1 - data_record[:, -1]
            client_data[t * len_interval:(t + 1) * len_interval, :] = data_record
            np.save(dirname + client_filename + '_data.npy', client_data[:, :-1])
            np.save(dirname + client_filename + '_target.npy', client_data[:, -1])

def data_split(filename, feedback_setting='non_iid', num_of_clients=8, dirname='original_data/', to_shuffle=True):
    X = np.load(dirname + filename + '_data.npy')
    y = np.load(dirname + filename + '_target.npy')
    y = np.reshape(y, (-1, 1))
    # print(f'X shape: {X.shape}; y shape: {y.shape}')
    data = np.hstack((X, y))
    if (feedback_setting == 'all'):
        iid_data_split(filename, data, num_of_clients)
        non_iid_data_split(filename, data, num_of_clients)
        adv_data_split(filename, data, num_of_clients, to_shuffle=False)
    elif (feedback_setting == 'non_iid'):
        non_iid_data_split(filename, data, num_of_clients, to_shuffle=to_shuffle)
    elif (feedback_setting == 'iid'):
        iid_data_split(filename, data, num_of_clients, to_shuffle=to_shuffle)
    elif (feedback_setting == 'adv'):
        adv_data_split(filename, data, num_of_clients, to_shuffle=to_shuffle)


def run(feedback_setting='all'):
    conf_dict = conf_load()
    dirname = conf_dict['dirname']
    filename = conf_dict['data']
    dimension = conf_dict['dimension']
    datasize = conf_dict['datasize']
    rpt_times = conf_dict['rpt_times']
    number_of_clients = conf_dict['number_of_clients']
    print("num of clients", number_of_clients)
    to_shuffle = conf_dict['to_shuffle']
    is_minus_one = conf_dict['is_minus_one']
    data_split(filename, feedback_setting=feedback_setting, num_of_clients=number_of_clients, dirname=dirname, to_shuffle=True)


if __name__ == '__main__':
    run()
