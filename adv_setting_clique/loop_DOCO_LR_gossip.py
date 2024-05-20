import time

import sys

sys.path.append('../')

from optimization_utils.gossip import *
from optimization_utils.generate_hyper_cube import *


from config_save_load import conf_load, conf_load_libsvm_data


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
    comm_budget_list = conf_dict['comm_budget_list']
    best_index_list_gossip = conf_dict['best_index_list_gossip']

    startTime = time.time()
    data_collection, target_collection = load_data_collection_from_file(filename, dimension,
                                                                        num_of_clients=number_of_clients,
                                                                        dirname=dirname)
    data_collection = data_collection_preprocess(data_collection)
    endTime = time.time()
    print("Data loaded: " + repr(endTime - startTime) + "seconds")
    print("data shape: ", data_collection[0].shape)
    print("y", target_collection[0][:10, :])
    # print("max ||x||_2:", np.max(np.linalg.norm(data_collection[0], axis=1)))

    for comm_bueget in comm_budget_list:
        # for stepsize_factor in [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]:
        # for stepsize_factor in [10000,]:
        plot_filename = './plot_data/DOCO_gossip_' + filename + '_' + repr(comm_bueget)
        fd = open(plot_filename, 'a')

        # loss_sum = 0
        # class_err_sum = 0
        # for selected_learner in range(number_of_clients):
        # The network topology
        # mat_A = nine_cycle()
        # mat_A = two2three_cycle()
        mat_A = np.ones((number_of_clients, number_of_clients)) / number_of_clients
        '''
        if (number_of_clients == 8):
            mat_A = np.array([[1 / 4, 1 / 4, 1 / 4, 1 / 4, 0, 0, 0, 0], [1 / 4, 1 / 4, 0, 0, 1 / 4, 1 / 4, 0, 0],
                          [1 / 4, 0, 1 / 4, 0, 1 / 4, 0, 1 / 4, 0], [1 / 4, 0, 0, 1 / 4, 0, 1 / 4, 1 / 4, 0],
                          [0, 1 / 4, 1 / 4, 0, 1 / 4, 0, 0, 1 / 4], [0, 1 / 4, 0, 1 / 4, 0, 1 / 4, 0, 1 / 4],
                          [0, 0, 1 / 4, 1 / 4, 0, 0, 1 / 4, 1 / 4], [0, 0, 0, 0, 1 / 4, 1 / 4, 1 / 4, 1 / 4]])
        elif (number_of_clients == 32):
            mat_A = five_cube()
        else:
            print("number of clients should be either 8 or 32")
            assert False
        # print(mat_A[0,7])
        '''

        loss, class_err, comm_cost = gossip(grad_logReg, data_collection, target_collection, number_of_clients,
                                            2, mat_A, comm_bueget, radius=radius, is_l2_norm=True,
                                            selected_learner=-1, is_clique=True,
                                            stepsize_factor=best_index_list_gossip[repr(comm_bueget)])

        for i in range(number_of_clients):
            fd.write(
                repr(loss[i]) + ' ' + repr(class_err[i]) + ' ' + repr(comm_cost) + ' ' + repr(best_index_list_gossip[repr(comm_bueget)]) + '\n')
        print()

        # print(f'filename = {filename}, comm_budget = {comm_bueget}, loss_mean = {loss_mean},'
        #       f' class_err_mean={class_err_mean}, comm_cost = {comm_cost}')

        # fd.write(repr(loss_mean) + ' ' + repr(class_err_mean) + ' ' + repr(comm_cost) + '\n')
        fd.close()


if __name__ == '__main__':
    run()
