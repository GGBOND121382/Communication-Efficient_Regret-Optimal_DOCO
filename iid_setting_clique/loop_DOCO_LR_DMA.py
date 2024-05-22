import time
import sys
sys.path.append('../')

from optimization_utils.DMA import *
from data.libsvm_data_load import *
from data.config_save_load import *

def run_comm():

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
        plot_filename = './plot_data_' + filename[:3] + '_' + repr(number_of_clients) + '/DMA_' + filename + '_' + repr(comm_bueget)
        fd = open(plot_filename, 'a')
        loss_mean, class_err_mean, comm_cost = DMA(grad_logReg, data_collection, target_collection, number_of_clients,
                                                   2, comm_bueget, radius=radius, is_l2_norm=True)
        print(f'filename = {filename}, comm_budget = {comm_bueget}, loss_mean = {loss_mean},'
              f' class_err_mean={class_err_mean}, comm_cost = {comm_cost}')

        fd.write(repr(loss_mean) + ' ' + repr(class_err_mean) + ' ' + repr(comm_cost) + '\n')
        fd.close()


def run_time():

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

    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        data_collection, target_collection = load_data_collection_interval_dist(original_data_collection,
                                                                                original_target_collection, dimension,
                                                                                0, running_time_horizon,
                                                                                number_of_clients)
        for comm_bueget in comm_budget_list:
            plot_filename = './plot_data_' + filename[:3] + '_' + repr(number_of_clients) + '/DMA_' + filename + '_' + repr(comm_bueget) + '_' + repr(running_time_horizon)
            fd = open(plot_filename, 'a')
            loss_mean, class_err_mean, comm_cost = DMA(grad_logReg, data_collection, target_collection, number_of_clients,
                                                       2, comm_bueget, radius=radius, is_l2_norm=True)
            print(f'filename = {filename}, comm_budget = {comm_bueget}, loss_mean = {loss_mean},'
                  f' class_err_mean={class_err_mean}, comm_cost = {comm_cost}')

            fd.write(repr(loss_mean) + ' ' + repr(class_err_mean) + ' ' + repr(comm_cost) + '\n')
            fd.close()


# if __name__ == '__main__':
#     run()
