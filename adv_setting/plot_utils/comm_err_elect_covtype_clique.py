import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sys

sys.path.append('../')


def election_adv(number_of_clients, src_dir_name, base_err_list, dst_name, filename="covtype.libsvm.binary"):
    datasize = 581012
    time_horizon = int(datasize / number_of_clients)

    comm_budget_list_TDOCO = [600, 800, ] + [i for i in range(1200, 6000, 600)] + [i for i in
                                                                                   range(6000, 60000, 6000)] + \
                             [i for i in range(60000, 240001, 60000)]

    labels = ["non-pri", "non-comm", "PDOM"]
    color = ["#4285F4", '#fbbc05', 'xkcd:red']
    plt.figure(figsize=(6.5, 3.85))

    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10

    # number_of_clients = 32

    ############################################################
    # TDOCO_delay
    # comm_budget_list_TDOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                          [i for i in range(60000, 120000, 60000)]
    TDOCO_delay_err_mean = []
    TDOCO_delay_err_max = []
    TDOCO_delay_err_min = []
    print("\nDB-TDOCO---------------------------")
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        for comm_budget in comm_budget_list_TDOCO:
            file_name = src_dir_name + 'DB_TDOCO_' + filename + '_' + repr(comm_budget) + '_' + repr(
                running_time_horizon)
            lossmean_list = []
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    rst = [float(str(tmp[0])), str(tmp[1])]
                    lossmean_list.append(float(rst[0]))
                    class_err_list.append(float(rst[1]))
                    # classerr_list.append(float(rst[1]))
            time_class_err_list.append(np.mean(class_err_list))
        index = 0
        while (time_class_err_list[index] > base_err_list[i - 1]):
            # print(time_class_err_list[index])
            index += 1
        # print(index)
        # print("lowest error rates:", np.min(time_class_err_list))
        # index = time_class_err_list.index(np.min(time_class_err_list))
        # print("comm bydget:", comm_budget_list_TDOCO[index])
        TDOCO_delay_err_mean.append(comm_budget_list_TDOCO[index])
    print("comm budget:", TDOCO_delay_err_mean)
    ############################################################

    ############################################################
    # DOCO_gossip
    # comm_budget_list_DOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                         [i for i in range(60000, 300000, 60000)] + [420000, ] + \
    #                         [i for i in range(600000, 1200001, 600000)]
    DOCO_gossip_err_mean = []
    DOCO_gossip_err_max = []
    DOCO_gossip_err_min = []
    print("\ngossip---------------------------")
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        comm_budget_list_DOCO = [int((number_of_clients - 1) * number_of_clients * running_time_horizon / j) + 1 for j
                                 in range(1, 4)]
        for comm_budget in comm_budget_list_DOCO:
            file_name = src_dir_name + 'DOCO_gossip_' + filename + '_' + repr(comm_budget) + '_' + repr(
                running_time_horizon)
            lossmean_list = []
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    rst = [float(str(tmp[0])), str(tmp[1])]
                    lossmean_list.append(float(rst[0]))
                    class_err_list.append(float(rst[1]))
                    # classerr_list.append(float(rst[1]))
            time_class_err_list.append(np.mean(class_err_list))
        # print("lowest error rates:", np.min(time_class_err_list))
        # index = time_class_err_list.index(np.min(time_class_err_list))
        index = len(comm_budget_list_DOCO) - 1
        while (time_class_err_list[index] > base_err_list[i - 1]):
            # print(time_class_err_list[index])
            index -= 1
        # print(index)
        # print("comm bydget:", comm_budget_list_DOCO[index])
        DOCO_gossip_err_mean.append(comm_budget_list_DOCO[index])
        # DOCO_gossip_err_mean.append(np.min(time_class_err_list))
    print("comm budget:", DOCO_gossip_err_mean)
    ############################################################
    ############################################################
    # DBOCG
    # comm_budget_list_DOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                         [i for i in range(60000, 300000, 60000)] + [420000, ] + \
    #                         [i for i in range(600000, 1200001, 600000)]
    DBOCG_err_mean = []
    DBOCG_err_max = []
    DBOCG_err_min = []
    print("\nDBOCG---------------------------")
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        comm_budget_list_DOCO = [int((number_of_clients - 1) * number_of_clients * running_time_horizon / j) + 1 for j
                                 in range(1, 4)]
        for comm_budget in comm_budget_list_DOCO:
            file_name = src_dir_name + 'DBOCG_' + filename + '_' + repr(comm_budget) + '_' + repr(running_time_horizon)
            lossmean_list = []
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    rst = [float(str(tmp[0])), str(tmp[1])]
                    lossmean_list.append(float(rst[0]))
                    class_err_list.append(float(rst[1]))
                    # classerr_list.append(float(rst[1]))
            time_class_err_list.append(np.mean(class_err_list))
        # print("lowest error rates:", np.min(time_class_err_list))
        # index = time_class_err_list.index(np.min(time_class_err_list))
        # print("comm bydget:", comm_budget_list_DOCO[index])
        index = len(comm_budget_list_DOCO) - 1
        while (time_class_err_list[index] > base_err_list[i - 1]):
            # print(time_class_err_list[index])
            index -= 1
        # print(time_class_err_list[index])
        # print(index)
        # print("comm bydget:", comm_budget_list_DOCO[index])
        # DBOCG_err_mean.append(np.min(time_class_err_list))
        DBOCG_err_mean.append(comm_budget_list_DOCO[index])
        # DOCO_gossip_err_mean.append(np.min(time_class_err_list))
    print("comm budget:", DBOCG_err_mean)
    ############################################################
    comm_err_dict = {'base_err_list': base_err_list, 'DB_TDOCO_comm_budget': TDOCO_delay_err_mean,
                     'gossip_comm_budget': DOCO_gossip_err_mean,
                     'DBOCG_comm_budget': DBOCG_err_mean}
    filename_json = dst_name + '.ini'
    with open(filename_json, 'w') as fd:
        json.dump(comm_err_dict, fd)


def election_adv_print(number_of_clients, src_dir_name, filename="covtype.libsvm.binary"):
    datasize = 581012
    time_horizon = int(datasize / number_of_clients)

    comm_budget_list_TDOCO = [600, 800, ] + [i for i in range(1200, 6000, 600)] + [i for i in
                                                                                   range(6000, 60000, 6000)] + \
                             [i for i in range(60000, 240001, 60000)]

    labels = ["non-pri", "non-comm", "PDOM"]
    color = ["#4285F4", '#fbbc05', 'xkcd:red']
    plt.figure(figsize=(6.5, 3.85))

    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10

    ############################################################
    # TDOCO_delay
    # comm_budget_list_TDOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                          [i for i in range(60000, 120000, 60000)]
    TDOCO_delay_err_mean = []
    TDOCO_delay_err_max = []
    TDOCO_delay_err_min = []
    print("\nDB-TDOCO---------------------------")
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        for comm_budget in comm_budget_list_TDOCO:
            file_name = src_dir_name + 'DB_TDOCO_' + filename + '_' + repr(comm_budget) + '_' + repr(
                running_time_horizon)
            lossmean_list = []
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    rst = [float(str(tmp[0])), str(tmp[1])]
                    lossmean_list.append(float(rst[0]))
                    class_err_list.append(float(rst[1]))
                    # classerr_list.append(float(rst[1]))
            time_class_err_list.append(np.mean(class_err_list))
        print("lowest error rates:", np.min(time_class_err_list))
        index = time_class_err_list.index(np.min(time_class_err_list))
        print("comm bydget:", comm_budget_list_TDOCO[index])
        # DMA_err_mean.append(np.mean(class_err_list))
        # DMA_err_max.append(np.max(class_err_list))
        # DMA_err_min.append(np.min(class_err_list))
        TDOCO_delay_err_mean.append(np.min(time_class_err_list))
    ############################################################

    ############################################################
    # DOCO_gossip
    # comm_budget_list_DOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                         [i for i in range(60000, 300000, 60000)] + [420000, ] + \
    #                         [i for i in range(600000, 1200001, 600000)]
    DOCO_gossip_err_mean = []
    DOCO_gossip_err_max = []
    DOCO_gossip_err_min = []
    print("\ngossip---------------------------")
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        comm_budget_list_DOCO = [int((number_of_clients - 1) * number_of_clients * running_time_horizon / j) + 1 for j
                                 in range(1, 4)]
        for comm_budget in comm_budget_list_DOCO:
            file_name = src_dir_name + 'DOCO_gossip_' + filename + '_' + repr(comm_budget) + '_' + repr(
                running_time_horizon)
            lossmean_list = []
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    rst = [float(str(tmp[0])), str(tmp[1])]
                    lossmean_list.append(float(rst[0]))
                    class_err_list.append(float(rst[1]))
                    # classerr_list.append(float(rst[1]))
            time_class_err_list.append(np.mean(class_err_list))
        print("lowest error rates:", np.min(time_class_err_list))
        index = time_class_err_list.index(np.min(time_class_err_list))
        print("comm bydget:", comm_budget_list_DOCO[index])
        DOCO_gossip_err_mean.append(np.min(time_class_err_list))
    ############################################################
    ############################################################
    # DBOCG
    # comm_budget_list_DOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                         [i for i in range(60000, 300000, 60000)] + [420000, ] + \
    #                         [i for i in range(600000, 1200001, 600000)]
    DBOCG_err_mean = []
    DBOCG_err_max = []
    DBOCG_err_min = []
    print("\nDBOCG---------------------------")
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        comm_budget_list_DOCO = [int((number_of_clients - 1) * number_of_clients * running_time_horizon / j) + 1 for j
                                 in range(1, 4)]
        for comm_budget in comm_budget_list_DOCO:
            file_name = src_dir_name + 'DBOCG_' + filename + '_' + repr(comm_budget) + '_' + repr(running_time_horizon)
            lossmean_list = []
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    rst = [float(str(tmp[0])), str(tmp[1])]
                    lossmean_list.append(float(rst[0]))
                    class_err_list.append(float(rst[1]))
                    # classerr_list.append(float(rst[1]))
            time_class_err_list.append(np.mean(class_err_list))
        print("lowest error rates:", np.min(time_class_err_list))
        index = time_class_err_list.index(np.min(time_class_err_list))
        print("comm bydget:", comm_budget_list_DOCO[index])
        DBOCG_err_mean.append(np.min(time_class_err_list))
    ############################################################
    base_err_list = [max(TDOCO_delay_err_mean[i], DOCO_gossip_err_mean[i], DBOCG_err_mean[i]) for i in range(5)]
    print("base_error_list:", base_err_list)
    return base_err_list


if __name__ == '__main__':
    # filename = "covtype.libsvm.binary"

    # src_dir_name = 'plot_data_purify/adv_setting_clique/'
    # dst_name = filename.replace('.', '_') + '_adv'
    # # plot_adv(src_dir_name, dst_name + '.png', format='png')
    # base_err_list = election_adv_print(src_dir_name, dst_name + '_clique.pdf', format='pdf')
    # election_adv(src_dir_name, base_err_list, dst_name, format='pdf')

    # clique network
    src_dir_name = 'plot_data_purify/adv_setting_clique/'
    # dst_name = filename.replace('.', '_') + '_adv'
    # plot_adv(src_dir_name, dst_name + '.png', format='png')
    base_err_list = election_adv_print(32, src_dir_name)
    election_adv(32, src_dir_name, base_err_list, 'adv_setting_clique')

    '''

    src_dir_name = 'plot_data/iid_setting/'
    dst_name = filename.replace('.', '_') + '_iid'
    # plot_stoc(src_dir_name, dst_name + '.png', format='png')
    plot_stoc(src_dir_name, dst_name + '.pdf', format='pdf')

    src_dir_name = 'plot_data/non_iid_setting/'
    dst_name = filename.replace('.', '_') + '_non_iid'
    # plot_stoc(src_dir_name, dst_name + '.png', format='png')
    plot_stoc(src_dir_name, dst_name + '.pdf', format='pdf')
    '''
