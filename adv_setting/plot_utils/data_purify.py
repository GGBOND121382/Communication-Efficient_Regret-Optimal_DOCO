import os
import shutil
import math
import sys

import numpy as np


def index_selection(num_learners):
    suffix_list = ['', ]
    prefix = 'plot_data_cov_' + repr(num_learners)
    
    datasize = 581012
    time_horizon = int(datasize / num_learners)
    number_of_clients = num_learners
    
    comm_budget_list_TDOCO = [800, ] + [i for i in range(600, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
                             [i for i in range(60000, 240001, 60000)]
    
    # comm_budget_list_AGD = [i for i in range(100, 1200, 100)] + [i for i in range(1200, 6000, 200)] + [i for i in
    #                                                                                                    range(6000,
    #                                                                                                          36000,
    #                                                                                                       6000)]
    
    filename = "covtype.libsvm.binary"
    # merge_dir_name = 'plot_utils/plot_data_purify_' + repr(num_learners)
    
    index_list = []
    
    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting', ]:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            # dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                comm_budget_list_DOCO = [int(2 * number_of_clients * running_time_horizon / j) + 1 for j in range(1, 4)]
                for comm_budget in comm_budget_list_DOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    all_err_list = [[] for i in range(num_learners)]
                    file_name = src_dir_name + 'DOCO_gossip_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    # dst_file_name = dst_dir_name + 'DOCO_gossip_' + filename + '_' + repr(comm_budget) + '_' + repr(
                    #     running_time_horizon)
                    # fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            index = int(np.around(math.log(index, 10))) + 2
                            all_err_list[index].append(float(rst[1]))
                            # print(index)
                            # fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(num_learners)])
                    avg_err_list = [np.mean(all_err_list[i]) for i in range(num_learners)]
                    best_index = avg_err_list.index(min(avg_err_list))
                    index_list.append(10 ** (best_index - 2))
                    # fd_w.close()
    
    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting_clique']:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            # dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                comm_budget_list_DOCO = [int((number_of_clients - 1) * number_of_clients * running_time_horizon / j) + 1 for
                                         j in range(1, 4)]
                for comm_budget in comm_budget_list_DOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    all_err_list = [[] for i in range(num_learners)]
                    file_name = src_dir_name + 'DOCO_gossip_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    # dst_file_name = dst_dir_name + 'DOCO_gossip_' + filename + '_' + repr(comm_budget) + '_' + repr(
                    #     running_time_horizon)
                    # fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            index = int(np.around(math.log(index, 10))) + 2
                            all_err_list[index].append(float(rst[1]))
                            # print(index)
                            # fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(num_learners)])
                    avg_err_list = [np.mean(all_err_list[i]) for i in range(num_learners)]
                    best_index = avg_err_list.index(min(avg_err_list))
                    index_list.append(10 ** (best_index - 2))
                    # fd_w.close()
                    # print(gossip_avg_err_list_clique)
                    # print(index_list)
    
    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting', ]:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            # dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                comm_budget_list_DOCO = [int(2 * number_of_clients * running_time_horizon / j) + 1 for j in range(1, 4)]
                for comm_budget in comm_budget_list_DOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    all_err_list = [[] for i in range(num_learners)]
                    file_name = src_dir_name + 'DBOCG_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    # dst_file_name = dst_dir_name + 'DBOCG_' + filename + '_' + repr(comm_budget) + '_' + repr(
                    #     running_time_horizon)
                    # fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            index = int(np.around(math.log(index, 10))) + 2
                            all_err_list[index].append(float(rst[1]))
                            # print(index)
                            # fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(num_learners)])
                    avg_err_list = [np.mean(all_err_list[i]) for i in range(num_learners)]
                    best_index = avg_err_list.index(min(avg_err_list))
                    index_list.append(10 ** (best_index - 2))
                    # fd_w.close()
    
    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting_clique']:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            # dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                comm_budget_list_DOCO = [int((number_of_clients - 1) * number_of_clients * running_time_horizon / j) + 1 for
                                         j in range(1, 4)]
                for comm_budget in comm_budget_list_DOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    all_err_list = [[] for i in range(num_learners)]
                    file_name = src_dir_name + 'DBOCG_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    # dst_file_name = dst_dir_name + 'DBOCG_' + filename + '_' + repr(comm_budget) + '_' + repr(
                    #     running_time_horizon)
                    # fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            index = int(np.around(math.log(index, 10))) + 2
                            all_err_list[index].append(float(rst[1]))
                            # print(index)
                            # fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(num_learners)])
                    avg_err_list = [np.mean(all_err_list[i]) for i in range(num_learners)]
                    best_index = avg_err_list.index(min(avg_err_list))
                    index_list.append(10 ** (best_index - 2))
                    # fd_w.close()
    
    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting', ]:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            # dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                for comm_budget in comm_budget_list_TDOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    all_err_list = [[] for i in range(num_learners)]
                    file_name = src_dir_name + 'DB_TDOCO_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    # dst_file_name = dst_dir_name + 'DB_TDOCO_' + filename + '_' + repr(comm_budget) + '_' + repr(
                    #     running_time_horizon)
                    # fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            index = int(np.around(math.log(index, 10))) + 2
                            all_err_list[index].append(float(rst[1]))
                            # print(index)
                            # fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(num_learners)])
                    avg_err_list = [np.mean(all_err_list[i]) for i in range(num_learners)]
                    best_index = avg_err_list.index(min(avg_err_list))
                    index_list.append(10 ** (best_index - 2))
                    # fd_w.close()
    
    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting_clique']:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            # dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                for comm_budget in comm_budget_list_TDOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    all_err_list = [[] for i in range(num_learners)]
                    file_name = src_dir_name + 'DB_TDOCO_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    # dst_file_name = dst_dir_name + 'DB_TDOCO_' + filename + '_' + repr(comm_budget) + '_' + repr(
                    #     running_time_horizon)
                    # fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            index = int(np.around(math.log(index, 10))) + 2
                            all_err_list[index].append(float(rst[1]))
                            # print(index)
                            # fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(num_learners)])
                    avg_err_list = [np.mean(all_err_list[i]) for i in range(num_learners)]
                    best_index = avg_err_list.index(min(avg_err_list))
                    index_list.append(10 ** (best_index - 2))
                    # fd_w.close()
    
    # print(index_list)
    return index_list


def data_purify(number_of_clients):
    index_list = index_selection(number_of_clients)
    suffix_list = ['', ]
    prefix = 'plot_data_cov_' + repr(number_of_clients)

    datasize = 581012
    time_horizon = int(datasize / number_of_clients)

    for filepath in ['iid_setting', 'non_iid_setting', 'adv_setting', 'iid_setting_clique', 'non_iid_setting_clique',
                     'adv_setting_clique']:
        if os.path.exists('plot_utils/plot_data_purify_' + repr(number_of_clients) + '/' + filepath):
            shutil.rmtree('plot_utils/plot_data_purify_' + repr(number_of_clients) + '/' + filepath)
        os.mkdir('plot_utils/plot_data_purify_' + repr(number_of_clients) + '/' + filepath)

    ######################### covtype

    # comm_budget_list_DOCO = [800, ] + [i for i in range(600, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                         [i for i in range(60000, 600000, 60000)] + [i for i in range(600000, 4800001, 600000)]

    comm_budget_list_TDOCO = [800, ] + [i for i in range(600, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
                             [i for i in range(60000, 240001, 60000)]

    # comm_budget_list_AGD = [i for i in range(100, 1200, 100)] + [i for i in range(1200, 6000, 200)] + [i for i in
    #                                                                                                    range(6000,
    #                                                                                                          36000,
    #                                                                                                       6000)]

    filename = "covtype.libsvm.binary"
    merge_dir_name = 'plot_utils/plot_data_purify_' + repr(number_of_clients)

    cnt = 0

    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting', ]:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                comm_budget_list_DOCO = [int(2 * number_of_clients * running_time_horizon / j) + 1 for j in range(1, 4)]
                for comm_budget in comm_budget_list_DOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    file_name = src_dir_name + 'DOCO_gossip_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    dst_file_name = dst_dir_name + 'DOCO_gossip_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            # print(index)
                            if (index == index_list[cnt]):
                                fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(32)])
                    fd_w.close()
                    cnt += 1

    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting_clique']:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                comm_budget_list_DOCO = [int((number_of_clients - 1) * number_of_clients * running_time_horizon / j) + 1
                                         for j in range(1, 4)]
                for comm_budget in comm_budget_list_DOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    file_name = src_dir_name + 'DOCO_gossip_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    dst_file_name = dst_dir_name + 'DOCO_gossip_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            # print(index)
                            if (index == index_list[cnt]):
                                fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(32)])
                    fd_w.close()
                    cnt += 1

    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting', ]:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                comm_budget_list_DOCO = [int(2 * number_of_clients * running_time_horizon / j) + 1 for j in range(1, 4)]
                for comm_budget in comm_budget_list_DOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    file_name = src_dir_name + 'DBOCG_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    dst_file_name = dst_dir_name + 'DBOCG_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            # print(index)
                            if (index == index_list[cnt]):
                                fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(32)])
                    fd_w.close()
                    cnt += 1

    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting_clique']:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                comm_budget_list_DOCO = [int((number_of_clients - 1) * number_of_clients * running_time_horizon / j) + 1
                                         for j in range(1, 4)]
                for comm_budget in comm_budget_list_DOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    file_name = src_dir_name + 'DBOCG_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    dst_file_name = dst_dir_name + 'DBOCG_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            # print(index)
                            if (index == index_list[cnt]):
                                fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(32)])
                    fd_w.close()
                    cnt += 1

    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting', ]:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                for comm_budget in comm_budget_list_TDOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    file_name = src_dir_name + 'DB_TDOCO_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    dst_file_name = dst_dir_name + 'DB_TDOCO_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            # print(index)
                            if (index == index_list[cnt]):
                                fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(32)])
                    fd_w.close()
                    cnt += 1

    for suffix in suffix_list:
        dir_name = prefix + suffix
        for setting in ['adv_setting_clique']:
            src_dir_name = '../' + setting + '/' +dir_name + '/'
            dst_dir_name = merge_dir_name + '/' + setting + '/'
            unit_time_step = int(time_horizon / 5)
            for i in range(1, 6):
                running_time_horizon = i * unit_time_step
                for comm_budget in comm_budget_list_TDOCO:
                    if (comm_budget >= 1800000 and 'clique' not in setting):
                        break
                    file_name = src_dir_name + 'DB_TDOCO_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    dst_file_name = dst_dir_name + 'DB_TDOCO_' + filename + '_' + repr(comm_budget) + '_' + repr(
                        running_time_horizon)
                    fd_w = open(dst_file_name, 'a')
                    lossmean_list = []
                    class_err_list = []
                    with open(file_name) as fd:
                        for line in fd.readlines():
                            tmp = line.split(" ")
                            rst = [float(str(tmp[0])), str(tmp[1]), tmp[2], tmp[3]]
                            lossmean_list.append(float(rst[0]))
                            class_err_list.append(float(rst[1]))
                            index = float(tmp[3])
                            # print(index)
                            if (index == index_list[cnt]):
                                fd_w.write(line)
                    # print([len(gossip_all_err_list_clique[i]) for i in range(32)])
                    fd_w.close()
                    cnt += 1
    print(cnt)
    print(len(index_list))
