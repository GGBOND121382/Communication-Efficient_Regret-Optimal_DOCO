import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sys
import json

sys.path.append('../')


def election_stoc(num_learners, src_dir_name, base_err_list, dst_name, filename="covtype.libsvm.binary"):
    datasize = 581012
    time_horizon = int(datasize / num_learners)
    comm_budget_list_DMA = [i for i in range(100, 6000, 100)] + [i for i in range(6000, 60000, 1000)] + \
                           [i for i in range(60000, 120001, 10000)] + [i for i in range(180000, 360000, 60000)]

    comm_budget_list_AGD = [i for i in range(100, 6000, 100)] + [i for i in range(6000, 60000, 1000)] + \
                           [i for i in range(60000, 120001, 10000)] + [i for i in range(180000, 360000, 60000)]

    if num_learners == 8:
        comm_budget_list_CP = [100, ] + [i for i in range(200, 1600, 200)] + [i for i in range(1600, 3000, 200)]
    else:
        comm_budget_list_CP = [100, ] + [i for i in range(200, 1600, 200)] + [i for i in range(1600, 8000, 200)]

    labels = ["non-pri", "non-comm", "PDOM"]
    color = ["#4285F4", '#fbbc05', 'xkcd:red']
    plt.figure(figsize=(6.5, 3.85))

    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10
    ############################################################
    # DMA
    print("DMA---------------------------")
    DMA_err_mean = []
    DMA_err_max = []
    DMA_err_min = []
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        for comm_budget in comm_budget_list_DMA:
            if (comm_budget > 120000 and 'clique' not in src_dir_name):
                break
            file_name = src_dir_name + 'DMA_' + filename + '_' + repr(comm_budget) + '_' + repr(running_time_horizon)
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
        # print(time_class_err_list)
        # print("lowest error rates:", np.min(time_class_err_list))
        # base_err_list.append(np.min(time_class_err_list))
        # index = time_class_err_list.index(np.min(time_class_err_list))
        # print("comm bydget:", comm_budget_list_DMA[index])
        index = 0
        while (time_class_err_list[index] > base_err_list[i - 1]):
            # print(time_class_err_list[index])
            index += 1
        DMA_err_mean.append(np.mean(comm_budget_list_DMA[index]))
        # DMA_err_max.append(np.max(class_err_list))
        # DMA_err_min.append(np.min(class_err_list))
    print("comm budget:", DMA_err_mean)
    ############################################################
    ############################################################
    # AGD
    print("\nAGD---------------------------")
    AGD_err_mean = []
    AGD_err_max = []
    AGD_err_min = []
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        comm_upp_bnd = 36000 / 5 * i
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        for comm_budget in comm_budget_list_AGD:
            if (comm_budget > comm_upp_bnd):  # if comm_budget > 36000, the batch size is too small
                break
            file_name = src_dir_name + 'AGD_' + filename + '_' + repr(comm_budget) + '_' + repr(running_time_horizon)
            lossmean_list = []
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    rst = [float(str(tmp[0])), str(tmp[1])]
                    lossmean_list.append(float(rst[0]))
                    class_err_list.append(float(rst[1]))
                    # classerr_list.append(float(rst[1]))
            # time_class_err_list.append(np.mean(class_err_list))
            if (np.isnan(np.mean(class_err_list))):
                time_class_err_list.append(1)
            else:
                time_class_err_list.append(np.mean(class_err_list))
        # print("lowest error rates:", np.min(time_class_err_list))
        # index = time_class_err_list.index(np.min(time_class_err_list))
        index = 0
        while (time_class_err_list[index] > base_err_list[i - 1]):
            # print(time_class_err_list[index])
            index += 1
        AGD_err_mean.append(np.mean(comm_budget_list_AGD[index]))
        # AGD_err_max.append(np.max(class_err_list))
        # AGD_err_min.append(np.min(class_err_list))
    print("comm budget:", AGD_err_mean)
    ############################################################
    ############################################################
    # CP
    print("\nCP---------------------------")
    # comm_budget_list_CP = [100, ] + [i for i in range(200, 1600, 200)] + [i for i in range(1600, 3000, 200)]
    CP_err_mean = []
    CP_err_max = []
    CP_err_min = []
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        for comm_budget in comm_budget_list_CP:
            if (comm_budget > 36000 / 5 * i):  # if comm_budget > 36000, the batch size is too small
                break
            file_name = src_dir_name + 'CP_' + filename + '_' + repr(comm_budget) + '_' + repr(running_time_horizon)
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
        index = 0
        while (time_class_err_list[index] > base_err_list[i - 1]):
            # print(time_class_err_list[index])
            index += 1
        # print("target error rates:", time_class_err_list[index])
        # print("comm bydget:", comm_budget_list_CP[index])
        CP_err_mean.append(comm_budget_list_CP[index])
        # CP_err_max.append(np.max(class_err_list))
        # CP_err_min.append(np.min(class_err_list))
    print("comm budget:", CP_err_mean)
    comm_err_dict = {'base_err_list': base_err_list, 'DMA_comm_budget': DMA_err_mean, 'AGD_comm_budget': AGD_err_mean, 'CP_comm_budget': CP_err_mean}
    filename_json = dst_name + '.ini'
    with open(filename_json, 'w') as fd:
        json.dump(comm_err_dict, fd)


def election_stoc_print(num_learners, src_dir_name, filename="covtype.libsvm.binary"):
    datasize = 581012
    time_horizon = int(datasize / num_learners)
    comm_budget_list_DMA = [i for i in range(100, 6000, 100)] + [i for i in range(6000, 60000, 1000)] + \
                           [i for i in range(60000, 120001, 10000)] + [i for i in range(180000, 360000, 60000)]

    comm_budget_list_AGD = [i for i in range(100, 6000, 100)] + [i for i in range(6000, 60000, 1000)] + \
                           [i for i in range(60000, 120001, 10000)] + [i for i in range(180000, 360000, 60000)]

    if num_learners == 8:
        comm_budget_list_CP = [100, ] + [i for i in range(200, 1600, 200)] + [i for i in range(1600, 3000, 200)]
    else:
        comm_budget_list_CP = [100, ] + [i for i in range(200, 1600, 200)] + [i for i in range(1600, 8000, 200)]

    labels = ["non-pri", "non-comm", "PDOM"]
    color = ["#4285F4", '#fbbc05', 'xkcd:red']
    plt.figure(figsize=(6.5, 3.85))

    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10
    ############################################################
    # DMA
    print("DMA---------------------------")
    DMA_err_mean = []
    DMA_err_max = []
    DMA_err_min = []
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        for comm_budget in comm_budget_list_DMA:
            if (comm_budget > 120000 and 'clique' not in src_dir_name):
                break
            file_name = src_dir_name + 'DMA_' + filename + '_' + repr(comm_budget) + '_' + repr(running_time_horizon)
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
        # print(time_class_err_list)
        print("lowest error rates:", np.min(time_class_err_list))
        index = time_class_err_list.index(np.min(time_class_err_list))
        print("comm bydget:", comm_budget_list_DMA[index])
        # DMA_err_mean.append(np.mean(class_err_list))
        # DMA_err_max.append(np.max(class_err_list))
        # DMA_err_min.append(np.min(class_err_list))
        DMA_err_mean.append(np.min(time_class_err_list))
    ############################################################
    ############################################################
    # AGD
    print("\nAGD---------------------------")
    AGD_err_mean = []
    AGD_err_max = []
    AGD_err_min = []
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        comm_upp_bnd = 36000 / 5 * i
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        for comm_budget in comm_budget_list_AGD:
            if (comm_budget > comm_upp_bnd):  # if comm_budget > 36000, the batch size is too small
                break
            file_name = src_dir_name + 'AGD_' + filename + '_' + repr(comm_budget) + '_' + repr(running_time_horizon)
            lossmean_list = []
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    rst = [float(str(tmp[0])), str(tmp[1])]
                    lossmean_list.append(float(rst[0]))
                    class_err_list.append(float(rst[1]))
                    # classerr_list.append(float(rst[1]))
            if(np.isnan(np.mean(class_err_list))):
                time_class_err_list.append(1)
            else:
                time_class_err_list.append(np.mean(class_err_list))
        print("lowest error rates:", np.min(time_class_err_list))
        index = time_class_err_list.index(np.min(time_class_err_list))
        print("comm bydget:", comm_budget_list_AGD[index])
        # AGD_err_mean.append(np.mean(class_err_list))
        # AGD_err_max.append(np.max(class_err_list))
        # AGD_err_min.append(np.min(class_err_list))
        AGD_err_mean.append(np.min(time_class_err_list))
    ############################################################
    ############################################################
    # CP
    print("\nCP---------------------------")
    # comm_budget_list_CP = [100, ] + [i for i in range(200, 1600, 200)] + [i for i in range(1600, 3000, 200)]
    CP_err_mean = []
    CP_err_max = []
    CP_err_min = []
    unit_time_step = int(time_horizon / 5)
    for i in range(1, 6):
        running_time_horizon = i * unit_time_step
        print("[*] time_horizon:", running_time_horizon)
        time_class_err_list = []
        for comm_budget in comm_budget_list_CP:
            if (comm_budget > 36000 / 5 * i and num_learners == 32):  # if comm_budget > 36000, the batch size is too small
                break
            file_name = src_dir_name + 'CP_' + filename + '_' + repr(comm_budget) + '_' + repr(running_time_horizon)
            lossmean_list = []
            class_err_list = []
            with open(file_name) as fd:
                for line in fd.readlines():
                    tmp = line.split(" ")
                    rst = [float(str(tmp[0])), str(tmp[1])]
                    lossmean_list.append(float(rst[0]))
                    class_err_list.append(float(rst[1]))
                    # classerr_list.append(float(rst[1]))
            # time_class_err_list.append(np.mean(class_err_list))
            if (np.isnan(np.mean(class_err_list))):
                time_class_err_list.append(1)
            else:
                time_class_err_list.append(np.mean(class_err_list))
        print("lowest error rates:", np.min(time_class_err_list))
        index = time_class_err_list.index(np.min(time_class_err_list))
        print("comm bydget:", comm_budget_list_CP[index])
        # CP_err_mean.append(np.mean(class_err_list))
        # CP_err_max.append(np.max(class_err_list))
        # CP_err_min.append(np.min(class_err_list))
        CP_err_mean.append(np.min(time_class_err_list))
    ############################################################
    base_err_list = [max(DMA_err_mean[i], AGD_err_mean[i], CP_err_mean[i]) for i in range(5)]
    print("base_error_list:", base_err_list)
    return base_err_list


# if __name__ == '__main__':
#     filename = "covtype.libsvm.binary"
#
#     '''
#     src_dir_name = 'plot_data/adv_setting/'
#     dst_name = filename.replace('.', '_') + '_adv'
#     # plot_adv(src_dir_name, dst_name + '.png', format='png')
#     plot_adv(src_dir_name, dst_name + '.pdf', format='pdf')
#     '''
#
#
#     # iid
#     src_dir_name = 'plot_data/iid_setting/'
#     dst_name = filename.replace('.', '_') + '_iid'
#     # plot_stoc(src_dir_name, dst_name + '.png', format='png')
#     base_err_list = election_stoc_print(src_dir_name, dst_name + '.pdf', format='iid_setting')
#     election_stoc(src_dir_name, base_err_list, dst_name, format='iid_setting')
#
#
#     src_dir_name = 'plot_data/iid_setting_clique/'
#     dst_name = filename.replace('.', '_') + '_iid'
#     # plot_stoc(src_dir_name, dst_name + '.png', format='png')
#     base_err_list = election_stoc_print(src_dir_name, dst_name + '_clique.pdf', format='iid_setting_clique')
#     election_stoc(src_dir_name, base_err_list, dst_name, format='iid_setting_clique')
#
#
#
#     # non-iid
#     src_dir_name = 'plot_data/non_iid_setting/'
#     dst_name = filename.replace('.', '_') + '_non_iid'
#     # plot_stoc(src_dir_name, dst_name + '.png', format='png')
#     base_err_list = election_stoc_print(src_dir_name, dst_name + '.pdf', format='non_iid_setting')
#     election_stoc(src_dir_name, base_err_list, dst_name, format='non_iid_setting')
#
#     src_dir_name = 'plot_data/non_iid_setting_clique/'
#     dst_name = filename.replace('.', '_') + '_non_iid'
#     # plot_stoc(src_dir_name, dst_name + '.png', format='png')
#     base_err_list = election_stoc_print(src_dir_name, dst_name + '_clique.pdf', format='non_iid_setting_clique')
#     election_stoc(src_dir_name, base_err_list, dst_name, format='non_iid_setting_clique')
#
#
#     '''
#     src_dir_name = 'plot_data/non_iid_setting/'
#     dst_name = filename.replace('.', '_') + '_non_iid'
#     # plot_stoc(src_dir_name, dst_name + '.png', format='png')
#     election_stoc(src_dir_name, dst_name + '.pdf', format='pdf')
#     '''
