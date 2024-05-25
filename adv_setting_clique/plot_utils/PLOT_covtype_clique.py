import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sys

sys.path.append('../../')


def plot_adv(src_dir_name, dst_name, format='png'):
    comm_budget_list_DOCO = [600, 800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
                            [i for i in range(60000, 600000, 60000)] + [i for i in range(600000, 4800001, 600000)]

    comm_budget_list_TDOCO = [600, 800, ] + [i for i in range(1200, 6000, 600)] + [i for i in
                                                                                   range(6000, 60000, 6000)] + \
                             [i for i in range(60000, 240001, 60000)]

    comm_budget_list_DMA = [i for i in range(100, 6000, 100)] + [i for i in range(6000, 60000, 6000)] + \
                           [i for i in range(60000, 120001, 60000)]

    comm_budget_list_AGD = [i for i in range(100, 6000, 100)] + [i for i in range(6000, 36000, 6000)]
    comm_budget_list_CP = [100, 200, 400, 600, 800, 1000, 1400, 1800, 2200]

    labels = ["non-pri", "non-comm", "PDOM"]
    color = ["#4285F4", '#fbbc05', 'xkcd:red']
    plt.figure(figsize=(6.5, 3.85))

    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10
    ############################################################
    # DOCO_gossip
    # comm_budget_list_DOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                         [i for i in range(60000, 300000, 60000)] + [420000, ] + \
    #                         [i for i in range(600000, 1200001, 600000)]
    DOCO_gossip_err_mean = []
    DOCO_gossip_err_max = []
    DOCO_gossip_err_min =[]
    for comm_budget in comm_budget_list_DOCO:
        file_name = src_dir_name + 'DOCO_gossip_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        DOCO_gossip_err_max.append(np.max(class_err_list))
        DOCO_gossip_err_min.append(np.min(class_err_list))
        DOCO_gossip_err_mean.append(np.mean(class_err_list))
    print(DOCO_gossip_err_max)
    print(DOCO_gossip_err_min)
    ############################################################
    ############################################################
    # DBOCG
    # comm_budget_list_DOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                         [i for i in range(60000, 300000, 60000)] + [420000, ] + \
    #                         [i for i in range(600000, 1200001, 600000)]
    DBOCG_err_mean = []
    DBOCG_err_max = []
    DBOCG_err_min = []
    for comm_budget in comm_budget_list_DOCO:
        file_name = src_dir_name + 'DBOCG_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        DBOCG_err_max.append(np.max(class_err_list))
        DBOCG_err_min.append(np.min(class_err_list))
        DBOCG_err_mean.append(np.mean(class_err_list))
    print(DBOCG_err_max)
    print(DBOCG_err_min)
    ############################################################
    ############################################################
    # TDOCO_delay
    # comm_budget_list_TDOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                          [i for i in range(60000, 120000, 60000)]
    DB_TDOCO_err_mean = []
    DB_TDOCO_err_max = []
    DB_TDOCO_err_min = []
    for comm_budget in comm_budget_list_TDOCO:
        file_name = src_dir_name + 'DB_TDOCO_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        DB_TDOCO_err_max.append(np.max(class_err_list))
        DB_TDOCO_err_min.append(np.min(class_err_list))
        DB_TDOCO_err_mean.append(np.mean(class_err_list))
    ############################################################

    plt.plot(comm_budget_list_DOCO, DOCO_gossip_err_mean, ':', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[1], label='gossip')

    plt.fill_between(comm_budget_list_DOCO, DOCO_gossip_err_max, DOCO_gossip_err_min,  # 上限，下限
                     facecolor=color[1],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    plt.plot(comm_budget_list_DOCO, DBOCG_err_mean, '--', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[0], label='D-BOCG')

    plt.fill_between(comm_budget_list_DOCO, DBOCG_err_max, DBOCG_err_min,  # 上限，下限
                     facecolor=color[0],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    plt.plot(comm_budget_list_TDOCO, DB_TDOCO_err_mean, '-.', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[2], label='DB-TDOCO')
    plt.fill_between(comm_budget_list_TDOCO, DB_TDOCO_err_max, DB_TDOCO_err_min,  # 上限，下限
                     facecolor=color[2],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    plt.xscale('log')
    plt.legend(fontsize=23)
    plt.tick_params(labelsize=23)
    # plt.title("Regret of POCO(L)")

    y_major_locator = MultipleLocator(0.05)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel("Communication Budget", fontsize=23)
    plt.ylabel("Error Rate", fontsize=23)
    fig = plt.gcf()

    # fig.savefig("fig_regret_loss_cube_conv.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(dst_name, format=format, bbox_inches="tight")
    plt.show()


def plot_stoc(src_dir_name, dst_name, format='png'):
    comm_budget_list_DOCO = [i for i in range(600, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
                            [i for i in range(60000, 300000, 60000)] + [420000, ] + \
                            [i for i in range(600000, 3000001, 600000)]

    comm_budget_list_TDOCO = [i for i in range(600, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
                             [i for i in range(60000, 240001, 60000)]

    comm_budget_list_DMA = [i for i in range(100, 6000, 100)] + [i for i in range(6000, 60000, 6000)] + \
                           [i for i in range(60000, 120001, 60000)]

    comm_budget_list_AGD = [i for i in range(100, 1200, 100)] + [i for i in range(1200, 6000, 200)] + [i for i in
                                                                                                       range(6000,
                                                                                                             36000,
                                                                                                             6000)]

    comm_budget_list_CP = [100, ] + [i for i in range(200, 1600, 200)] + [i for i in range(1600, 3000, 200)]

    labels = ["non-pri", "non-comm", "PDOM"]
    color = ["#4285F4", '#fbbc05', 'xkcd:red']
    plt.figure(figsize=(6.5, 3.85))

    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10
    ############################################################
    # DMA
    DMA_err_mean = []
    DMA_err_max = []
    DMA_err_min = []
    for comm_budget in comm_budget_list_DMA:
        file_name = src_dir_name + 'DMA_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        DMA_err_mean.append(np.mean(class_err_list))
        DMA_err_max.append(np.max(class_err_list))
        DMA_err_min.append(np.min(class_err_list))
    ############################################################
    ############################################################
    # AGD
    AGD_err_mean = []
    AGD_err_max = []
    AGD_err_min = []
    for comm_budget in comm_budget_list_AGD:
        file_name = src_dir_name + 'AGD_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        AGD_err_mean.append(np.mean(class_err_list))
        AGD_err_max.append(np.max(class_err_list))
        AGD_err_min.append(np.min(class_err_list))
    ############################################################
    ############################################################
    # CP
    # comm_budget_list_CP = [100, ] + [i for i in range(200, 1600, 200)] + [i for i in range(1600, 3000, 200)]
    CP_err_mean = []
    CP_err_max = []
    CP_err_min = []
    for comm_budget in comm_budget_list_CP:
        file_name = src_dir_name + 'CP_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        CP_err_mean.append(np.mean(class_err_list))
        CP_err_max.append(np.max(class_err_list))
        CP_err_min.append(np.min(class_err_list))
    ############################################################

    plt.plot(comm_budget_list_DMA, DMA_err_mean, ':', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[1], label='$DMA$')
    plt.fill_between(comm_budget_list_DMA, DMA_err_max, DMA_err_min,  # 上限，下限
                     facecolor=color[1],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    plt.plot(comm_budget_list_AGD, AGD_err_mean, '--', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[0], label='$DB2O_a$')
    plt.fill_between(comm_budget_list_AGD, AGD_err_max, AGD_err_min,  # 上限，下限
                     facecolor=color[0],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度


    plt.plot(comm_budget_list_CP, CP_err_mean, '-.', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[2], label='$DB2O_c$')
    plt.fill_between(comm_budget_list_CP, CP_err_max, CP_err_min,  # 上限，下限
                     facecolor=color[2],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    # if ('non' not in dst_name):
    #     plt.ylim((0.15, 0.5))

    plt.xscale('log')
    plt.legend(fontsize=23)
    plt.tick_params(labelsize=23)
    # plt.title("Regret of POCO(L)")

    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel("Communication Budget", fontsize=23)
    plt.ylabel("Error Rate", fontsize=23)
    fig = plt.gcf()

    # fig.savefig("fig_regret_loss_cube_conv.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(dst_name, format=format, bbox_inches="tight")
    plt.show()


def plot_adv_32_8(src_dirname_8, src_dirname_32):
    labels = ["non-pri", "non-comm", "PDOM"]
    filename = "covtype.libsvm.binary"
    # color = ["#4285F4", '#fbbc05', 'xkcd:red']
    # color = ["#4285F4", '#fbbc05', 'xkcd:red']
    # color_light = ["#4285F4", '#fbfa05', 'xkcd:red']
    # color_dark = ["#4285F4", '#fb7f05', 'xkcd:red']
    # color = ["#4285F4", '#c29103', 'xkcd:red']
    # # color_light = ["#4285F4", '#fbbc05', 'xkcd:red']
    # color_light = ["#4285F4", '#c29103', 'xkcd:red']
    # color_dark = ["#4285F4", '#c29103', 'xkcd:red']
    # plt.figure(figsize=(6.5, 3.49))

    # color = ["#72a4f7", '#dea604', 'xkcd:red']
    # color_light = ["#72a4f7", '#dea604', 'xkcd:red']
    # color_dark = ["#72a4f7", '#dea604', 'xkcd:red']

    # color = ["#a2c3fa", '#c29103', 'xkcd:red']
    # color_light = ["#a2c3fa", '#c29103', 'xkcd:red']
    # color_dark = ["#a2c3fa", '#c29103', 'xkcd:red']
    color = ["#8ab4f8", '#c29103', 'xkcd:red']
    color_light = ["#8ab4f8", '#c29103', 'xkcd:red']
    color_dark = ["#8ab4f8", '#c29103', 'xkcd:red']
    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10

    ###################################################################################### 8
    comm_budget_list_DOCO = [600, 800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
                            [i for i in range(60000, 600000, 60000)] + [i for i in range(600000, 4800001, 600000)]

    comm_budget_list_TDOCO = [600, 800, ] + [i for i in range(1200, 6000, 600)] + [i for i in
                                                                                   range(6000, 60000, 6000)] + \
                             [i for i in range(60000, 240001, 60000)]

    ############################################################
    # DOCO_gossip
    # comm_budget_list_DOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                         [i for i in range(60000, 300000, 60000)] + [420000, ] + \
    #                         [i for i in range(600000, 1200001, 600000)]
    DOCO_gossip_err_mean = []
    DOCO_gossip_err_max = []
    DOCO_gossip_err_min = []
    for comm_budget in comm_budget_list_DOCO:
        file_name = src_dirname_8 + 'DOCO_gossip_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        DOCO_gossip_err_max.append(np.max(class_err_list))
        DOCO_gossip_err_min.append(np.min(class_err_list))
        DOCO_gossip_err_mean.append(np.mean(class_err_list))
    print(DOCO_gossip_err_max)
    print(DOCO_gossip_err_min)
    ############################################################
    ############################################################
    # DBOCG
    # comm_budget_list_DOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                         [i for i in range(60000, 300000, 60000)] + [420000, ] + \
    #                         [i for i in range(600000, 1200001, 600000)]
    DBOCG_err_mean = []
    DBOCG_err_max = []
    DBOCG_err_min = []
    for comm_budget in comm_budget_list_DOCO:
        file_name = src_dirname_8 + 'DBOCG_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        DBOCG_err_max.append(np.max(class_err_list))
        DBOCG_err_min.append(np.min(class_err_list))
        DBOCG_err_mean.append(np.mean(class_err_list))
    print(DBOCG_err_max)
    print(DBOCG_err_min)
    ############################################################
    ############################################################
    # TDOCO_delay
    # comm_budget_list_TDOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                          [i for i in range(60000, 120000, 60000)]
    DB_TDOCO_err_mean = []
    DB_TDOCO_err_max = []
    DB_TDOCO_err_min = []
    for comm_budget in comm_budget_list_TDOCO:
        file_name = src_dirname_8 + 'DB_TDOCO_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        DB_TDOCO_err_max.append(np.max(class_err_list))
        DB_TDOCO_err_min.append(np.min(class_err_list))
        DB_TDOCO_err_mean.append(np.mean(class_err_list))
    ############################################################

    plt.plot(comm_budget_list_DOCO, DOCO_gossip_err_mean, '-', linewidth=param_linewidth,
             markersize=param_markersize_, c=color_light[1], label='gossip')

    plt.fill_between(comm_budget_list_DOCO, DOCO_gossip_err_max, DOCO_gossip_err_min,  # 上限，下限
                     facecolor=color_dark[1],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    plt.plot(comm_budget_list_DOCO, DBOCG_err_mean, '-', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[0], label='D-BOCG')

    plt.fill_between(comm_budget_list_DOCO, DBOCG_err_max, DBOCG_err_min,  # 上限，下限
                     facecolor=color[0],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    plt.plot(comm_budget_list_TDOCO, DB_TDOCO_err_mean, '-', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[2], label='DB-TDOCO')
    plt.fill_between(comm_budget_list_TDOCO, DB_TDOCO_err_max, DB_TDOCO_err_min,  # 上限，下限
                     facecolor=color[2],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    ###################################################################################### 32
    comm_budget_list_DOCO = [600, 800] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
                            [i for i in range(60000, 600000, 60000)] + [i for i in range(600000, 4800000, 600000)] + \
                            [i for i in range(4800000, 4800000 * 4 + 1, 4800000)]

    comm_budget_list_TDOCO = [600, 800, ] + [i for i in range(1200, 6000, 600)] + [i for i in
                                                                                   range(6000, 60000, 6000)] + \
                             [i for i in range(60000, 240001, 60000)]

    ############################################################
    # DOCO_gossip
    # comm_budget_list_DOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                         [i for i in range(60000, 300000, 60000)] + [420000, ] + \
    #                         [i for i in range(600000, 1200001, 600000)]
    DOCO_gossip_err_mean = []
    DOCO_gossip_err_max = []
    DOCO_gossip_err_min = []
    for comm_budget in comm_budget_list_DOCO:
        file_name = src_dirname_32 + 'DOCO_gossip_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        DOCO_gossip_err_max.append(np.max(class_err_list))
        DOCO_gossip_err_min.append(np.min(class_err_list))
        DOCO_gossip_err_mean.append(np.mean(class_err_list))
    print(DOCO_gossip_err_max)
    print(DOCO_gossip_err_min)
    ############################################################
    ############################################################
    # DBOCG
    # comm_budget_list_DOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                         [i for i in range(60000, 300000, 60000)] + [420000, ] + \
    #                         [i for i in range(600000, 1200001, 600000)]
    DBOCG_err_mean = []
    DBOCG_err_max = []
    DBOCG_err_min = []
    for comm_budget in comm_budget_list_DOCO:
        file_name = src_dirname_32 + 'DBOCG_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        DBOCG_err_max.append(np.max(class_err_list))
        DBOCG_err_min.append(np.min(class_err_list))
        DBOCG_err_mean.append(np.mean(class_err_list))
    print(DBOCG_err_max)
    print(DBOCG_err_min)
    ############################################################
    ############################################################
    # TDOCO_delay
    # comm_budget_list_TDOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                          [i for i in range(60000, 120000, 60000)]
    DB_TDOCO_err_mean = []
    DB_TDOCO_err_max = []
    DB_TDOCO_err_min = []
    for comm_budget in comm_budget_list_TDOCO:
        file_name = src_dirname_32 + 'DB_TDOCO_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        DB_TDOCO_err_max.append(np.max(class_err_list))
        DB_TDOCO_err_min.append(np.min(class_err_list))
        DB_TDOCO_err_mean.append(np.mean(class_err_list))
    ############################################################

    plt.plot(comm_budget_list_DOCO, DOCO_gossip_err_mean, '--', linewidth=param_linewidth,
             markersize=param_markersize_, c=color_light[1], label='gossip')

    plt.fill_between(comm_budget_list_DOCO, DOCO_gossip_err_max, DOCO_gossip_err_min,  # 上限，下限
                     facecolor=color_dark[1],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    plt.plot(comm_budget_list_DOCO, DBOCG_err_mean, '--', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[0], label='D-BOCG')

    plt.fill_between(comm_budget_list_DOCO, DBOCG_err_max, DBOCG_err_min,  # 上限，下限
                     facecolor=color[0],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    plt.plot(comm_budget_list_TDOCO, DB_TDOCO_err_mean, '--', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[2], label='DB-TDOCO')
    plt.fill_between(comm_budget_list_TDOCO, DB_TDOCO_err_max, DB_TDOCO_err_min,  # 上限，下限
                     facecolor=color[2],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    ############################################################
    plt.xscale('log')
    # plt.legend(fontsize=23)
    plt.tick_params(labelsize=23)
    # plt.title("Regret of POCO(L)")

    # y_major_locator = MultipleLocator(0.1)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y_major_locator)
    plt.xticks([10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10**7])
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    y_minor_locator = MultipleLocator(0.01)
    # ax = plt.gca()
    ax.yaxis.set_minor_locator(y_minor_locator)
    # ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.xaxis.set_tick_params(which='minor', size=1)
    # ax.xaxis.set_tick_params(which='minor', width=1)
    x_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0), numticks=10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.tick_params(labelsize=23)

    #########
    # # 设置图例
    # # labels = ["non-pri", "non-comm", "PDOM"]
    # labels = ["D-BOCG", "gossip", "DB-TDOCO"]
    #
    # # patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in [1, 0, 2]]
    # # legend1 = plt.legend(handles=patches, bbox_to_anchor=(0.99, 1.21), ncol=3, frameon=False, fontsize=23)
    # # e1 = mlines.Line2D([], [], color='black',
    # #                    label="8-learner", linestyle="-")
    # # e2 = mlines.Line2D([], [], color='black',
    # #                    label="32-learner", linestyle="--")
    # # # e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9,
    # # #                    label="DB-TDOCO", linestyle='-')
    # # plt.legend(handles=[e1, e2], bbox_to_anchor=(0.82, 1.14), ncol=2, frameon=False, fontsize=23)
    # # plt.gca().add_artist(legend1)
    # # plt.tick_params(labelsize=23)
    # patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in [1, 0, 2]]
    # legend1 = plt.legend(handles=patches, handletextpad=0.1, columnspacing=0.4, bbox_to_anchor=(0.82 + 0.26, 1.36),
    #                      ncol=3, frameon=False, fontsize=23)
    # e1 = mlines.Line2D([], [], color='black',
    #                    label="8-learner", linestyle="-")
    # e2 = mlines.Line2D([], [], color='black',
    #                    label="32-learner", linestyle="--")
    # # e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9,
    # #                    label="DB-TDOCO", linestyle='-')
    # plt.legend(handles=[e1, e2], columnspacing=0.6, bbox_to_anchor=(1.05, 1.24), ncol=2, frameon=False, fontsize=23)
    # plt.gca().add_artist(legend1)
    # plt.tick_params(labelsize=23)
    # # fig = plt.gcf()
    ##########
    # plt.title("(c) ${\\mathtt{covtype}}$ (clique)", y=-0.5, font={'family': 'Helvetica', 'size': 27})
    plt.title("($\mathtt{c}$) ${\\mathrm{covtype}}$ ($\mathtt{clique}$)", y=-0.55,
              # font={'family':'Helvetica'},
              fontsize=35)
    plt.xlabel("Communication Budget", fontsize=23)
    plt.ylabel("Error Rate", fontsize=23)
    # fig = plt.gcf()
    #
    # # fig.savefig("fig_regret_loss_cube_conv.pdf", format="pdf", bbox_inches="tight")
    # fig.savefig(dst_name, format=format, bbox_inches="tight")
    # plt.show()


if __name__ == '__main__':

    filename = "covtype.libsvm.binary"

    '''
    src_dir_name = 'plot_data/adv_setting_clique/'
    dst_name = filename.replace('.', '_') + '_adv'
    # plot_adv(src_dir_name, dst_name + '_clique.png', format='png')
    plot_adv(src_dir_name, dst_name + '_clique.pdf', format='pdf')
    

    src_dir_name = 'plot_data/iid_setting_clique/'
    dst_name = filename.replace('.', '_') + '_iid'
    # plot_stoc(src_dir_name, dst_name + '_clique.png', format='png')
    plot_stoc(src_dir_name, dst_name + '_clique.pdf', format='pdf')

    src_dir_name = 'plot_data/non_iid_setting_clique/'
    dst_name = filename.replace('.', '_') + '_non_iid'
    # plot_stoc(src_dir_name, dst_name + '_clique.png', format='png')
    plot_stoc(src_dir_name, dst_name + '_clique.pdf', format='pdf')
    '''

    src_dir_name_32 = '32-merge-data/adv_setting_clique/'
    src_dir_name_8 = '8-merge-data/adv_setting_clique/'
    dst_name = filename.replace('.', '_') + '_adv'
    # plot_stoc(src_dir_name, dst_name + '.png', format='png')
    plot_adv_32_8(src_dir_name_8, src_dir_name_32, dst_name + '_clique.pdf', format='pdf')