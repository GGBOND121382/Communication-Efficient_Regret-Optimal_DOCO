import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sys

sys.path.append('../../')


def plot_adv(src_dir_name, dst_name, format='png'):
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
    # TDOCO_delay
    # comm_budget_list_TDOCO = [800, ] + [i for i in range(1200, 6000, 600)] + [i for i in range(6000, 60000, 6000)] + \
    #                          [i for i in range(60000, 120000, 60000)]
    TDOCO_delay_err_mean = []
    TDOCO_delay_err_max = []
    TDOCO_delay_err_min = []
    for comm_budget in comm_budget_list_TDOCO:
        file_name = src_dir_name + 'TDOCO_delay_' + filename + '_' + repr(comm_budget)
        lossmean_list = []
        class_err_list = []
        with open(file_name) as fd:
            for line in fd.readlines():
                tmp = line.split(" ")
                rst = [float(str(tmp[0])), str(tmp[1])]
                lossmean_list.append(float(rst[0]))
                class_err_list.append(float(rst[1]))
                # classerr_list.append(float(rst[1]))
        TDOCO_delay_err_max.append(np.max(class_err_list))
        TDOCO_delay_err_min.append(np.min(class_err_list))
        TDOCO_delay_err_mean.append(np.mean(class_err_list))
    ############################################################

    plt.plot(comm_budget_list_DOCO, DOCO_gossip_err_mean, '--', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[1], label='gossip')

    plt.fill_between(comm_budget_list_DOCO, DOCO_gossip_err_max, DOCO_gossip_err_min,  # 上限，下限
                     facecolor=color[1],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    plt.plot(comm_budget_list_TDOCO, TDOCO_delay_err_mean, '-.', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[0], label='$TDOCO$')
    plt.fill_between(comm_budget_list_TDOCO, TDOCO_delay_err_max, TDOCO_delay_err_min,  # 上限，下限
                     facecolor=color[0],  # 填充颜色
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
    # plt.show()


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

    # y_major_locator = MultipleLocator(0.05)
    ax = plt.gca()
    # ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel("Communication Budget", fontsize=23)
    plt.ylabel("Error Rate", fontsize=23)
    fig = plt.gcf()

    # fig.savefig("fig_regret_loss_cube_conv.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(dst_name, format=format, bbox_inches="tight")
    # plt.show()


def plot_stoc_32_8(src_dirname_8, src_dirname_32, dst_name, format='png'):
    filename = "covtype.libsvm.binary"
    labels = ["non-pri", "non-comm", "PDOM"]
    # color = ["#4285F4", '#fbbc05', 'xkcd:red']
    # plt.figure(figsize=(6.5, 3.49))
    # color = ["#4285F4", '#c29103', 'xkcd:red']
    # color = ["#72a4f7", '#dea604', 'xkcd:red']
    # color = ["#a2c3fa", '#c29103', 'xkcd:red']
    color = ["#8ab4f8", '#c29103', 'xkcd:red']

    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10

    ###################################################################################### 8

    comm_budget_list_DMA = [i for i in range(100, 6000, 100)] + [i for i in range(6000, 60000, 6000)] + \
                           [i for i in range(60000, 120001, 60000)]

    comm_budget_list_AGD = [i for i in range(100, 1200, 100)] + [i for i in range(1200, 6000, 200)] + [i for i in
                                                                                                       range(6000,
                                                                                                             36000,
                                                                                                             6000)]

    comm_budget_list_CP = [100, ] + [i for i in range(200, 1600, 200)] + [i for i in range(1600, 3000, 200)]

    # DMA
    DMA_err_mean = []
    DMA_err_max = []
    DMA_err_min = []
    for comm_budget in comm_budget_list_DMA:
        file_name = src_dirname_8 + 'DMA_' + filename + '_' + repr(comm_budget)
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

    # AGD
    AGD_err_mean = []
    AGD_err_max = []
    AGD_err_min = []
    for comm_budget in comm_budget_list_AGD:
        file_name = src_dirname_8 + 'AGD_' + filename + '_' + repr(comm_budget)
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

    # CP
    # comm_budget_list_CP = [100, ] + [i for i in range(200, 1600, 200)] + [i for i in range(1600, 3000, 200)]
    CP_err_mean = []
    CP_err_max = []
    CP_err_min = []
    for comm_budget in comm_budget_list_CP:
        file_name = src_dirname_8 + 'CP_' + filename + '_' + repr(comm_budget)
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

    plt.plot(comm_budget_list_DMA, DMA_err_mean, '-', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[1], label='$DMA$')
    plt.fill_between(comm_budget_list_DMA, DMA_err_max, DMA_err_min,  # 上限，下限
                     facecolor=color[1],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    plt.plot(comm_budget_list_AGD, AGD_err_mean, '-', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[0], label='$DB2O_a$')
    plt.fill_between(comm_budget_list_AGD, AGD_err_max, AGD_err_min,  # 上限，下限
                     facecolor=color[0],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    plt.plot(comm_budget_list_CP, CP_err_mean, '-', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[2], label='$DB2O_c$')
    plt.fill_between(comm_budget_list_CP, CP_err_max, CP_err_min,  # 上限，下限
                     facecolor=color[2],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度
    ###################################################################################### 32
    comm_budget_list_DMA = [i for i in range(100, 6000, 100)] + [i for i in range(6000, 60000, 6000)] + \
                           [i for i in range(60000, 120001, 60000)]

    comm_budget_list_AGD = [i for i in range(100, 6000, 100)] + [i for i in range(6000, 24000, 6000)] + [20000, ]

    comm_budget_list_CP = [100, ] + [i for i in range(200, 3000, 200)] + [i for i in range(3000, 7000, 800)]
    # DMA
    DMA_err_mean = []
    DMA_err_max = []
    DMA_err_min = []
    for comm_budget in comm_budget_list_DMA:
        file_name = src_dirname_32 + 'DMA_' + filename + '_' + repr(comm_budget)
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

    # AGD
    AGD_err_mean = []
    AGD_err_max = []
    AGD_err_min = []
    for comm_budget in comm_budget_list_AGD:
        if (('non' in dst_name and comm_budget > 806) or ('non' not in dst_name and comm_budget > 620)):
            file_name = src_dirname_32 + 'AGD_' + filename + '_' + repr(comm_budget)
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
        else:
            AGD_err_mean.append(0.4876160084820445)
            AGD_err_max.append(0.4876160084820445)
            AGD_err_min.append(0.4876160084820445)

    # CP
    # comm_budget_list_CP = [100, ] + [i for i in range(200, 1600, 200)] + [i for i in range(1600, 3000, 200)]
    CP_err_mean = []
    CP_err_max = []
    CP_err_min = []
    for comm_budget in comm_budget_list_CP:
        file_name = src_dirname_32 + 'CP_' + filename + '_' + repr(comm_budget)
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

    plt.plot(comm_budget_list_DMA, DMA_err_mean, '--', linewidth=param_linewidth,
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


    plt.plot(comm_budget_list_CP, CP_err_mean, '--', linewidth=param_linewidth,
             markersize=param_markersize_, c=color[2], label='$DB2O_c$')
    plt.fill_between(comm_budget_list_CP, CP_err_max, CP_err_min,  # 上限，下限
                     facecolor=color[2],  # 填充颜色
                     # edgecolor='red',  # 边界颜色
                     alpha=0.3)  # 透明度

    ############################################################
    # #########
    # # 设置图例
    # # labels = ["non-pri", "non-comm", "PDOM"]
    # labels = ["$DB2O_a$", "$DMA$", "$DB2O_c$"]
    #
    # # patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in [1, 0, 2]]
    # # legend1 = plt.legend(handles=patches, bbox_to_anchor=(0.82+0.22, 1.35-0.02), ncol=3, frameon=False, fontsize=20)
    # # e1 = mlines.Line2D([], [], color='black',
    # #                    label="8-learner", linestyle="-")
    # # e2 = mlines.Line2D([], [], color='black',
    # #                    label="32-learner", linestyle="--")
    # # # e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9,
    # # #                    label="DB-TDOCO", linestyle='-')
    # # plt.legend(handles=[e1, e2], bbox_to_anchor=(1.05-0.1, 1.24-0.02), ncol=2, frameon=False, fontsize=20)
    # # plt.gca().add_artist(legend1)
    # # plt.tick_params(labelsize=23)
    # patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in [1, 0, 2]]
    # legend1 = plt.legend(handles=patches, handletextpad=0.2, columnspacing=0.6, bbox_to_anchor=(0.82 + 0.24, 1.36),
    #                      ncol=3, frameon=False, fontsize=23)
    # e1 = mlines.Line2D([], [], color='black',
    #                    label="8-learner", linestyle="-")
    # e2 = mlines.Line2D([], [], color='black',
    #                    label="32-learner", linestyle="--")
    # plt.legend(handles=[e1, e2], columnspacing=0.6, bbox_to_anchor=(1.05, 1.24), ncol=2, frameon=False, fontsize=23)
    # plt.gca().add_artist(legend1)
    plt.tick_params(labelsize=23)
    # # fig = plt.gcf()
    ##########
    # if ('non' not in dst_name):
    #     plt.ylim((0.15, 0.52))

    plt.xscale('log')
    # plt.legend(fontsize=23)
    plt.tick_params(labelsize=23)
    # plt.title("Regret of POCO(L)")

    # y_major_locator = MultipleLocator(0.05)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y_major_locator)
    plt.xticks([10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5])
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

    plt.xlabel("Communication Budget", fontsize=23)
    plt.ylabel("Error Rate", fontsize=23)

    if ('non' in dst_name):
        plt.title("($\mathtt{g}$) ${\\mathrm{covtype}}$ ($\mathtt{clique}$)", y=-0.55,
                  # font={'family': 'Helvetica'},
                  fontsize=35)
    else:
        plt.title("($\mathtt{c}$) ${\\mathrm{covtype}}$ ($\mathtt{clique}$)", y=-0.55,
                  # font={'family': 'Helvetica'},
                  fontsize=35)
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
    '''

    src_dir_name_32 = '32-merge-data/iid_setting_clique/'
    src_dir_name_8 = '8-merge-data/iid_setting_clique/'
    dst_name = filename.replace('.', '_') + '_iid'
    # plot_stoc(src_dir_name, dst_name + '.png', format='png')
    plot_stoc_32_8(src_dir_name_8, src_dir_name_32, dst_name + '_clique.pdf', format='pdf')

    src_dir_name_32 = '32-merge-data/non_iid_setting_clique/'
    src_dir_name_8 = '8-merge-data/non_iid_setting_clique/'
    dst_name = filename.replace('.', '_') + '_non_iid'
    # plot_stoc(src_dir_name, dst_name + '.png', format='png')
    plot_stoc_32_8(src_dir_name_8, src_dir_name_32, dst_name + '_clique.pdf', format='pdf')
