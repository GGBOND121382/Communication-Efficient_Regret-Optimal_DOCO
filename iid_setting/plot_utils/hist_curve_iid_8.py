import json

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as mp

# mp.rcParams['text.usetex'] = True #Let TeX do the typsetting
# mp.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
# mp.rcParams['font.family'] = 'sans-serif' # ... for regular text
# # mp.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here
# mp.rcParams['font.sans-serif'] = 'Helvetica'


def to_percent(num, position):
    return str(num) + '%'


def conf_load(filename='conf.ini'):
    with open(filename) as fd:
        conf_dict = json.load(fd)
    return conf_dict


def draw_trend_image_test(ax1, cycle_json_fn, clique_json_fn):
    # color = ["#4285F4", '#fbbc05', 'xkcd:red']
    # color = ["#4285F4", '#fbbc05']
    # color = ["#4285F4", '#c29103']
    # color = ["#8ab4f8", '#c29103']
    color = ['#c29103', "#8ab4f8"]
    # attributes
    bar_width = 0.3
    cyc_color = color[0]
    clique_color = color[1]
    # font_name = 'Calibri'
    label_size = 10
    text_size = 8
    title_size = 14

    param_linewidth = 3.0
    param_markersize = 8
    param_markersize_ = 10

    # set the color for failure
    smoke_fail_color = []

    reg_fail_color = []

    total_pass_rate = [0.5, 0.5, 0.5, 0.5, 0.5]

    # ######## cyc
    # comm_budget_CP_cyc = [1000, 1200, 1200, 1000, 1600]
    # comm_budget_DMA_cyc = [12000.0, 30000.0, 48000.0, 60000.0, 80000.0]
    # comm_budget_AGD_cyc = [3100.0, 6000.0, 4800.0, 5800.0, 6000.0]
    # ########
    # ######## clique
    # comm_budget_CP_clique = [1600, 2000, 1800, 2600, 2800]
    # comm_budget_DMA_clique = [36000.0, 120000.0, 180000.0, 240000.0, 180000.0]
    # comm_budget_AGD_clique = [5700.0, 8000.0, 11000.0, 18000.0, 11000.0]
    # ########

    conf_cyc = conf_load(cycle_json_fn)
    err_cyc = conf_cyc['base_err_list']
    comm_budget_CP_cyc = conf_cyc['CP_comm_budget']
    comm_budget_DMA_cyc = conf_cyc['DMA_comm_budget']
    comm_budget_AGD_cyc = conf_cyc['AGD_comm_budget']

    conf_clique = conf_load(clique_json_fn)
    err_clique = conf_clique['base_err_list']
    comm_budget_CP_clique = conf_clique['CP_comm_budget']
    comm_budget_DMA_clique = conf_clique['DMA_comm_budget']
    comm_budget_AGD_clique = conf_clique['AGD_comm_budget']

    for i in range(5):
        l = comm_budget_CP_cyc[i]
        r = comm_budget_DMA_cyc[i]
        print(l / r)
    print()
    for i in range(5):
        l = comm_budget_CP_clique[i]
        r = comm_budget_DMA_clique[i]
        print(l / r)

    # if len(err_cyc) > 5:
    #     fig, ax1 = plt.subplots(figsize=(6.5, 3.49))
    # else:
    #     fig, ax1 = plt.subplots(figsize=(6.5, 3.49))

    # draw bar
    x = np.arange(len(err_cyc))
    ax1.bar(x - bar_width / 2, err_cyc, color=cyc_color, edgecolor=cyc_color,
            width=bar_width, label="Clique", alpha=0.8)
    ax1.bar(x + bar_width / 2, err_clique, color=clique_color, edgecolor=clique_color,
            width=bar_width, label="Cycle", alpha=0.8)


    # set title, labels
    # ax1.set_title("Test Result Trend", fontsize=title_size)
    ax1.set_xticks(x, [1.44, 2.88, 4.32, 5.76, 7.20], fontsize=23)
    ax1.set_yticks([0.0, 0.5])
    ax1.set_xlabel("$T (\\times 10^4)$", fontsize=23)
    ax1.set_ylabel("Error Rate", fontsize=23)
    ax1.set_ylim(0, 1)
    ax1.tick_params(labelsize=23)

    '''
    # set bar text
    for i in x:
        if smoke_total_count_fail[i] > 0:
            ax1.text(i - bar_width / 2, smoke_total_count_fail[i] + smoke_total_count_pass[i],
                     smoke_total_count_fail[i], horizontalalignment='center', verticalalignment='bottom',
                     fontsize=text_size, family=font_name, color='red', weight='bold')

        ax1.text(i - bar_width, smoke_total_count_pass[i], smoke_total_count_pass[i], horizontalalignment='right',
                 verticalalignment='top', fontsize=text_size, family=font_name, color=smoke_pass_color, weight='bold')
        ax1.text(i, reg_total_count_pass[i], reg_total_count_pass[i], horizontalalignment='right',
                 verticalalignment='top', fontsize=text_size, family=font_name, color=regression_pass_color,
                 weight='bold')

        if reg_total_count_fail[i] > 0:
            ax1.text(i + bar_width / 2, reg_total_count_fail[i] + reg_total_count_pass[i], reg_total_count_fail[i],
                     horizontalalignment='center', verticalalignment='bottom', fontsize=text_size, family=font_name,
                     color='red', weight='bold')
    '''

    # draw plot
    ax2 = ax1.twinx()
    # ax2.plot(x, total_pass_rate, label='Total Pass Rate',
    #          linewidth=2, color='#FFB90F')

    ######
    ax2.plot(x, comm_budget_CP_cyc, '-', marker='o', color=cyc_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_)
    ax2.plot(x, comm_budget_DMA_cyc, ':', marker='v', color=cyc_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_)
    ax2.plot(x, comm_budget_AGD_cyc, '--', marker='^', color=cyc_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_)
    ######
    ######
    ax2.plot(x, comm_budget_CP_clique, '-', marker='o', color=clique_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_)
    ax2.plot(x, comm_budget_DMA_clique, ':', marker='v', color=clique_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_)
    ax2.plot(x, comm_budget_AGD_clique, '--', marker='^', color=clique_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_)
    ######

    ######

    # ax2.yaxis.set_major_formatter(FuncFormatter(to_percent))
    # ax2.set_ylim(0, 100)
    ax2.set_ylabel("Percent", fontsize=23)
    ax2.set_yscale('log')
    ax2.tick_params(labelsize=23)
    # plt.show() # should comment it if save the pic as a file, or the saved pic is blank
    # dpi: image resolution, better resolution, bigger image size
    # legend_font = font_manager.FontProperties(weight='normal', style='normal', size=label_size)
    # fig.legend(loc="upper center", ncol=4, frameon=False, prop=legend_font)

    #########
    # 设置图例
    # labels = ["non-pri", "non-comm", "PDOM"]
    # labels = ["cycle", "clique"]
    #
    # patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(labels))]
    # legend1 = plt.legend(handles=patches, bbox_to_anchor=(0.82+0.14, 1.37), ncol=2, frameon=False, fontsize=23)
    # e1 = mlines.Line2D([], [], color='black', marker='v', markersize=9,
    #                    label="$DMA$", linestyle=":")
    # e2 = mlines.Line2D([], [], color='black', marker='^', markersize=9,
    #                    label="$DB2O_a$", linestyle="--")
    # e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2,
    #                    label="$DB2O_c$", linestyle='-')
    # plt.legend(handles=[e1, e2, e3], handletextpad=0.2, columnspacing=0.6, bbox_to_anchor=(1.05+0.05, 1.25), ncol=3, frameon=False, fontsize=23)
    # plt.gca().add_artist(legend1)
    # plt.tick_params(labelsize=23)
    # # plt.title("Regret of POCO(L)")
    # plt.xlabel("$T$ $(\\times 10^3)$", fontsize=23)
    # plt.ylabel("Communication Cost", fontsize=23)

    # labels = ["cycle", "clique"]
    #
    # patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(labels))]
    # legend1 = plt.legend(handles=patches, bbox_to_anchor=(0.82 + 0.14, 1.37), ncol=2, frameon=False, fontsize=23)
    # e1 = mlines.Line2D([], [], color='black', marker='v', markersize=9, markerfacecolor='white', markeredgewidth=2,
    #                    label="$\\mathtt{DMA}$", linestyle=":")
    # e2 = mlines.Line2D([], [], color='black', marker='^', markersize=9, markerfacecolor='white', markeredgewidth=2,
    #                    label="$\\mathtt{DB2O}_a$", linestyle="--")
    # e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2,
    #                    label="$\\mathtt{DB2O}_c$", linestyle='-')
    # plt.legend(handles=[e1, e2, e3], handletextpad=0.2, columnspacing=0.6, bbox_to_anchor=(.97, 1.25), ncol=3,
    #            frameon=False, fontsize=23)
    # plt.gca().add_artist(legend1)
    plt.tick_params(labelsize=23)
    # plt.title("Regret of POCO(L)")
    plt.xlabel("$T$ $(\\times 10^3)$", fontsize=23)
    plt.ylabel("Comm.", fontsize=23)
    # plt.title("($\\mathtt{b}$) $\\mathtt{Stochastic}$ $\\mathtt{DOCO}$", y=-0.5, font={'family': 'Helvetica'},
    #           fontsize=35)

    # plt.title("($\\mathtt{c}$) $\\mathtt{i.i.d.}$ $\\mathtt{Stochastic}$ $\\mathtt{DOCO}$", y=-0.5, font={'family': 'Helvetica'},
    #           fontsize=35)

    plt.title("($\\mathtt{a}$) $\\mathtt{8}$-$\\mathtt{learner}$", y=-0.5,
              # font={'family': 'Helvetica'},
              fontsize=35)

    # fig = plt.gcf()
    ##########

    # plt.show()
    # fig.savefig(image_file_name, format="pdf", bbox_inches="tight", dpi=450)


if __name__ == '__main__':
    err_cyc = [0.32125645438898454, 0.31079388984509465, 0.30585233792312105, 0.3022442448364888, 0.29965103270223753]
    err_clique = [0.3120895008605852, 0.3021675989672978, 0.29658880419637734, 0.29340304073436607, 0.2925357142857143]
    draw_trend_image_test(err_cyc, err_clique, image_file_name='iid_comm_time.pdf')
