import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import json


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

    conf_cyc = conf_load(cycle_json_fn)
    err_cyc = conf_cyc['base_err_list']
    comm_budget_CP_cyc = conf_cyc['CP_comm_budget']
    comm_budget_DMA_cyc = conf_cyc['DMA_comm_budget']
    comm_budget_AGD_cyc = conf_cyc['AGD_comm_budget']

    print("Communication ratios in cycle network:")
    print("AGD/DMA:", np.array(comm_budget_AGD_cyc) / np.array(comm_budget_DMA_cyc))
    print("CP/DMA:", np.array(comm_budget_CP_cyc) / np.array(comm_budget_DMA_cyc))

    conf_clique = conf_load(clique_json_fn)
    err_clique = conf_clique['base_err_list']
    comm_budget_CP_clique = conf_clique['CP_comm_budget']
    comm_budget_DMA_clique = conf_clique['DMA_comm_budget']
    comm_budget_AGD_clique = conf_clique['AGD_comm_budget']

    print("Communication ratios in clique network:")
    print("AGD/DMA:", np.array(comm_budget_AGD_clique) / np.array(comm_budget_DMA_clique))
    print("CP/DMA:", np.array(comm_budget_CP_clique) / np.array(comm_budget_DMA_clique))

    # ######## cyc
    # comm_budget_CP_cyc = [800, 1400, 1400, 2000, 1800]
    # comm_budget_DMA_cyc = [12000.0, 18000.0, 30000.0, 36000.0, 42000.0]
    # comm_budget_AGD_cyc = [1900.0, 2700.0, 3500.0, 3800.0, 4100.0]
    # ########
    # ######## clique
    # comm_budget_CP_clique = [1400, 2600, 2800, 2600, 2800]
    # comm_budget_DMA_clique = [24000.0, 42000.0, 48000.0, 60000.0, 70000.0]
    # comm_budget_AGD_clique = [3300.0, 4600.0, 5100.0, 5400.0, 5800.0]
    # ########

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
    ax1.set_xticks(x, [3.6, 7.2, 10.8, 14.4, 18], fontsize=23)
    ax1.set_yticks([0.0, 0.5])
    ax1.set_xlabel("$T (\\times 10^3)$", fontsize=23)
    ax1.set_ylabel("Error Rate", fontsize=23)
    ax1.set_ylim(0, 1)
    ax1.tick_params(labelsize=23)
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
    # x_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0), numticks=10)
    # ax2.yaxis.set_minor_locator(x_minor)
    # ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # ax2.set_yticks([10**4, 10**5, 10**6])
    ax2.set_ylabel("Comm.", fontsize=23)
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
    # e1 = mlines.Line2D([], [], color='black', marker='v', markersize=9, markerfacecolor='white', markeredgewidth=2,
    #                    label="$\\mathtt{DMA}$", linestyle=":")
    # e2 = mlines.Line2D([], [], color='black', marker='^', markersize=9, markerfacecolor='white', markeredgewidth=2,
    #                    label="$\\mathtt{DB2O}_a$", linestyle="--")
    # e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2,
    #                    label="$\\mathtt{DB2O}_c$", linestyle='-')
    # plt.legend(handles=[e1, e2, e3], handletextpad=0.2, columnspacing=0.6, bbox_to_anchor=(.97, 1.25), ncol=3, frameon=False, fontsize=23)
    # plt.gca().add_artist(legend1)
    plt.tick_params(labelsize=23)
    # plt.title("Regret of POCO(L)")
    plt.xlabel("$T$ $(\\times 10^3)$", fontsize=23)
    # plt.ylabel("Communication Cost", fontsize=23)
    # plt.title("($\\mathtt{b}$)"
    #           # " $\\mathtt{Non}$-$\\mathtt{i.i.d.}$"
    #           " $\\mathtt{Stochastic}$ $\\mathtt{DOCO}$", y=-0.5, font={'family': 'Helvetica'},
    #           fontsize=35)
    plt.title("($\\mathtt{d}$)"
              " $\\mathtt{32}$-$\\mathtt{learner}$", y=-0.5,
              # font={'family': 'Helvetica'},
              fontsize=35)


# if __name__ == '__main__':
#     err_cyc = [0.29913403614457834, 0.2809321213425129, 0.2694797045324154, 0.26538484294320136, 0.260782487091222]
#     err_clique = [0.2786510327022375, 0.264785391566265, 0.25934380378657484, 0.2557610800344234, 0.25284272805507746]
#     draw_trend_image_test(err_cyc, err_clique, image_file_name='non_iid_comm_time.pdf')
