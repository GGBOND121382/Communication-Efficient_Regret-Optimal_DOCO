import json

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


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
    # comm_budget_DB_TDOCO_cyc = [4200, 12000, 18000, 24000, 36000]
    # comm_budget_gossip_cyc = [232401, 464801, 697201, 929601, 1162001]
    # comm_budget_DBOCG_cyc = [232401, 464801, 697201, 929601, 1162001]
    # ########
    # ######## clique
    # comm_budget_DB_TDOCO_clique = [24000, 42000, 120000, 120000, 180000]
    # comm_budget_gossip_clique = [813401, 1626801, 2440201, 3253601, 4067001]
    # comm_budget_DBOCG_clique = [813401, 1626801, 2440201, 3253601, 4067001]
    # ########

    conf_cyc = conf_load(cycle_json_fn)
    err_cyc = conf_cyc['base_err_list']
    comm_budget_DB_TDOCO_cyc = conf_cyc['DB_TDOCO_comm_budget']
    comm_budget_gossip_cyc = conf_cyc['gossip_comm_budget']
    comm_budget_DBOCG_cyc = conf_cyc['DBOCG_comm_budget']

    conf_clique = conf_load(clique_json_fn)
    err_clique = conf_clique['base_err_list']
    comm_budget_DB_TDOCO_clique = conf_clique['DB_TDOCO_comm_budget']
    comm_budget_gossip_clique = conf_clique['gossip_comm_budget']
    comm_budget_DBOCG_clique = conf_clique['DBOCG_comm_budget']

    # if len(err_cyc) > 5:
    #     fig, ax1 = plt.subplots(figsize=(6.5, 3.49))
    # else:
    #     fig, ax1 = plt.subplots(figsize=(6.5, 3.49))
    # fig, ax1 = plt.subplots(figsize=(6.5, 3.49))
    # fig = plt.figure()
    # ax1 = plt.figure().add_subplot()
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
    # draw plot
    ax2 = ax1.twinx()
    # ax2.plot(x, total_pass_rate, label='Total Pass Rate',
    #          linewidth=2, color='#FFB90F')

    ######
    ax2.plot(x, comm_budget_DB_TDOCO_cyc, '-', marker='o', color=cyc_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_)
    ax2.plot(x + 0.01, comm_budget_gossip_cyc, ':', marker='v', color=cyc_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_ + 2)
    ax2.plot(x - 0.01, comm_budget_DBOCG_cyc, '--', marker='^', color=cyc_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_)
    ######
    ######
    ax2.plot(x, comm_budget_DB_TDOCO_clique, '-', marker='o', color=clique_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_)
    ax2.plot(x + 0.01, comm_budget_gossip_clique, ':', marker='v', color=clique_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_ + 2)
    ax2.plot(x - 0.01, comm_budget_DBOCG_clique, '--', marker='^', color=clique_color, linewidth = param_linewidth, markerfacecolor='white', markeredgewidth=2,
    markersize = param_markersize_)
    ######

    ######
    ax2.set_ylabel("Percent", fontsize=23)
    ax2.set_yscale('log')
    ax2.tick_params(labelsize=23)

    #########
    # 设置图例
    # labels = ["non-pri", "non-comm", "PDOM"]
    # labels = ["cycle", "clique"]

    # patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(labels))]
    # legend1 = plt.legend(handles=patches, bbox_to_anchor=(0.82+0.14, 1.37), ncol=2, frameon=False, fontsize=23)
    # e1 = mlines.Line2D([], [], color='black', marker='v', markersize=9, markerfacecolor='white', markeredgewidth=2,
    #                    label="$\\mathtt{gossip}$", linestyle=":")
    # e2 = mlines.Line2D([], [], color='black', marker='^', markersize=9, markerfacecolor='white', markeredgewidth=2,
    #                    label="$\\mathtt{D}$-$\\mathtt{BOCG}$", linestyle="--")
    # e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9,  markerfacecolor='white', markeredgewidth=2,
    #                    label="$\\mathtt{DB}$-$\\mathtt{TDOCO}$", linestyle='-')
    # plt.legend(handles=[e1, e2, e3], handletextpad=0.1, columnspacing=0.4, bbox_to_anchor=(1.05+0, 1.25), ncol=3, frameon=False, fontsize=23)
    # plt.gca().add_artist(legend1)
    plt.tick_params(labelsize=23)
    # plt.title("Regret of POCO(L)")
    plt.xlabel("$T$ $(\\times 10^3)$", fontsize=23)
    plt.ylabel("Comm.", fontsize=23)
    # plt.title("($\\mathtt{a}$) $\\mathtt{Adversarial}$ $\\mathtt{DOCO}$", y=-0.5, font={'family': 'Helvetica'},
    #           fontsize=35)
    plt.title("($\\mathtt{a}$) $\\mathtt{8}$-$\\mathtt{learner}$", y=-0.5,
              # font={'family': 'Helvetica'},
              fontsize=35)
    # fig = plt.gcf()
    ##########



    # plt.show()
    # fig.savefig(image_file_name, format="pdf", bbox_inches="tight", dpi=450)

# if __name__ == '__main__':
#     err_cyc = [0.4330510434595525, 0.4068829335197935, 0.3920852158634538, 0.37693268610154906, 0.36471985262478485]
#     err_clique = [0.36407446751290873, 0.34584088586488815, 0.3341013159781985, 0.32188175693846816, 0.3137241555507745]
#     draw_trend_image_test(err_cyc, err_clique, image_file_name='adv_comm_time.pdf')
