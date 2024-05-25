import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from matplotlib.ticker import FuncFormatter
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
    comm_budget_DB_TDOCO_cyc = conf_cyc['DB_TDOCO_comm_budget']
    comm_budget_gossip_cyc = conf_cyc['gossip_comm_budget']
    comm_budget_DBOCG_cyc = conf_cyc['DBOCG_comm_budget']

    conf_clique = conf_load(clique_json_fn)
    err_clique = conf_clique['base_err_list']
    comm_budget_DB_TDOCO_clique = conf_clique['DB_TDOCO_comm_budget']
    comm_budget_gossip_clique = conf_clique['gossip_comm_budget']
    comm_budget_DBOCG_clique = conf_clique['DBOCG_comm_budget']

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
    # draw plot
    ax2 = ax1.twinx()

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
    ax2.set_yscale('log')
    x_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0), numticks=10)
    ax2.yaxis.set_minor_locator(x_minor)
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_yticks([10**3, 10**4, 10**5, 10**6, 10**7])
    ax2.tick_params(labelsize=23)
    ax2.set_ylabel("Comm.", fontsize=23)

    plt.xlabel("$T$ $(\\times 10^3)$", fontsize=23)
    plt.title("($\\mathtt{b}$) $\\mathtt{32}$-$\\mathtt{learner}$", y=-0.5,
              # font={'family': 'Helvetica'},
              fontsize=35)

