import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from matplotlib import rcParams
import sys

# import PLOT_covtype
# import PLOT_epsilon
# import PLOT_covtype_clique
# import PLOT_epsilon_clique

import numpy as np

import iid_setting.plot_utils.comm_err_elect_covtype
import iid_setting.plot_utils.hist_curve_iid_8
import iid_setting.plot_utils.hist_curve_iid_32

import matplotlib.pyplot as mp

# mp.rcParams['text.usetex'] = True  # Let TeX do the typsetting
# mp.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}',
#                                       r'\sansmath']  # Force sans-serif math mode (for axes labels)
# mp.rcParams['font.family'] = 'sans-serif'  # ... for regular text
# # mp.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here
# mp.rcParams['font.sans-serif'] = 'Helvetica'

os.chdir('iid_setting')

src_dir_name = 'plot_data_cov_8/'
base_err_list = iid_setting.plot_utils.comm_err_elect_covtype.election_stoc_print(8, src_dir_name)
iid_setting.plot_utils.comm_err_elect_covtype.election_stoc(8, src_dir_name, base_err_list,
                                                            'plot_utils/8_iid_setting')

src_dir_name = '../iid_setting_clique/plot_data_cov_8/'
base_err_list = iid_setting.plot_utils.comm_err_elect_covtype.election_stoc_print(8, src_dir_name)
iid_setting.plot_utils.comm_err_elect_covtype.election_stoc(8, src_dir_name, base_err_list,
                                                            'plot_utils/8_iid_setting_clique')

src_dir_name = 'plot_data_cov_32/'
base_err_list = iid_setting.plot_utils.comm_err_elect_covtype.election_stoc_print(32, src_dir_name)
iid_setting.plot_utils.comm_err_elect_covtype.election_stoc(32, src_dir_name, base_err_list,
                                                            'plot_utils/32_iid_setting')

src_dir_name = '../iid_setting_clique/plot_data_cov_32/'
base_err_list = iid_setting.plot_utils.comm_err_elect_covtype.election_stoc_print(32, src_dir_name)
iid_setting.plot_utils.comm_err_elect_covtype.election_stoc(32, src_dir_name, base_err_list,
                                                            'plot_utils/32_iid_setting_clique')

os.chdir('../')

fig, axs = plt.subplots(1, 2, figsize=(6.5 * 2 + 3, 3.49))

color = ['#c29103', "#8ab4f8"]
labels = ["cycle", "clique"]
patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(labels))]
legend1 = plt.legend(handles=patches, bbox_to_anchor=(1.08 + 0.04 - 1.38, 1.31), ncol=2, frameon=True, fontsize=23)
e1 = mlines.Line2D([], [], color='black', marker='v', markersize=9, markerfacecolor='white', markeredgewidth=2,
                   label="$\\mathtt{DMA}$", linestyle=":")
e2 = mlines.Line2D([], [], color='black', marker='^', markersize=9, markerfacecolor='white', markeredgewidth=2,
                   label="$\\mathtt{DB2O}_a$", linestyle="--")
e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2,
                   label="$\\mathtt{DB2O}_c$", linestyle='-')
plt.legend(handles=[e1, e2, e3], handletextpad=0.2, columnspacing=0.6, bbox_to_anchor=(2.2 + 0.04 - 1.38, 1.31), ncol=3,
           frameon=True, fontsize=23)
plt.gca().add_artist(legend1)

iid_setting.plot_utils.hist_curve_iid_8.draw_trend_image_test(axs[0],
                                                              'iid_setting/plot_utils/8_iid_setting.ini',
                                                              'iid_setting/plot_utils/8_iid_setting_clique.ini')

# # plt.subplot(121)
# TDOCO_32.TDOCO_pure_plot_adv_varying_T.hist_curve.draw_trend_image_test(axs[0][1],
#                                                                # err_cyc, err_clique,
#                                                                image_file_name='adv_comm_time.pdf')
#
# # plt.subplot(122)
# TDOCO_32.TDOCO_pure_plot_varying_T.hist_curve_iid.draw_trend_image_test(axs[1][1],
#                                                                    # err_cyc, err_clique,
#                                                                    image_file_name='iid_comm_time.pdf')

iid_setting.plot_utils.hist_curve_iid_32.draw_trend_image_test(axs[1],
                                                               'iid_setting/plot_utils/32_iid_setting.ini',
                                                               'iid_setting/plot_utils/32_iid_setting_clique.ini')

# src_dir_name_32 = '32-merge-data/adv_setting/'
# src_dir_name_8 = '8-merge-data/adv_setting/'
# dst_name = filename.replace('.', '_') + '_adv'
# # plot_stoc(src_dir_name, dst_name + '.png', format='png')
# PLOT_epsilon.plot_adv_32_8(src_dir_name_8, src_dir_name_32, dst_name + '.pdf', format='pdf')
#
# plt.subplot(143)
# src_dir_name_32 = '32-merge-data/adv_setting_clique/'
# src_dir_name_8 = '8-merge-data/adv_setting_clique/'
# dst_name = filename.replace('.', '_') + '_adv'
# # plot_stoc(src_dir_name, dst_name + '.png', format='png')
# PLOT_covtype_clique.plot_adv_32_8(src_dir_name_8, src_dir_name_32, dst_name + '_clique.pdf', format='pdf')
#
# plt.subplot(144)
# src_dir_name_32 = '32-merge-data/adv_setting_clique/'
# src_dir_name_8 = '8-merge-data/adv_setting_clique/'
# dst_name = filename.replace('.', '_') + '_adv'
# # plot_stoc(src_dir_name, dst_name + '.png', format='png')
# PLOT_epsilon_clique.plot_adv_32_8(src_dir_name_8, src_dir_name_32, dst_name + '_clique.pdf', format='pdf')


# patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in [1, 0, 2]]
# legend1 = plt.legend(handles=patches, bbox_to_anchor=(-1.5+0.5, 1.3),
#                      ncol=3, frameon=True, fontsize=23)
# e1 = mlines.Line2D([], [], color='black',
#                    label="8-learner", linestyle="-")
# e2 = mlines.Line2D([], [], color='black',
#                    label="32-learner", linestyle="--")
# # e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9,
# #                    label="DB-TDOCO", linestyle='-')
# plt.legend(handles=[e1, e2], bbox_to_anchor=(0+0.5, 1.3), ncol=2, frameon=True, fontsize=23)
# plt.gca().add_artist(legend1)

plt.subplots_adjust(wspace=0.40, hspace=0.8)
# plt.tight_layout()

fig = plt.gcf()
# plt.show()
# fig.tight_layout()
fig.savefig("DOCO_iid_varying_time.pdf", format="pdf", bbox_inches="tight", pad_inches=0.07)
