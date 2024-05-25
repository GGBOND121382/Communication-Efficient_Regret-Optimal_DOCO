import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


import os
import adv_setting.plot_utils.data_purify
import adv_setting.plot_utils.comm_err_elect_covtype_cyc
import adv_setting.plot_utils.comm_err_elect_covtype_clique
import non_iid_setting.plot_utils.comm_err_elect_covtype

import adv_setting.plot_utils.hist_curve_adv_8
import adv_setting.plot_utils.hist_curve_adv_32
import non_iid_setting.plot_utils.hist_curve_non_iid_8
import non_iid_setting.plot_utils.hist_curve_non_iid_32

import matplotlib.pyplot as mp

os.chdir('adv_setting')

adv_setting.plot_utils.data_purify.data_purify(8)
adv_setting.plot_utils.data_purify.data_purify(32)

src_dir_name = 'plot_utils/plot_data_purify_8/adv_setting/'
base_err_list = adv_setting.plot_utils.comm_err_elect_covtype_cyc.election_adv_print(8, src_dir_name)
adv_setting.plot_utils.comm_err_elect_covtype_cyc.election_adv(8, src_dir_name, base_err_list,
                                                               'plot_utils/8_adv_setting')

src_dir_name = 'plot_utils/plot_data_purify_8/adv_setting_clique/'
base_err_list = adv_setting.plot_utils.comm_err_elect_covtype_clique.election_adv_print(8, src_dir_name)
adv_setting.plot_utils.comm_err_elect_covtype_clique.election_adv(8, src_dir_name, base_err_list,
                                                                  'plot_utils/8_adv_setting_clique')

src_dir_name = 'plot_utils/plot_data_purify_32/adv_setting/'
base_err_list = adv_setting.plot_utils.comm_err_elect_covtype_cyc.election_adv_print(32, src_dir_name)
adv_setting.plot_utils.comm_err_elect_covtype_cyc.election_adv(32, src_dir_name, base_err_list,
                                                               'plot_utils/32_adv_setting')

src_dir_name = 'plot_utils/plot_data_purify_32/adv_setting_clique/'
base_err_list = adv_setting.plot_utils.comm_err_elect_covtype_clique.election_adv_print(32, src_dir_name)
adv_setting.plot_utils.comm_err_elect_covtype_clique.election_adv(32, src_dir_name, base_err_list,
                                                                  'plot_utils/32_adv_setting_clique')

os.chdir('../non_iid_setting')

src_dir_name = 'plot_data_cov_8/'
base_err_list = non_iid_setting.plot_utils.comm_err_elect_covtype.election_stoc_print(8, src_dir_name)
non_iid_setting.plot_utils.comm_err_elect_covtype.election_stoc(8, src_dir_name, base_err_list,
                                                                'plot_utils/8_non_iid_setting')

src_dir_name = '../non_iid_setting_clique/plot_data_cov_8/'
base_err_list = non_iid_setting.plot_utils.comm_err_elect_covtype.election_stoc_print(8, src_dir_name)
non_iid_setting.plot_utils.comm_err_elect_covtype.election_stoc(8, src_dir_name, base_err_list,
                                                                'plot_utils/8_non_iid_setting_clique')

src_dir_name = 'plot_data_cov_32/'
base_err_list = non_iid_setting.plot_utils.comm_err_elect_covtype.election_stoc_print(32, src_dir_name)
non_iid_setting.plot_utils.comm_err_elect_covtype.election_stoc(32, src_dir_name, base_err_list,
                                                                'plot_utils/32_non_iid_setting')

src_dir_name = '../non_iid_setting_clique/plot_data_cov_32/'
base_err_list = non_iid_setting.plot_utils.comm_err_elect_covtype.election_stoc_print(32, src_dir_name)
non_iid_setting.plot_utils.comm_err_elect_covtype.election_stoc(32, src_dir_name, base_err_list,
                                                                'plot_utils/32_non_iid_setting_clique')

os.chdir('../')

# mp.rcParams['text.usetex'] = True  # Let TeX do the typsetting
# mp.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}',
#                                       r'\sansmath']  # Force sans-serif math mode (for axes labels)
# mp.rcParams['font.family'] = 'sans-serif'  # ... for regular text
# # mp.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here
# mp.rcParams['font.sans-serif'] = 'Helvetica'

# color = ["#4285F4", '#fbbc05', 'xkcd:red']
# color = ["#4285F4", '#c29103']
# labels = ["non-pri", "non-comm", "PDOM"]
# labels = ["D-BOCG", "gossip", "DB-TDOCO"]

# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42

# plt.switch_backend('agg')
# plt.rcParams['pdf.use14corefonts'] = True
# font = {'family': 'Times New Roman'}
# plt.rc('font', **font)

# plt.figure(figsize=(6.5 * 2, 3.49))

# # 画第1个图：折线图
# x=np.arange(1,100)
# plt.subplot(141)
# plt.plot(x,x*x)
# # 画第2个图：散点图
# plt.subplot(142)
# plt.scatter(np.arange(0,10), np.random.rand(10))
# # 画第3个图：饼图
# plt.subplot(143)
# plt.pie(x=[15,30,45,10],labels=list('ABCD'),autopct='%.0f',explode=[0,0.05,0,0])
# # 画第4个图：条形图
# plt.subplot(144)
# plt.bar([20,10,30,25,15],[25,15,35,30,20],color='b')
fig, axs = plt.subplots(2, 2, figsize=(6.5 * 2 + 3., 3.49 * 2 + 2))

plt.text(x=-9.03,  # 文本x轴坐标
         y=1.92,  # 文本y轴坐标
         s='$\\mathbf{Adversarial}$',  # 文本内容

         fontdict=dict(fontsize=27,
                       # family='sans-serif',
                       weight='bold'),  # 字体属性字典

         # # 添加文字背景色
         bbox={
             'facecolor': 'white',  # 填充色
             'edgecolor': 'black',  # 外框色
             #   'alpha': 0.5,  # 框透明度
             'pad': 4,  # 本文与框周围距离
         },

         rotation=90,

         )

plt.text(x=-9.03,  # 文本x轴坐标
         y=0.17,  # 文本y轴坐标
         s='$\\mathbf{Stochastic}$',  # 文本内容

         fontdict=dict(fontsize=27,
                       # family='sans-serif',
                       weight='bold'),  # 字体属性字典

         # # 添加文字背景色
         bbox={
             'facecolor': 'white',  # 填充色
             'edgecolor': 'black',  # 外框色
             #   'alpha': 0.5,  # 框透明度
             'pad': 4,  # 本文与框周围距离
         },

         rotation=90,

         )

color = ['#c29103', "#8ab4f8"]
labels = ["cycle", "clique"]
patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(labels))]
legend1 = plt.legend(handles=patches, bbox_to_anchor=(-0.35, 3.13), ncol=2, frameon=True, fontsize=23)
e1 = mlines.Line2D([], [], color='black', marker='v', markersize=9, markerfacecolor='white', markeredgewidth=2,
                   label="$\\mathtt{gossip}$", linestyle=":")
e2 = mlines.Line2D([], [], color='black', marker='^', markersize=9, markerfacecolor='white', markeredgewidth=2,
                   label="$\\mathtt{D}$-$\\mathtt{BOCG}$", linestyle="--")
e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2,
                   label="$\\mathtt{DB}$-$\\mathtt{TDOCO}$", linestyle='-')
plt.legend(handles=[e1, e2, e3], handletextpad=0.1, columnspacing=0.4, bbox_to_anchor=(0.95, 3.13), ncol=3,
           frameon=True, fontsize=23)
plt.gca().add_artist(legend1)

# plt.subplot(121)
# err_cyc = [0.4330510434595525, 0.4068829335197935, 0.3920852158634538, 0.37693268610154906, 0.36471985262478485]
# err_clique = [0.36407446751290873, 0.34584088586488815, 0.3341013159781985, 0.32188175693846816, 0.3137241555507745]
adv_setting.plot_utils.hist_curve_adv_8.draw_trend_image_test(axs[0][0],
                                                              'adv_setting/plot_utils/8_adv_setting.ini',
                                                              'adv_setting/plot_utils/8_adv_setting_clique.ini')

# plt.subplot(122)
# err_cyc = [0.29913403614457834, 0.2809321213425129, 0.2694797045324154, 0.26538484294320136, 0.260782487091222]
# err_clique = [0.2786510327022375, 0.264785391566265, 0.25934380378657484, 0.2557610800344234, 0.25284272805507746]
non_iid_setting.plot_utils.hist_curve_non_iid_8.draw_trend_image_test(axs[1][0],
                                                                      'non_iid_setting/plot_utils/8_non_iid_setting.ini',
                                                                      'non_iid_setting/plot_utils/8_non_iid_setting_clique.ini')

# err_cyc = [0.32125645438898454, 0.31079388984509465, 0.30585233792312105, 0.3022442448364888, 0.29965103270223753]
# err_clique = [0.3120895008605852, 0.3021675989672978, 0.29658880419637734, 0.29340304073436607, 0.2925357142857143]
# TDOCO_8.TDOCO_pure_plot_varying_T.hist_curve_iid.draw_trend_image_test(axs[0][2],
#                                                                err_cyc, err_clique,
#                                                                image_file_name='iid_comm_time.pdf')

labels = ["cycle", "clique"]
patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(labels))]
legend1 = plt.legend(handles=patches, bbox_to_anchor=(1.08 + 0.04, 1.31), ncol=2, frameon=True, fontsize=23)
e1 = mlines.Line2D([], [], color='black', marker='v', markersize=9, markerfacecolor='white', markeredgewidth=2,
                   label="$\\mathtt{DMA}$", linestyle=":")
e2 = mlines.Line2D([], [], color='black', marker='^', markersize=9, markerfacecolor='white', markeredgewidth=2,
                   label="$\\mathtt{DB2O}_a$", linestyle="--")
e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2,
                   label="$\\mathtt{DB2O}_c$", linestyle='-')
plt.legend(handles=[e1, e2, e3], handletextpad=0.2, columnspacing=0.6, bbox_to_anchor=(2.2 + 0.04, 1.31), ncol=3,
           frameon=True, fontsize=23)
plt.gca().add_artist(legend1)

# plt.subplot(121)
adv_setting.plot_utils.hist_curve_adv_32.draw_trend_image_test(axs[0][1], 'adv_setting/plot_utils/32_adv_setting.ini',
                                                               'adv_setting/plot_utils/32_adv_setting_clique.ini')

# plt.subplot(122)
non_iid_setting.plot_utils.hist_curve_non_iid_32.draw_trend_image_test(axs[1][1],
                                                                       'non_iid_setting/plot_utils/32_non_iid_setting.ini',
                                                                       'non_iid_setting/plot_utils/32_non_iid_setting_clique.ini')

# TDOCO_32.TDOCO_pure_plot_varying_T.hist_curve_iid.draw_trend_image_test(axs[1][2],
#                                                                # err_cyc, err_clique,
#                                                                image_file_name='iid_comm_time.pdf')

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
fig.savefig("DOCO_adv_stoc_varying_time.pdf", format="pdf", bbox_inches="tight", pad_inches=0.07)
