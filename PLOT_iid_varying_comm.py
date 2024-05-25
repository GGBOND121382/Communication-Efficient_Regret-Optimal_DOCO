import sys

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import iid_setting.plot_utils.PLOT_epsilon
import iid_setting_clique.plot_utils.PLOT_epsilon_clique
import iid_setting.plot_utils.PLOT_covtype
import iid_setting_clique.plot_utils.PLOT_covtype_clique

import os
import matplotlib.pyplot as mp


def draw(data_filename):
    assert data_filename in ['all', "covtype.binary", "epsilon"]

    if data_filename == "covtype.binary":
        data_filename = "covtype.libsvm.binary"
    elif data_filename == "epsilon":
        data_filename = "epsilon_normalized.all"

    # mp.rcParams['text.usetex'] = True  # Let TeX do the typsetting
    # mp.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}',
    #                                       r'\sansmath']  # Force sans-serif math mode (for axes labels)
    # mp.rcParams['font.family'] = 'sans-serif'  # ... for regular text
    # # mp.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here
    # mp.rcParams['font.sans-serif'] = 'Helvetica'

    # color = ["#4285F4", '#fbbc05', 'xkcd:red']
    # color = ["#4285F4", '#c29103', 'xkcd:red']
    # color = ["#72a4f7", '#dea604', 'xkcd:red']
    # labels = ["non-pri", "non-comm", "PDOM"]
    color = ["#8ab4f8", '#c29103', 'xkcd:red']
    labels = ["$\\mathtt{D}$-$\\mathtt{BOCG}$", "$\\mathtt{gossip}$", "$\\mathtt{DB}$-$\\mathtt{TDOCO}$"]

    # rcParams['pdf.fonttype'] = 42
    # rcParams['ps.fonttype'] = 42

    # plt.switch_backend('agg')
    # plt.rcParams['pdf.use14corefonts'] = True
    # font = {'family': 'Times New Roman'}
    # plt.rc('font', **font)

    plt.figure(figsize=(6.5 * 4 + 3, 3.49))

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

    # os.chdir('TDOCO_pure_plot_adv_32/')
    # plt.subplot(241)
    # filename = "covtype.libsvm.binary"
    # src_dir_name_32 = '32-merge-data/adv_setting/'
    # src_dir_name_8 = '8-merge-data/adv_setting/'
    # dst_name = filename.replace('.', '_') + '_adv'
    # TDOCO_pure_plot_adv_32.PLOT_covtype.plot_adv_32_8(src_dir_name_8, src_dir_name_32, dst_name + '.pdf', format='pdf')
    #
    # plt.text(x=30.,  # 文本x轴坐标
    #             y=0.28,  # 文本y轴坐标
    #             s='$\\mathbf{Adversarial\;\; DOCO}$',  # 文本内容
    #
    #             fontdict=dict(fontsize=27, family='sans-serif', weight='bold'),  # 字体属性字典
    #
    #             # # 添加文字背景色
    #             bbox={
    #                 'facecolor': 'white',  # 填充色
    #                 'edgecolor': 'black',  # 外框色
    #                 #   'alpha': 0.5,  # 框透明度
    #                 'pad': 8,  # 本文与框周围距离
    #             },
    #
    #             rotation=90,
    #
    #             )
    #
    # # text.set_color('b')
    #
    # plt.subplot(242)
    # src_dir_name_32 = '32-merge-data/adv_setting/'
    # src_dir_name_8 = '8-merge-data/adv_setting/'
    # dst_name = filename.replace('.', '_') + '_adv'
    # # plot_stoc(src_dir_name, dst_name + '.png', format='png')
    # TDOCO_pure_plot_adv_32.PLOT_epsilon.plot_adv_32_8(src_dir_name_8, src_dir_name_32, dst_name + '.pdf', format='pdf')
    #
    # plt.subplot(243)
    # src_dir_name_32 = '32-merge-data/adv_setting_clique/'
    # src_dir_name_8 = '8-merge-data/adv_setting_clique/'
    # dst_name = filename.replace('.', '_') + '_adv'
    # # plot_stoc(src_dir_name, dst_name + '.png', format='png')
    # TDOCO_pure_plot_adv_32.PLOT_covtype_clique.plot_adv_32_8(src_dir_name_8, src_dir_name_32, dst_name + '_clique.pdf', format='pdf')
    #
    # plt.subplot(244)
    # src_dir_name_32 = '32-merge-data/adv_setting_clique/'
    # src_dir_name_8 = '8-merge-data/adv_setting_clique/'
    # dst_name = filename.replace('.', '_') + '_adv'
    # # plot_stoc(src_dir_name, dst_name + '.png', format='png')
    # TDOCO_pure_plot_adv_32.PLOT_epsilon_clique.plot_adv_32_8(src_dir_name_8, src_dir_name_32, dst_name + '_clique.pdf', format='pdf')
    #
    # patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in [1, 0, 2]]
    # legend1 = plt.legend(handles=patches, bbox_to_anchor=(-1.5 + 0.5 - 0.3, 1.3),
    #                      ncol=3, frameon=True, fontsize=23)
    # # legend1 = plt.legend(handles=patches, bbox_to_anchor=(-1.5 + 0.5, 1.3),
    # #                      ncol=3, frameon=True, fontsize=27, borderpad=.25)
    # e1 = mlines.Line2D([], [], color='black',
    #                    label="8-learner", linestyle="-")
    # e2 = mlines.Line2D([], [], color='black',
    #                    label="32-learner", linestyle="--")
    # # e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9,
    # #                    label="DB-TDOCO", linestyle='-')
    # plt.legend(handles=[e1, e2], bbox_to_anchor=(0 + 0.5 - 0.3, 1.3), ncol=2, frameon=True, fontsize=23)
    # plt.gca().add_artist(legend1)

    os.chdir('iid_setting/')

    labels = ["$\\mathtt{DB2O}_a$", "$\\mathtt{DMA}$", "$\\mathtt{DB2O}_c$"]

    plt.subplot(141)
    if (data_filename == 'all' or data_filename == 'covtype.libsvm.binary'):
        src_dir_name_32 = 'plot_data_cov_32/'
        src_dir_name_8 = 'plot_data_cov_8/'
        # dst_name = filename.replace('.', '_') + '_iid'
        # plot_stoc(src_dir_name, dst_name + '.png', format='png')
        iid_setting.plot_utils.PLOT_covtype.plot_stoc_32_8(src_dir_name_8, src_dir_name_32, dst_name='iid_setting')

    # plt.text(x=6.2,  # 文本x轴坐标
    #         y=0.2,  # 文本y轴坐标
    #         s='$\\mathbf{Stochastic\;\; DOCO}$',  # 文本内容
    #
    #         fontdict=dict(fontsize=27, family='sans-serif', weight='bold'),  # 字体属性字典
    #
    #         # # 添加文字背景色
    #         bbox={
    #             'facecolor': 'white',  # 填充色
    #             'edgecolor': 'black',  # 外框色
    #             #   'alpha': 0.5,  # 框透明度
    #             'pad': 8,  # 本文与框周围距离
    #         },
    #
    #         rotation=90,
    #
    #         )

    plt.subplot(142)
    if (data_filename == 'all' or data_filename == 'epsilon_normalized.all'):
        src_dir_name_32 = 'plot_data_eps_32/'
        src_dir_name_8 = 'plot_data_eps_8/'
        # plot_stoc(src_dir_name, dst_name + '.png', format='png')
        iid_setting.plot_utils.PLOT_epsilon.plot_stoc_32_8(src_dir_name_8, src_dir_name_32, dst_name='iid_setting')

    os.chdir('../iid_setting_clique/')

    plt.subplot(143)
    if (data_filename == 'all' or data_filename == 'covtype.libsvm.binary'):
        src_dir_name_32 = 'plot_data_cov_32/'
        src_dir_name_8 = 'plot_data_cov_8/'
        # plot_stoc(src_dir_name, dst_name + '.png', format='png')
        iid_setting_clique.plot_utils.PLOT_covtype_clique.plot_stoc_32_8(src_dir_name_8, src_dir_name_32, dst_name='iid_setting_clique')

    plt.subplot(144)
    if (data_filename == 'all' or data_filename == 'epsilon_normalized.all'):
        src_dir_name_32 = 'plot_data_eps_32/'
        src_dir_name_8 = 'plot_data_eps_8/'
        # plot_stoc(src_dir_name, dst_name + '.png', format='png')
        iid_setting_clique.plot_utils.PLOT_epsilon_clique.plot_stoc_32_8(src_dir_name_8, src_dir_name_32,
                                                                         dst_name='iid_setting_clique')

    os.chdir('../')

    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in [1, 0, 2]]
    legend1 = plt.legend(handles=patches, bbox_to_anchor=(-1.5 + 0.5 - 0.43, 1.33),
                         ncol=3, frameon=True, fontsize=23, )
    # legend1 = plt.legend(handles=patches, bbox_to_anchor=(-1.5 + 0.5-0.2, 1.33),
    #                      ncol=3, frameon=True, fontsize=27, borderpad=.25)
    e1 = mlines.Line2D([], [], color='black',
                       label="8-learner", linestyle="-")
    e2 = mlines.Line2D([], [], color='black',
                       label="32-learner", linestyle="--")
    # e3 = mlines.Line2D([], [], color='black', marker='o', markersize=9,
    #                    label="DB-TDOCO", linestyle='-')
    plt.legend(handles=[e1, e2], bbox_to_anchor=(0 + 0.5 - 0.43, 1.33), ncol=2, frameon=True, fontsize=23)
    plt.gca().add_artist(legend1)

    plt.subplots_adjust(wspace=0.25, hspace=0.9)
    # plt.show()

    fig = plt.gcf()
    # fig.tight_layout()
    fig.savefig("DOCO_iid_varying_comm.pdf", format="pdf", bbox_inches="tight", pad_inches=0.2)


if __name__ == '__main__':
    data_filename = sys.argv[1]
    draw(data_filename)
