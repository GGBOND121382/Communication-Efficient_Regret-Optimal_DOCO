import os
import iid_setting.loop_DOCO_LR_CP
import iid_setting.loop_DOCO_LR_AGD
import iid_setting.loop_DOCO_LR_DMA
import iid_setting.config_save_load

import iid_setting_clique.loop_DOCO_LR_CP
import iid_setting_clique.loop_DOCO_LR_AGD
import iid_setting_clique.loop_DOCO_LR_DMA
import iid_setting_clique.config_save_load

import non_iid_setting.loop_DOCO_LR_CP
import non_iid_setting.loop_DOCO_LR_AGD
import non_iid_setting.loop_DOCO_LR_DMA
import non_iid_setting.config_save_load

import non_iid_setting_clique.loop_DOCO_LR_CP
import non_iid_setting_clique.loop_DOCO_LR_AGD
import non_iid_setting_clique.loop_DOCO_LR_DMA
import non_iid_setting_clique.config_save_load

import adv_setting.loop_DOCO_LR_gossip
import adv_setting.loop_DOCO_LR_DB_TDOCO
import adv_setting.loop_DOCO_LR_DBOCG
import adv_setting.config_save_load

import adv_setting_clique.loop_DOCO_LR_gossip
import adv_setting_clique.loop_DOCO_LR_DB_TDOCO
import adv_setting_clique.loop_DOCO_LR_DBOCG
import adv_setting_clique.config_save_load

import data.data_split
import data.config_save_load
import sys

if __name__ == '__main__':
    data_filename = sys.argv[1]
    num_learners = int(sys.argv[2])
    rpt_times = int(sys.argv[3])

    assert data_filename in ['all', "covtype.binary", "epsilon"]

    if data_filename == "covtype.binary":
        data_filename = "covtype.libsvm.binary"
    elif data_filename == "epsilon":
        data_filename = "epsilon_normalized.all"

    if data_filename == 'all':
        for data_filename in ["covtype.binary", "epsilon"]:
            os.chdir('data')
            data.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

            os.chdir('../iid_setting')
            iid_setting.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

            os.chdir('../iid_setting_clique')
            iid_setting_clique.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

            os.chdir('../non_iid_setting')
            non_iid_setting.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

            os.chdir('../non_iid_setting_clique')
            non_iid_setting_clique.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

            os.chdir('../adv_setting')
            adv_setting.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

            os.chdir('../adv_setting_clique')
            adv_setting_clique.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)
            os.chdir('../')

            for i in range(rpt_times):
                os.chdir('data')
                data.data_split.run(feedback_setting='all')

                os.chdir('../iid_setting')
                iid_setting.loop_DOCO_LR_CP.run_comm()
                iid_setting.loop_DOCO_LR_AGD.run_comm()
                iid_setting.loop_DOCO_LR_DMA.run_comm()

                os.chdir('../iid_setting_clique')
                iid_setting_clique.loop_DOCO_LR_CP.run_comm()
                iid_setting_clique.loop_DOCO_LR_AGD.run_comm()
                iid_setting_clique.loop_DOCO_LR_DMA.run_comm()

                os.chdir('../non_iid_setting')
                non_iid_setting.loop_DOCO_LR_CP.run_comm()
                non_iid_setting.loop_DOCO_LR_AGD.run_comm()
                non_iid_setting.loop_DOCO_LR_DMA.run_comm()

                os.chdir('../non_iid_setting_clique')
                non_iid_setting_clique.loop_DOCO_LR_CP.run_comm()
                non_iid_setting_clique.loop_DOCO_LR_AGD.run_comm()
                non_iid_setting_clique.loop_DOCO_LR_DMA.run_comm()

                os.chdir('../adv_setting')
                adv_setting.loop_DOCO_LR_gossip.run_comm()
                adv_setting.loop_DOCO_LR_DB_TDOCO.run_comm()
                adv_setting.loop_DOCO_LR_DBOCG.run_comm()

                os.chdir('../adv_setting_clique')
                adv_setting_clique.loop_DOCO_LR_gossip.run_comm()
                adv_setting_clique.loop_DOCO_LR_DB_TDOCO.run_comm()
                adv_setting_clique.loop_DOCO_LR_DBOCG.run_comm()
                os.chdir('../')
    else:
        os.chdir('data')
        data.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

        os.chdir('../iid_setting')
        iid_setting.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

        os.chdir('../iid_setting_clique')
        iid_setting_clique.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

        os.chdir('../non_iid_setting')
        non_iid_setting.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

        os.chdir('../non_iid_setting_clique')
        non_iid_setting_clique.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

        os.chdir('../adv_setting')
        adv_setting.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)

        os.chdir('../adv_setting_clique')
        adv_setting_clique.config_save_load.config_data_numLearners_rpt(data_filename, num_learners)
        os.chdir('../')

        for i in range(rpt_times):
            os.chdir('data')
            data.data_split.run(feedback_setting='all')

            os.chdir('../iid_setting')
            iid_setting.loop_DOCO_LR_CP.run_comm()
            iid_setting.loop_DOCO_LR_AGD.run_comm()
            iid_setting.loop_DOCO_LR_DMA.run_comm()

            os.chdir('../iid_setting_clique')
            iid_setting_clique.loop_DOCO_LR_CP.run_comm()
            iid_setting_clique.loop_DOCO_LR_AGD.run_comm()
            iid_setting_clique.loop_DOCO_LR_DMA.run_comm()

            os.chdir('../non_iid_setting')
            non_iid_setting.loop_DOCO_LR_CP.run_comm()
            non_iid_setting.loop_DOCO_LR_AGD.run_comm()
            non_iid_setting.loop_DOCO_LR_DMA.run_comm()

            os.chdir('../non_iid_setting_clique')
            non_iid_setting_clique.loop_DOCO_LR_CP.run_comm()
            non_iid_setting_clique.loop_DOCO_LR_AGD.run_comm()
            non_iid_setting_clique.loop_DOCO_LR_DMA.run_comm()

            os.chdir('../adv_setting')
            adv_setting.loop_DOCO_LR_gossip.run_comm()
            adv_setting.loop_DOCO_LR_DB_TDOCO.run_comm()
            adv_setting.loop_DOCO_LR_DBOCG.run_comm()

            os.chdir('../adv_setting_clique')
            adv_setting_clique.loop_DOCO_LR_gossip.run_comm()
            adv_setting_clique.loop_DOCO_LR_DB_TDOCO.run_comm()
            adv_setting_clique.loop_DOCO_LR_DBOCG.run_comm()
            os.chdir('../')

