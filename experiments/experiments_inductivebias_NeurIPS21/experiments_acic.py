"""
Utils to replicate ACIC2016 experiments (Appendix E.1)
"""
# Author: Alicia Curth
import csv
import os
from pathlib import Path

from sklearn import clone
import numpy as np

from catenets.datasets import load
from catenets.experiment_utils.base import eval_root_mse
from catenets.models.jax import TARNet, TNet, OffsetNet, FlexTENet, DRNet
from catenets.models.jax import TARNET_NAME, T_NAME, OFFSET_NAME, FLEXTE_NAME, DRNET_NAME

RESULT_DIR = Path("results/experiments_inductive_bias/acic2016/original")
SEP = "_"

PARAMS_DEPTH = {'n_layers_r': 1, 'n_layers_out': 1}
PARAMS_DEPTH_2 = {'n_layers_r': 1, 'n_layers_out': 1, 'n_layers_r_t': 1, 'n_layers_out_t': 1}
PENALTY_DIFF = 0.01
PENALTY_ORTHOGONAL = 0.1


ALL_MODELS = {
    T_NAME: TNet(**PARAMS_DEPTH),
    T_NAME
    + "_reg": TNet(train_separate=False, penalty_diff=PENALTY_DIFF, **PARAMS_DEPTH),
    TARNET_NAME: TARNet(**PARAMS_DEPTH),
    TARNET_NAME
    + "_reg": TARNet(
        reg_diff=True, penalty_diff=PENALTY_DIFF, same_init=True, **PARAMS_DEPTH
    ),
    OFFSET_NAME: OffsetNet(penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH),
    FLEXTE_NAME: FlexTENet(
        penalty_orthogonal=PENALTY_ORTHOGONAL, penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH
    ),
    FLEXTE_NAME + "_noortho_reg_same": FlexTENet(penalty_orthogonal=0, **PARAMS_DEPTH),
    DRNET_NAME: DRNet(**PARAMS_DEPTH_2),
    DRNET_NAME + "_TAR": DRNet(first_stage_strategy="Tar", **PARAMS_DEPTH_2),
}


def do_acic_orig_loop(simu_nums, n_exp: int = 10, file_name: str = 'results',
                      models: dict = None, train_size: float = 0.8):
    if models is None:
        models = ALL_MODELS
    for simu_num in simu_nums:
        print('Running simulation setting {}'.format(simu_num))
        do_acic_experiments(n_exp=n_exp, file_name=file_name, simu_num=simu_num, models=models,
                            train_size=train_size)


def do_acic_experiments(n_exp: int = 10, file_name: str = 'results_catenets',
                        simu_num: int = 1, models: dict = None, train_size: float = 0.8,
                        pre_trans: bool = False):
    if models is None:
        models = ALL_MODELS

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    out_file = open(RESULT_DIR / (file_name + SEP + str(pre_trans) +
                    SEP + str(simu_num) + SEP + str(train_size) + '.csv'), 'w', buffering=1)
    writer = csv.writer(out_file)
    header = ['file_name', 'cate_var_in', 'cate_var_out', 'y_var_in'] + \
             [name + '_in' for name in models.keys()] + [name + '_out' for name in
                                                         models.keys()]
    writer.writerow(header)

    for i_exp in range(n_exp):
        # get data
        X, w, y, po_train, X_test, w_test, y_test, po_test = load(
            "acic2016", preprocessed=pre_trans, original_acic_outcomes=True,
            keep_categorical=False, random_split=True,
            i_exp=i_exp, simu_num=simu_num, train_size=train_size)

        cate_in = po_train[:, 1] - po_train[:, 0]
        cate_out = po_test[:, 1] - po_test[:, 0]

        cate_var_in = np.var(cate_in)
        cate_var_out = np.var(cate_out)
        y_var_in = np.var(y)

        pehe_in = []
        pehe_out = []

        for model_name, estimator in models.items():
            print("Experiment {} with {}".format(i_exp, model_name))
            estimator_temp = clone(estimator)

            # fit estimator
            estimator_temp.fit(X=X, y=y, w=w)

            cate_pred_in = estimator_temp.predict(X, return_po=False)
            cate_pred_out = estimator_temp.predict(X_test, return_po=False)

            pehe_in.append(eval_root_mse(cate_pred_in, cate_in))
            pehe_out.append(eval_root_mse(cate_pred_out, cate_out))

        writer.writerow(
                [i_exp, cate_var_in, cate_var_out, y_var_in] + pehe_in + pehe_out)

    out_file.close()
