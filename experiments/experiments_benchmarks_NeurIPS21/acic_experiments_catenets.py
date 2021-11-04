"""
Utils to replicate ACIC2016 experiments with catenets
"""
# Author: Alicia Curth
import csv
import os
from pathlib import Path

from sklearn import clone
import numpy as np

from catenets.datasets import load
from catenets.experiment_utils.base import eval_root_mse
from catenets.models.jax import RNet, TARNet, TNet
from catenets.models.jax import RNET_NAME, TARNET_NAME, T_NAME

RESULT_DIR = Path("results/experiments_benchmarking/acic2016/")
SEP = "_"

PARAMS_DEPTH = {'n_layers_r': 3, 'n_layers_out': 2}
PARAMS_DEPTH_2 = {'n_layers_r': 3, 'n_layers_out': 2, 'n_layers_r_t': 3, 'n_layers_out_t': 2}

ALL_MODELS = {T_NAME: TNet(**PARAMS_DEPTH),
              TARNET_NAME: TARNet(**PARAMS_DEPTH),
              RNET_NAME: RNet(**PARAMS_DEPTH_2)
              }


def do_acic_experiments(n_exp: int = 10, n_reps=5, file_name: str = 'results_catenets',
                        simu_num: int = 1, models: dict = None, train_size: int = 4000,
                        pre_trans: bool = True):
    if models is None:
        models = ALL_MODELS

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    out_file = open(RESULT_DIR / (file_name + SEP + str(pre_trans) +
                    SEP + str(simu_num) + SEP + str(train_size) + '.csv'), 'w', buffering=1)
    writer = csv.writer(out_file)
    header = ['file_name', 'run', 'cate_var_in', 'cate_var_out', 'y_var_in'] + \
             [name + '_in' for name in models.keys()] + [name + '_out' for name in
                                                         models.keys()]
    writer.writerow(header)

    for i_exp in range(n_exp):
        # get data
        X, w, y, po_train, X_test, w_test, y_test, po_test = load(
            "acic2016", preprocessed=pre_trans, original_acic_outcomes=True,
            i_exp=i_exp, simu_num=simu_num, train_size=train_size)

        cate_in = po_train[:, 1] - po_train[:, 0]
        cate_out = po_test[:, 1] - po_test[:, 0]

        cate_var_in = np.var(cate_in)
        cate_var_out = np.var(cate_out)
        y_var_in = np.var(y)
        for k in range(n_reps):
            pehe_in = []
            pehe_out = []

            for model_name, estimator in models.items():
                print("Experiment {}, run {}, with {}".format(i_exp, k, model_name))
                estimator_temp = clone(estimator)
                estimator_temp.set_params(seed=k)

                # fit estimator
                estimator_temp.fit(X=X, y=y, w=w)

                cate_pred_in = estimator_temp.predict(X, return_po=False)
                cate_pred_out = estimator_temp.predict(X_test, return_po=False)

                pehe_in.append(eval_root_mse(cate_pred_in, cate_in))
                pehe_out.append(eval_root_mse(cate_pred_out, cate_out))

            writer.writerow(
                [i_exp, k, cate_var_in, cate_var_out, y_var_in] + pehe_in + pehe_out)

    out_file.close()
