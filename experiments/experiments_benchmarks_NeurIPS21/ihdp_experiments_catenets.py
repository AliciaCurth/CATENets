"""
Utils to replicate IHDP experiments with catenets
"""
# Author: Alicia Curth
import csv
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn import clone

from catenets.datasets.dataset_ihdp import get_one_data_set, load_raw, prepare_ihdp_data
from catenets.experiment_utils.base import eval_root_mse
from catenets.models.jax import RNet, TARNet, TNet
from catenets.models.jax import RNET_NAME, TARNET_NAME, T_NAME

DATA_DIR = Path("catenets/datasets/data/")
RESULT_DIR = Path("results/experiments_benchmarking/ihdp/")
SEP = "_"

PARAMS_DEPTH = {'n_layers_r': 3, 'n_layers_out': 2}
PARAMS_DEPTH_2 = {'n_layers_r': 3, 'n_layers_out': 2, 'n_layers_r_t': 3, 'n_layers_out_t': 2}

ALL_MODELS = {T_NAME: TNet(**PARAMS_DEPTH),
              TARNET_NAME: TARNet(**PARAMS_DEPTH),
              RNET_NAME: RNet(**PARAMS_DEPTH_2)
              }


def do_ihdp_experiments(
    n_exp: Union[int, list] = 100,
    n_reps: int = 5,
    file_name: str = "ihdp_all",
    model_params: Optional[dict] = None,
    models: Optional[dict] = None,
    setting: str = "original",
) -> None:
    if models is None:
        models = ALL_MODELS

    if (setting == 'original') or (setting == 'C'):
        setting = 'C'
    elif (setting == 'modified') or (setting == 'D'):
        setting = 'D'
    else:
        raise ValueError('Setting should be one of original or modified. You passed {}.'.format(
            setting))

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    out_file = open(RESULT_DIR / (file_name + SEP + setting + ".csv"), "w", buffering=1)
    writer = csv.writer(out_file)
    header = ['exp', 'run', 'cate_var_in', 'cate_var_out', 'y_var_in'] + \
             [name + "_in" for name in models.keys()] + \
             [name + "_out" for name in models.keys()]
    writer.writerow(header)

    # get data
    data_train, data_test = load_raw(DATA_DIR)

    if isinstance(n_exp, int):
        experiment_loop = list(range(1, n_exp + 1))
    elif isinstance(n_exp, list):
        experiment_loop = n_exp
    else:
        raise ValueError("n_exp should be either an integer or a list of integers.")

    for i_exp in experiment_loop:
        # get data
        data_exp = get_one_data_set(data_train, i_exp=i_exp, get_po=True)
        data_exp_test = get_one_data_set(data_test, i_exp=i_exp, get_po=True)

        X, y, w, cate_true_in, X_t, cate_true_out = prepare_ihdp_data(
            data_exp, data_exp_test, setting=setting
        )

        # compute some stats
        cate_var_in = np.var(cate_true_in)
        cate_var_out = np.var(cate_true_out)
        y_var_in = np.var(y)

        for k in range(n_reps):
            pehe_in = []
            pehe_out = []

            for model_name, estimator in models.items():
                print(f"Experiment {i_exp}, run {k}, with {model_name}")
                estimator_temp = clone(estimator)
                estimator_temp.set_params(seed=k)
                if model_params is not None:
                    estimator_temp.set_params(**model_params)

                # fit estimator
                estimator_temp.fit(X=X, y=y, w=w)

                cate_pred_in = estimator_temp.predict(X, return_po=False)
                cate_pred_out = estimator_temp.predict(X_t, return_po=False)

                pehe_in.append(eval_root_mse(cate_pred_in, cate_true_in))
                pehe_out.append(eval_root_mse(cate_pred_out, cate_true_out))

            writer.writerow([i_exp, k,  cate_var_in, cate_var_out, y_var_in] + pehe_in + pehe_out)

    out_file.close()
