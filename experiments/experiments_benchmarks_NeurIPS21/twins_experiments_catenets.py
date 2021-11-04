"""
Utils to replicate Twins experiments with catenets
"""
# Author: Alicia Curth
import os
import csv
from pathlib import Path

import numpy as onp
import pandas as pd

from sklearn import clone
from sklearn.model_selection import train_test_split

from catenets.datasets import load
from catenets.experiment_utils.base import eval_root_mse
from catenets.models.jax import RNet, TARNet, TNet
from catenets.models.jax import RNET_NAME, TARNET_NAME, T_NAME

RESULT_DIR = Path("results/experiments_benchmarking/twins/")
EXP_DIR = Path('experiments/experiments_benchmarks_NeurIPS21/twins_datasets/')
SEP = '_'

PARAMS_DEPTH = {'n_layers_r': 1, 'n_layers_out': 1}
PARAMS_DEPTH_2 = {'n_layers_r': 1, 'n_layers_out': 1, 'n_layers_r_t': 1, 'n_layers_out_t': 1}

ALL_MODELS = {T_NAME: TNet(**PARAMS_DEPTH),
              TARNET_NAME: TARNet(**PARAMS_DEPTH),
              RNET_NAME: RNet(**PARAMS_DEPTH_2)
              }


def do_twins_experiment_loop(n_train_loop=[500, 1000, 2000, 5000, None],
                             n_exp: int = 10, file_name: str = 'twins', models: dict = None,
                             test_size=0.5):
    for n in n_train_loop:
        print("Running twins experiments for subset_train {}".format(n))
        do_twins_experiments(n_exp=n_exp, file_name=file_name, models=models, subset_train=n,
                             test_size=test_size)


def do_twins_experiments(n_exp: int = 10, file_name: str = 'twins',
                         models: dict = None, subset_train: int = None,
                         prop_treated=0.5, test_size=0.5):
    if models is None:
        models = ALL_MODELS

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    out_file = open(RESULT_DIR / (file_name + SEP + str(prop_treated) + SEP + str(subset_train) + \
                    '.csv'), 'w', buffering=1)

    writer = csv.writer(out_file)
    header = [name + '_pehe' for name in models.keys()]

    writer.writerow(header)

    for i_exp in range(n_exp):
        pehe_out = []

        # get data
        X, X_t, y, w, y0_out, y1_out = prepare_twins(seed=i_exp, treat_prop=prop_treated,
                                                     subset_train=subset_train,
                                                     test_size=test_size)

        ite_out = y1_out - y0_out

        # split data
        for model_name, estimator in models.items():
            print("Experiment {} with {}".format(i_exp, model_name))
            estimator_temp = clone(estimator)
            estimator_temp.set_params(**{'binary_y': True, 'seed': i_exp})

            # fit estimator
            estimator_temp.fit(X=X, y=y, w=w)

            cate_pred_out = estimator_temp.predict(X_t)

            pehe_out.append(eval_root_mse(cate_pred_out, ite_out))

        writer.writerow(pehe_out)

    out_file.close()


# utils ---------------------------------------------------------------------
def prepare_twins(treat_prop=0.5, seed=42, test_size=0.5,
                  subset_train: int = None):
    if not os.path.isdir(EXP_DIR):
        os.makedirs(EXP_DIR)

    out_base = 'preprocessed' + SEP + str(treat_prop) + SEP + str(subset_train) + \
               SEP + str(test_size) + SEP + str(seed)
    outfile_train = EXP_DIR / (out_base + SEP + 'train.csv')
    outfile_test = EXP_DIR / (out_base + SEP + 'test.csv')

    feat_list = [
        'dmage', 'mpcb', 'cigar', 'drink', 'wtgain', 'gestat', 'dmeduc',
        'nprevist', 'dmar', 'anemia', 'cardiac', 'lung',
        'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
        'eclamp', 'incervix', 'pre4000', 'dtotord', 'preterm',
        'renal', 'rh', 'uterine', 'othermr', 'adequacy_1', 'adequacy_2',
        'adequacy_3', 'pldel_1', 'pldel_2', 'pldel_3',
        'pldel_4', 'pldel_5', 'resstatb_1', 'resstatb_2', 'resstatb_3', 'resstatb_4'
    ]

    if os.path.exists(outfile_train):
        print('Reading existing preprocessed twins file {}'.format(out_base))
        # use existing file
        df_train = pd.read_csv(outfile_train)
        X = onp.asarray(df_train[feat_list])
        y = onp.asarray(df_train[['y']]).reshape((-1,))
        w = onp.asarray(df_train[['w']]).reshape((-1,))

        df_test = pd.read_csv(outfile_test)
        X_t = onp.asarray(df_test[feat_list])
        y0_out = onp.asarray(df_test[['y0']]).reshape((-1,))
        y1_out = onp.asarray(df_test[['y1']]).reshape((-1,))
    else:
        # create file
        print('Creating preprocessed twins file {}'.format(out_base))
        onp.random.seed(seed)

        x, w, y, pos, _, _ = load('twins', seed=seed, treat_prop=treat_prop, train_ratio=1)

        X, X_t, y, y_t, w, w_t, y0_in, y0_out, y1_in, y1_out = train_test_split(x, y, w,
                                                                                pos[:, 0],
                                                                                pos[:, 1],
                                                                                test_size=test_size,
                                                                                random_state=seed)
        if subset_train is not None:
            X, y, w, y0_in, y1_in = X[:subset_train, :], y[:subset_train], w[:subset_train], \
                                    y0_in[:subset_train], y1_in[:subset_train]

        # save data
        save_df_train = pd.DataFrame(X, columns=feat_list)
        save_df_train['y0'] = y0_in
        save_df_train['y1'] = y1_in
        save_df_train['w'] = w
        save_df_train['y'] = y
        save_df_train.to_csv(outfile_train)

        save_df_train = pd.DataFrame(X_t, columns=feat_list)
        save_df_train['y0'] = y0_out
        save_df_train['y1'] = y1_out
        save_df_train['w'] = w_t
        save_df_train['y'] = y_t
        save_df_train.to_csv(outfile_test)

    return X, X_t, y, w, y0_out, y1_out
