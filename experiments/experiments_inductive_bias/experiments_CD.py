"""
Author: Alicia Curth
Utils to replicate experiments C and D
"""
import numpy as onp
import csv
import os

from sklearn import clone

from catenets.experiment_utils.base import eval_root_mse
from catenets.models import TARNet, DRNet, TNet, FlexTENet, OffsetNet
from catenets.models import TARNET_NAME, DRNET_NAME, T_NAME, FLEXTE_NAME, OFFSET_NAME

DATA_DIR = 'data/ihdp/'
IHDP_TRAIN_NAME = 'ihdp_npci_1-100.train.npz'
IHDP_TEST_NAME = 'ihdp_npci_1-100.test.npz'
RESULT_DIR = 'results/experiments_inductive_bias/ihdp/'
SEP = '_'

PARAMS_DEPTH = {'n_layers_r': 2, 'n_layers_out': 2}
PARAMS_DEPTH_2 = {'n_layers_r': 2, 'n_layers_out': 2, 'n_layers_r_t': 2, 'n_layers_out_t': 2}
PENALTY_DIFF = 0.01
PENALTY_ORTHOGONAL = 0.1

ALL_MODELS = {T_NAME: TNet(**PARAMS_DEPTH),
              T_NAME + '_reg': TNet(train_separate=False, penalty_diff=PENALTY_DIFF,
                                    **PARAMS_DEPTH),
              TARNET_NAME: TARNet(**PARAMS_DEPTH),
              TARNET_NAME + '_reg': TARNet(reg_diff=True, penalty_diff=PENALTY_DIFF,
                                           same_init=True, **PARAMS_DEPTH),
              OFFSET_NAME: OffsetNet(penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH),
              FLEXTE_NAME: FlexTENet(penalty_orthogonal=PENALTY_ORTHOGONAL,
                                     penalty_l2_p=PENALTY_DIFF,
                                     **PARAMS_DEPTH),
              FLEXTE_NAME + '_noortho_reg_same': FlexTENet(penalty_orthogonal=0,
                                                           **PARAMS_DEPTH),
              DRNET_NAME: DRNet(**PARAMS_DEPTH_2),
              DRNET_NAME + '_TAR': DRNet(first_stage_strategy='Tar', **PARAMS_DEPTH_2)
              }


def do_ihdp_experiments(n_exp: int = 100, file_name: str = 'ihdp_all',
                        model_params: dict = None, models: dict = None, setting='C'):
    if models is None:
        models = ALL_MODELS

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    out_file = open(RESULT_DIR + file_name + SEP + setting + '.csv', 'w', buffering=1)
    writer = csv.writer(out_file)
    header = [name + '_in' for name in models.keys()] + [name + '_out' for \
                                                       name in models.keys()]
    writer.writerow(header)

    # get data
    data_train = load_data_npz(DATA_DIR + IHDP_TRAIN_NAME, get_po=True)
    data_test = load_data_npz(DATA_DIR + IHDP_TEST_NAME, get_po=True)

    if isinstance(n_exp, int):
        experiment_loop = range(1, n_exp + 1)
    elif isinstance(n_exp, list):
        experiment_loop = n_exp
    else:
        raise ValueError('n_exp should be either an integer or a list of integers.')

    for i_exp in experiment_loop:
        pehe_in = []
        pehe_out = []

        # get data
        data_exp = get_one_data_set(data_train, i_exp=i_exp, get_po=True)
        data_exp_test = get_one_data_set(data_test, i_exp=i_exp, get_po=True)

        X, y, w, cate_true_in, X_t, cate_true_out = prepare_ihdp_data(data_exp, data_exp_test,
                                                                      setting=setting)

        for model_name, estimator in models.items():
            print("Experiment {} with {}".format(i_exp, model_name))
            estimator_temp = clone(estimator)
            if model_params is not None:
                estimator_temp.set_params(**model_params)

            # fit estimator
            estimator_temp.fit(X=X, y=y, w=w)

            cate_pred_in = estimator_temp.predict(X, return_po=False)
            cate_pred_out = estimator_temp.predict(X_t, return_po=False)

            pehe_in.append(eval_root_mse(cate_pred_in, cate_true_in))
            pehe_out.append(eval_root_mse(cate_pred_out, cate_true_out))

        writer.writerow(pehe_in + pehe_out)

    out_file.close()


# data utils -------------------------------------------------------------------
def load_data_npz(fname, get_po: bool = True):
    """ Load data set (adapted from https://github.com/clinicalml/cfrnet)"""
    if fname[-3:] == 'npz':
        data_in = onp.load(fname)
        data = {'X': data_in['x'], 'w': data_in['t'], 'y': data_in['yf']}
        try:
            data['ycf'] = data_in['ycf']
        except:
            data['ycf'] = None
    else:
        raise ValueError('This loading function is only for npz files.')

    if get_po:
        data['mu0'] = data_in['mu0']
        data['mu1'] = data_in['mu1']

    data['HAVE_TRUTH'] = not data['ycf'] is None
    data['dim'] = data['X'].shape[1]
    data['n'] = data['X'].shape[0]

    return data


def get_one_data_set(D, i_exp, get_po: bool = True):
    """ Get data for one experiment. Adapted from https://github.com/clinicalml/cfrnet"""
    D_exp = {}
    D_exp['X'] = D['X'][:, :, i_exp - 1]
    D_exp['w'] = D['w'][:, i_exp - 1:i_exp]
    D_exp['y'] = D['y'][:, i_exp - 1:i_exp]
    if D['HAVE_TRUTH']:
        D_exp['ycf'] = D['ycf'][:, i_exp - 1:i_exp]
    else:
        D_exp['ycf'] = None

    if get_po:
        D_exp['mu0'] = D['mu0'][:, i_exp - 1:i_exp]
        D_exp['mu1'] = D['mu1'][:, i_exp - 1:i_exp]

    return D_exp


def prepare_ihdp_data(data_train, data_test, setting='C', return_pos=False):
    if setting == 'C':
        X, y, w, mu0, mu1 = data_train['X'], data_train['y'], data_train['w'], data_train['mu0'], \
                            data_train['mu1']

        X_t, y_t, w_t, mu0_t, mu1_t = data_test['X'], data_test['y'], data_test['w'], \
                                      data_test['mu0'], data_test['mu1']

    elif setting == 'D':
        X, y, w, mu0, mu1 = data_train['X'], data_train['y'], data_train['w'], data_train['mu0'], \
                            data_train['mu1']

        X_t, y_t, w_t, mu0_t, mu1_t = data_test['X'], data_test['y'], data_test['w'], \
                                      data_test['mu0'], data_test['mu1']
        y[w == 1] = y[w == 1] + mu0[w == 1]
        mu1 = mu0 + mu1
        mu1_t = mu0_t + mu1_t

    cate_true_in = mu1 - mu0
    cate_true_out = mu1_t - mu0_t

    if return_pos:
        return X, y, w, cate_true_in, X_t, cate_true_out, mu0, mu1, mu0_t, mu1_t

    return X, y, w, cate_true_in, X_t, cate_true_out
