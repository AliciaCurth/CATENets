"""
Utils to replicate Twins experiments (Appendix E.2)
"""
# Author: Alicia Curth
import csv
import os
from pathlib import Path

from sklearn import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

from catenets.datasets import load
from catenets.models.jax.base import check_shape_1d_data
from catenets.experiment_utils.base import eval_root_mse
from catenets.models.jax import TARNet, TNet, OffsetNet, FlexTENet, DRNet
from catenets.models.jax import TARNET_NAME, T_NAME, OFFSET_NAME, FLEXTE_NAME, DRNET_NAME

RESULT_DIR = Path("results/experiments_inductive_bias/twins")
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


def do_twins_experiment_loop(n_train_loop=[500, 1000, 2000, 5000, None],
                             prop_loop=[0.1, 0.25, 0.5, 0.75, 0.9],
                             n_exp: int = 10, file_name: str = 'twins', models: dict = None,
                             test_size=0.5):
    for n in n_train_loop:
        for prop in prop_loop:
            print('Running twins experiment for {} training samples with {} treated.'.format(n,
                                                                                             prop))
            do_twins_experiments(n_exp=n_exp, file_name=file_name, models=models, subset_train=n,
                                 prop_treated=prop, test_size=test_size)


def do_twins_experiments(n_exp: int = 10, file_name: str = 'twins',
                         models: dict = None, subset_train: int = None, prop_treated=0.5,
                         test_size=0.5):
    if models is None:
        models = ALL_MODELS

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    out_file = open(RESULT_DIR / (file_name + SEP + str(prop_treated) + SEP + str(subset_train) +
                    '.csv'), 'w', buffering=1)
    writer = csv.writer(out_file)
    header = [name + '_cate' for name in models.keys()] + \
             [name + '_auc_ite' for name in models.keys() if 'R' not in name and 'X' not in
              name] + \
             [name + '_auc_mu0' for name in models.keys() if 'R' not in name and 'X' not in
              name] + \
             [name + '_auc_mu1' for name in models.keys() if 'R' not in name and 'X' not in
              name] + [name + '_ap_mu0' for name in models.keys() if 'R' not in name and 'X'
                       not in name] + \
             [name + '_ap_mu1' for name in models.keys() if 'R' not in name and 'X' not in name]

    writer.writerow(header)

    for i_exp in range(n_exp):
        pehe_out = []
        auc_ite = []
        auc_mu0 = []
        auc_mu1 = []
        ap_mu0 = []
        ap_mu1 = []

        # get data
        x, w, y, pos, _, _ = load('twins', seed=i_exp, treat_prop=prop_treated, train_ratio=1)

        # split data
        X, X_t, y, y_t, w, w_t, y0_in, y0_out, y1_in, y1_out = split_data(x, y, w, pos,
                                                                          random_state=i_exp,
                                                                          subset_train=subset_train,
                                                                          test_size=test_size)

        ite_out = y1_out - y0_out

        ite_out_encoded = label_binarize(ite_out, [-1, 0, 1])

        n_test = X_t.shape[0]

        # split data
        for model_name, estimator in models.items():
            print("Experiment {} with {}".format(i_exp, model_name))
            estimator_temp = clone(estimator)
            estimator_temp.set_params(**{'binary_y': True})

            # fit estimator
            estimator_temp.fit(X=X, y=y, w=w)

            if 'DR' not in model_name and 'R' not in model_name and 'X' not in model_name:
                cate_pred_out, mu0_pred, mu1_pred = estimator_temp.predict(X_t, return_po=True)

                # create probabilities for each possible level of ITE
                probs = np.zeros((n_test, 3))
                probs[:, 0] = (mu0_pred * (1 - mu1_pred)).reshape((-1,))  # P(Y1-Y0=-1)
                probs[:, 1] = ((mu0_pred * mu1_pred) + ((1 - mu0_pred) * (1 - mu1_pred))).reshape(
                    (-1,))  # P(Y1-Y0=0)
                probs[:, 2] = (mu1_pred * (1 - mu0_pred)).reshape((-1,))  # P(Y1-Y0=1)
                auc_ite.append(roc_auc_score(ite_out_encoded, probs))

                # evaluate performance on potential outcomes
                auc_mu0.append(eval_roc_auc(y0_out, mu0_pred))
                auc_mu1.append(eval_roc_auc(y1_out, mu1_pred))
                ap_mu0.append(eval_ap(y0_out, mu0_pred))
                ap_mu1.append(eval_ap(y1_out, mu1_pred))
            else:
                cate_pred_out = estimator_temp.predict(X_t)

            pehe_out.append(eval_root_mse(cate_pred_out, ite_out))

        writer.writerow(pehe_out + auc_ite + auc_mu0 + auc_mu1 + ap_mu0 + ap_mu1)

    out_file.close()


# utils -------
def split_data(X, y, w, pos, test_size=0.5, random_state=42,
               subset_train: int = None):
    X, X_t, y, y_t, w, w_t, y0_in, \
        y0_out, y1_in, y1_out = train_test_split(X, y, w, pos[:, 0], pos[:, 1],
                                                 test_size=test_size,
                                                 random_state=random_state)
    if subset_train is not None:
        X, y, w, y0_in, y1_in = X[:subset_train, :], y[:subset_train], w[:subset_train], \
                                y0_in[:subset_train], y1_in[:subset_train]

    return X, X_t, y, y_t, w, w_t, y0_in, y0_out, y1_in, y1_out


def eval_roc_auc(targets, preds):
    preds = check_shape_1d_data(preds)
    targets = check_shape_1d_data(targets)
    return roc_auc_score(targets, preds)


def eval_ap(targets, preds):
    preds = check_shape_1d_data(preds)
    targets = check_shape_1d_data(targets)
    return average_precision_score(targets, preds)
