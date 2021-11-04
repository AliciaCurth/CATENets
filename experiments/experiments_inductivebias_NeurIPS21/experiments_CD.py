"""
Utils to replicate experiments C and D
"""
# Author: Alicia Curth
import csv
import os
from pathlib import Path
from typing import Optional, Union

from sklearn import clone

from catenets.datasets.dataset_ihdp import get_one_data_set, load_raw, prepare_ihdp_data
from catenets.experiment_utils.base import eval_root_mse
from catenets.models.jax import (
    DRNET_NAME,
    FLEXTE_NAME,
    OFFSET_NAME,
    T_NAME,
    TARNET_NAME,
    DRNet,
    FlexTENet,
    OffsetNet,
    TARNet,
    TNet,
)

DATA_DIR = Path("catenets/datasets/data/")
RESULT_DIR = Path("results/experiments_inductive_bias/ihdp/")
SEP = "_"

PARAMS_DEPTH: dict = {"n_layers_r": 2, "n_layers_out": 2}
PARAMS_DEPTH_2: dict = {
    "n_layers_r": 2,
    "n_layers_out": 2,
    "n_layers_r_t": 2,
    "n_layers_out_t": 2,
}
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


def do_ihdp_experiments(
    n_exp: Union[int, list] = 100,
    file_name: str = "ihdp_all",
    model_params: Optional[dict] = None,
    models: Optional[dict] = None,
    setting: str = "C",
) -> None:
    if models is None:
        models = ALL_MODELS

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    out_file = open(RESULT_DIR / (file_name + SEP + setting + ".csv"), "w", buffering=1)
    writer = csv.writer(out_file)
    header = [name + "_in" for name in models.keys()] + [
        name + "_out" for name in models.keys()
    ]
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
        pehe_in = []
        pehe_out = []

        # get data
        data_exp = get_one_data_set(data_train, i_exp=i_exp, get_po=True)
        data_exp_test = get_one_data_set(data_test, i_exp=i_exp, get_po=True)

        X, y, w, cate_true_in, X_t, cate_true_out = prepare_ihdp_data(
            data_exp, data_exp_test, setting=setting
        )

        for model_name, estimator in models.items():
            print(f"Experiment {i_exp} with {model_name}")
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
