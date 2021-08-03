"""
Utils to replicate setups A & B
"""
# Author: Alicia Curth
import csv
import os
from typing import Optional, Tuple, Union

import numpy as onp
import pandas as pd
from sklearn import clone

from catenets.experiment_utils.base import eval_root_mse
from catenets.models.jax import (
    DRAGON_NAME,
    DRNET_NAME,
    FLEXTE_NAME,
    OFFSET_NAME,
    RANET_NAME,
    RNET_NAME,
    SNET_NAME,
    T_NAME,
    TARNET_NAME,
    XNET_NAME,
    DragonNet,
    DRNet,
    FlexTENet,
    OffsetNet,
    RANet,
    RNet,
    SNet,
    TARNet,
    TNet,
    XNet,
)

DATA_DIR = "data/acic2016/"
ACIC_COV = "x.csv"
ACIC_COV_TRANS = "x_trans.csv"
RESULT_DIR_SIMU = "results/experiments_inductive_bias/acic2016/simulations/"
SEP = "_"

NUMERIC_COLS = [
    0,
    3,
    4,
    16,
    17,
    18,
    20,
    21,
    22,
    24,
    24,
    25,
    30,
    31,
    32,
    33,
    39,
    40,
    41,
    53,
    54,
]
N_NUM_COLS = len(NUMERIC_COLS)

# Hyperparms for all models
PARAMS_DEPTH: dict = {"n_layers_r": 1, "n_layers_out": 1}
PARAMS_DEPTH_2: dict = {
    "n_layers_r": 1,
    "n_layers_out": 1,
    "n_layers_r_t": 1,
    "n_layers_out_t": 1,
}
PENALTY_DIFF = 0.01
PENALTY_ORTHOGONAL = 0.1

# For main results
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

# For figure 4 in main text
ABLATIONS = {
    T_NAME: TNet(**PARAMS_DEPTH),
    T_NAME
    + "_reg": TNet(train_separate=False, penalty_diff=PENALTY_DIFF, **PARAMS_DEPTH),
    T_NAME + "_reg_same": TNet(train_separate=False, **PARAMS_DEPTH),
    OFFSET_NAME: OffsetNet(penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH),
    OFFSET_NAME + "_reg_same": OffsetNet(**PARAMS_DEPTH),
    FLEXTE_NAME: FlexTENet(
        penalty_orthogonal=PENALTY_ORTHOGONAL, penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH
    ),
    FLEXTE_NAME
    + "_reg_same": FlexTENet(penalty_orthogonal=PENALTY_ORTHOGONAL, **PARAMS_DEPTH),
    FLEXTE_NAME
    + "_noortho": FlexTENet(
        penalty_orthogonal=0, penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH
    ),
    FLEXTE_NAME + "_noortho_reg_same": FlexTENet(penalty_orthogonal=0, **PARAMS_DEPTH),
}

# For results in appendix D.1
TWOSTEP_LEARNERS = {
    XNET_NAME: XNet(**PARAMS_DEPTH_2),
    RANET_NAME: RANet(**PARAMS_DEPTH_2),
    RNET_NAME: RNet(**PARAMS_DEPTH_2),
    DRNET_NAME: DRNet(**PARAMS_DEPTH_2),
    T_NAME: TNet(**PARAMS_DEPTH),
}

# For results in Appendix D.2
DRAGON_VARIANTS = {
    DRAGON_NAME: DragonNet(**PARAMS_DEPTH),
    DRAGON_NAME
    + "_reg": DragonNet(
        reg_diff=True, penalty_diff=PENALTY_DIFF, same_init=True, **PARAMS_DEPTH
    ),
}

SNET_VARIANTS = {
    SNET_NAME: SNet(
        n_units_r=100,
        n_units_r_small=100,
        ortho_reg_type="fro",
        penalty_orthogonal=PENALTY_ORTHOGONAL,
        with_prop=False,
        **PARAMS_DEPTH,
    ),
    SNET_NAME
    + "_reg": SNet(
        n_units_r=100,
        n_units_r_small=100,
        ortho_reg_type="fro",
        penalty_orthogonal=PENALTY_ORTHOGONAL,
        with_prop=False,
        penalty_diff=PENALTY_DIFF,
        same_init=True,
        reg_diff=True,
        **PARAMS_DEPTH,
    ),
}

# For results in appendix D.6
DR_VARIANTS = {
    DRNET_NAME
    + "_t_reg": DRNet(
        first_stage_args={"train_separate": False, "penalty_diff": PENALTY_DIFF},
        **PARAMS_DEPTH_2,
    ),
    DRNET_NAME
    + "_Flex": DRNet(
        first_stage_strategy="Flex",
        first_stage_args={
            "private_out": False,
            "penalty_orthogonal": PENALTY_ORTHOGONAL,
            "penalty_l2_p": PENALTY_DIFF,
            "normalize_ortho": False,
        },
        **PARAMS_DEPTH_2,
    ),
}

# results in appendix D.6
X_VARIANTS = {
    XNET_NAME
    + "_t_reg": XNet(
        first_stage_args={"train_separate": False, "penalty_diff": PENALTY_DIFF},
        **PARAMS_DEPTH_2,
    ),
    XNET_NAME
    + "_Flex": XNet(
        first_stage_strategy="Flex",
        first_stage_args={
            "private_out": False,
            "penalty_orthogonal": PENALTY_ORTHOGONAL,
            "penalty_l2_p": PENALTY_DIFF,
            "normalize_ortho": False,
        },
        **PARAMS_DEPTH_2,
    ),
}


def do_acic_simu_loops(
    rho_loop: list = [0, 0.05, 0.1, 0.2, 0.5, 0.8],
    n1_loop: list = [200, 2000, 500],
    n_exp: int = 10,
    file_name: str = "acic_simu",
    models: Optional[dict] = None,
    n_0: int = 2000,
    n_test: int = 500,
    setting: str = "A",
) -> None:
    if models is None:
        models = ALL_MODELS

    for n_1 in n1_loop:
        if setting == "A":
            for rho in rho_loop:
                do_acic_simu(
                    n_1=n_1,
                    n_exp=n_exp,
                    file_name=file_name,
                    models=models,
                    n_0=n_0,
                    n_test=n_test,
                    prop_omega=0,
                    prop_gamma=rho,
                )
        else:
            for rho in rho_loop:
                do_acic_simu(
                    n_1=n_1,
                    n_exp=n_exp,
                    file_name=file_name,
                    models=models,
                    n_0=n_0,
                    n_test=n_test,
                    prop_gamma=0,
                    prop_omega=rho,
                )


def do_acic_simu_loop_n1(
    n1_loop: list,
    n_exp: int = 10,
    file_name: str = "acic_simu",
    models: Optional[dict] = None,
    n_0: int = 2000,
    n_test: int = 500,
    prop_gamma: float = 0,
    prop_omega: float = 0,
) -> None:
    for n in n1_loop:
        do_acic_simu(
            n_exp=n_exp,
            file_name=file_name,
            models=models,
            n_0=n_0,
            n_1=n,
            n_test=n_test,
            prop_gamma=prop_gamma,
            prop_omega=prop_omega,
        )


def do_acic_simu(
    n_exp: Union[int, list] = 10,
    file_name: str = "acic_simu",
    models: Union[dict, str, None] = None,
    n_0: int = 2000,
    n_1: int = 200,
    n_test: int = 500,
    error_sd: float = 1,
    sp_lin: float = 0.6,
    sp_nonlin: float = 0.3,
    prop_gamma: float = 0,
    ate_goal: float = 0,
    inter: bool = True,
    prop_omega: float = 0,
) -> None:
    if models is None:
        models = ALL_MODELS
    elif isinstance(models, str):
        if models == "all":
            models = ALL_MODELS
        elif models == "ablations":
            models = ABLATIONS
        elif models == "snet":
            models = SNET_VARIANTS
        elif models == "dragon":
            models = DRAGON_VARIANTS
        elif models == "twostep":
            models = TWOSTEP_LEARNERS
        elif models == "dr":
            models = DR_VARIANTS
        elif models == "x":
            models = X_VARIANTS
        else:
            raise ValueError(f"{models} is not a valid model selection string.")

    # get file to write in
    if not os.path.isdir(RESULT_DIR_SIMU):
        os.makedirs(RESULT_DIR_SIMU)

    out_file = open(
        RESULT_DIR_SIMU
        + file_name
        + SEP
        + str(n_0)
        + SEP
        + str(n_1)
        + SEP
        + str(prop_gamma)
        + SEP
        + str(prop_omega)
        + ".csv",
        "w",
        buffering=1,
    )
    writer = csv.writer(out_file)
    header = (
        ["y_var", "cate_var"]
        + [name + "_cate" for name in models.keys()]
        + [
            name + "_mu0"
            for name in models.keys()
            if "DR" not in name and "X" not in name
        ]
        + [
            name + "_mu1"
            for name in models.keys()
            if "DR" not in name and "X" not in name
        ]
    )
    writer.writerow(header)

    if isinstance(n_exp, int):
        experiment_loop = list(range(1, n_exp + 1))
    elif isinstance(n_exp, list):
        experiment_loop = n_exp
    else:
        raise ValueError("n_exp should be either an integer or a list of integers.")

    # get data
    X_full = pd.read_csv(DATA_DIR + ACIC_COV_TRANS).iloc[:, 1:].values

    for i_exp in experiment_loop:
        rmse_cate = []
        rmse_mu0 = []
        rmse_mu1 = []

        # get data
        X, y, w, X_t, mu_0_t, mu_1_t, cate_t = acic_simu(
            X_full,
            i_exp,
            n_0=n_0,
            n_1=n_1,
            n_test=n_test,
            error_sd=error_sd,
            sp_lin=sp_lin,
            sp_nonlin=sp_nonlin,
            prop_gamma=prop_gamma,
            ate_goal=ate_goal,
            inter=inter,
            prop_omega=prop_omega,
        )

        y_var = onp.var(y)
        cate_var = onp.var(cate_t)

        # split data
        for model_name, estimator in models.items():
            print(f"Experiment {i_exp} with {model_name}")
            estimator_temp = clone(estimator)

            # fit estimator
            estimator_temp.fit(X=X, y=y, w=w)

            if (
                "DR" not in model_name
                and "R" not in model_name
                and "X" not in model_name
            ):
                cate_pred_out, mu0_pred, mu1_pred = estimator_temp.predict(
                    X_t, return_po=True
                )
                rmse_mu0.append(eval_root_mse(mu0_pred, mu_0_t))
                rmse_mu1.append(eval_root_mse(mu1_pred, mu_1_t))
            else:
                cate_pred_out = estimator_temp.predict(X_t)

            rmse_cate.append(eval_root_mse(cate_pred_out, cate_t))

        writer.writerow([y_var, cate_var] + rmse_cate + rmse_mu0 + rmse_mu1)

    out_file.close()


def acic_simu(
    X: onp.ndarray,
    i_exp: onp.ndarray,
    n_0: int = 2000,
    n_1: int = 200,
    n_test: int = 500,
    error_sd: float = 1,
    sp_lin: float = 0.6,
    sp_nonlin: float = 0.3,
    prop_gamma: float = 0,
    prop_omega: float = 0,
    ate_goal: float = 0,
    inter: bool = True,
) -> Tuple:
    onp.random.seed(i_exp)

    # shuffle indices
    n_total, n_cov = X.shape
    ind = onp.arange(n_total)
    onp.random.shuffle(ind)
    ind_test = ind[-n_test:]
    ind_1 = ind[n_0 : (n_0 + n_1)]

    # create treatment indicator (treatment assignment does not matter in test set)
    w = onp.zeros(n_total).reshape((-1, 1))
    w[ind_1] = 1

    # create dgp
    coeffs_ = [0, 1]
    # sample baseline coefficients
    beta_0 = onp.random.choice(
        coeffs_, size=n_cov, replace=True, p=[1 - sp_lin, sp_lin]
    )
    intercept = onp.random.choice([x for x in onp.arange(-1, 1.25, 0.25)])

    # sample treatment effect coefficients
    gamma = onp.random.choice(
        coeffs_, size=n_cov, replace=True, p=[1 - prop_gamma, prop_gamma]
    )
    omega = onp.random.choice(
        [0, 1], replace=True, size=n_cov, p=[prop_omega, 1 - prop_omega]
    )

    # simulate mu_0 and mu_1
    mu_0 = (intercept + onp.dot(X, beta_0)).reshape((-1, 1))
    mu_1 = (intercept + onp.dot(X, gamma + beta_0 * omega)).reshape((-1, 1))
    if sp_nonlin > 0:
        coefs_sq = [0, 0.1]
        beta_sq = onp.random.choice(
            coefs_sq, size=N_NUM_COLS, replace=True, p=[1 - sp_nonlin, sp_nonlin]
        )
        omega = onp.random.choice(
            [0, 1], replace=True, size=N_NUM_COLS, p=[prop_omega, 1 - prop_omega]
        )
        X_sq = X[:, NUMERIC_COLS] ** 2
        mu_0 = mu_0 + onp.dot(X_sq, beta_sq).reshape((-1, 1))
        mu_1 = mu_1 + onp.dot(X_sq, beta_sq * omega).reshape((-1, 1))

        if inter:
            # randomly add some interactions
            ind_c = onp.arange(n_cov)
            onp.random.shuffle(ind_c)
            inter_list = list()
            for i in range(0, n_cov - 2, 2):
                inter_list.append(X[:, ind_c[i]] * X[:, ind_c[i + 1]])

            X_inter = onp.array(inter_list).T
            n_inter = X_inter.shape[1]
            beta_inter = onp.random.choice(
                coefs_sq, size=n_inter, replace=True, p=[1 - sp_nonlin, sp_nonlin]
            )
            omega = onp.random.choice(
                [0, 1], replace=True, size=n_inter, p=[prop_omega, 1 - prop_omega]
            )
            mu_0 = mu_0 + onp.dot(X_inter, beta_inter).reshape((-1, 1))
            mu_1 = mu_1 + onp.dot(X_inter, beta_inter * omega).reshape((-1, 1))

    ate = onp.mean(mu_1 - mu_0)
    mu_1 = mu_1 - ate + ate_goal

    y = (
        w * mu_1
        + (1 - w) * mu_0
        + onp.random.normal(0, error_sd, n_total).reshape((-1, 1))
    )

    cate = mu_1 - mu_0

    X_train, y_train, w_train = (
        X[ind[: (n_0 + n_1)], :],
        y[ind[: (n_0 + n_1)]],
        w[ind[: (n_0 + n_1)]],
    )
    X_test, mu_0_t, mu_1_t, cate_t = (
        X[ind_test, :],
        mu_0[ind_test],
        mu_1[ind_test],
        cate[ind_test],
    )

    return X_train, y_train, w_train, X_test, mu_0_t, mu_1_t, cate_t
