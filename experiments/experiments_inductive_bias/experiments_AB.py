"""
Utils to replicate setups A & B
"""
# Author: Alicia Curth
import csv
import os
from typing import Optional, Tuple, Union

import numpy as onp
from sklearn import clone

from catenets.datasets import load
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

RESULT_DIR_SIMU = "results/experiments_inductive_bias/acic2016/simulations/"
SEP = "_"

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

# For results in Appendix B.3
FLEX_LAMBDA = {'FlexTENet_001': FlexTENet(penalty_orthogonal=PENALTY_ORTHOGONAL,
                                          penalty_l2_p=1 / 100,
                                          **PARAMS_DEPTH),
               'FlexTENet_01': FlexTENet(penalty_orthogonal=PENALTY_ORTHOGONAL,
                                         penalty_l2_p=1 / 10,
                                         **PARAMS_DEPTH),
               'FlexTENet_0001': FlexTENet(penalty_orthogonal=PENALTY_ORTHOGONAL,
                                           penalty_l2_p=1 / 1000,
                                           **PARAMS_DEPTH),
               'FlexTENet_00001': FlexTENet(penalty_orthogonal=PENALTY_ORTHOGONAL,
                                            penalty_l2_p=1 / 10000,
                                            **PARAMS_DEPTH)
               }

T_LAMBDA = {T_NAME: TNet(**PARAMS_DEPTH),
            T_NAME + '_reg_01': TNet(train_separate=False, penalty_diff=1 / 10, **PARAMS_DEPTH),
            T_NAME + '_reg_001': TNet(train_separate=False, penalty_diff=1 / 100, **PARAMS_DEPTH),
            T_NAME + '_reg_0001': TNet(train_separate=False, penalty_diff=1 / 1000,
                                       **PARAMS_DEPTH),
            T_NAME + '_reg_00001': TNet(train_separate=False, penalty_diff=1 / 10000,
                                        **PARAMS_DEPTH)}

OFFSET_LAMBDA = {OFFSET_NAME + '_reg_01': OffsetNet(penalty_l2_p=1 / 10, **PARAMS_DEPTH),
                 OFFSET_NAME + '_reg_001': OffsetNet(penalty_l2_p=1 / 100, **PARAMS_DEPTH),
                 OFFSET_NAME + '_reg_0001': OffsetNet(penalty_l2_p=1 / 1000, **PARAMS_DEPTH),
                 OFFSET_NAME + '_reg_00001': OffsetNet(penalty_l2_p=1 / 10000, **PARAMS_DEPTH),
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
        factual_eval: bool = False
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
                    factual_eval=factual_eval,
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
                    factual_eval=factual_eval
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
        factual_eval: bool = False,
) -> None:
    if models is None:
        models = ALL_MODELS
    elif isinstance(models, str):
        if models == "all":
            models = ALL_MODELS
        elif models == "ablations":
            models = ABLATIONS
        elif models == 'flex_lambda':
            models = FLEX_LAMBDA
        elif models == 't_lambda':
            models = T_LAMBDA
        elif models == 'offset_lambda':
            models = OFFSET_LAMBDA
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
                if "R" not in name and "X" not in name
            ]
            + [
                name + "_mu1"
                for name in models.keys()
                if "R" not in name and "X" not in name
            ]
    )

    if factual_eval:
        header = header + [
            name + '_factual' for name in models.keys()
            if 'R' not in name and 'X' not in name
        ]

    writer.writerow(header)

    if isinstance(n_exp, int):
        experiment_loop = list(range(1, n_exp + 1))
    elif isinstance(n_exp, list):
        experiment_loop = n_exp
    else:
        raise ValueError("n_exp should be either an integer or a list of integers.")

    for i_exp in experiment_loop:
        rmse_cate = []
        rmse_mu0 = []
        rmse_mu1 = []

        # get data
        if not factual_eval:
            X, y, w, X_t, mu_0_t, mu_1_t, cate_t = acic_simu(
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
        else:
            rmse_factual = []
            X, y, w, X_t, y_t, w_t, mu_0_t, mu_1_t, cate_t = acic_simu(
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
                return_ytest=True,
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
                    "R" not in model_name
                    and "X" not in model_name
            ):
                cate_pred_out, mu0_pred, mu1_pred = estimator_temp.predict(
                    X_t, return_po=True
                )
                rmse_mu0.append(eval_root_mse(mu0_pred, mu_0_t))
                rmse_mu1.append(eval_root_mse(mu1_pred, mu_1_t))
                if factual_eval:
                    pred_factual = w_t * mu1_pred + (1 - w_t) * mu0_pred
                    rmse_factual.append(eval_root_mse(pred_factual, y_t))
            else:
                cate_pred_out = estimator_temp.predict(X_t)

            rmse_cate.append(eval_root_mse(cate_pred_out, cate_t))

        if not factual_eval:
            writer.writerow([y_var, cate_var] + rmse_cate + rmse_mu0 + rmse_mu1)
        else:
            writer.writerow([y_var, cate_var] + rmse_cate + rmse_mu0 + rmse_mu1 + rmse_factual)

    out_file.close()


def acic_simu(
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
        return_ytest: bool = False,
) -> Tuple:
    X_train, w_train, y_train, _, X_test, w_test, y_test, po_test = load(
        "acic2016",
        i_exp=i_exp,
        n_0=n_0,
        n_1=n_1,
        n_test=n_test,
        error_sd=error_sd,
        sp_lin=sp_lin,
        sp_nonlin=sp_nonlin,
        prop_gamma=prop_gamma,
        prop_omega=prop_omega,
        ate_goal=ate_goal,
        inter=inter,
    )
    mu_0_t = po_test[:, 0]
    mu_1_t = po_test[:, 1]
    cate_t = mu_1_t - mu_0_t

    if return_ytest:
        return X_train, y_train, w_train, X_test, y_test, w_test, mu_0_t, mu_1_t, cate_t

    return X_train, y_train, w_train, X_test, mu_0_t, mu_1_t, cate_t
