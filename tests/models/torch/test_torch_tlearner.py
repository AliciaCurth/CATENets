from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from torch import nn
from xgboost import XGBClassifier, XGBRegressor

from catenets.datasets import load
from catenets.experiments.torch.metrics import sqrt_PEHE
from catenets.models.torch import TLearner


def test_nn_model_params() -> None:
    model = TLearner(
        2,
        True,
        n_layers_out=1,
        n_units_out=2,
        weight_decay=0.5,
        lr=0.6,
        n_iter=700,
        batch_size=80,
        val_split_prop=0.9,
        n_iter_print=10,
        seed=11,
    )

    assert len(model._plug_in) == 2

    for mod in model._plug_in:
        assert mod.n_iter == 700
        assert mod.batch_size == 80
        assert mod.n_iter_print == 10
        assert mod.seed == 11
        assert mod.val_split_prop == 0.9
        assert len(mod.model) == 6  # 2 in + NL + 2 * n_layers_hidden + 2 out


@pytest.mark.parametrize("nonlin", ["elu", "relu", "sigmoid"])
def test_nn_model_params_nonlin(nonlin: str) -> None:
    model = TLearner(2, True, nonlin=nonlin)

    assert len(model._plug_in) == 2

    nonlins = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
    }

    for mod in model._plug_in:
        assert isinstance(mod.model[1], nonlins[nonlin])


@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
def test_nn_model_sanity(dataset: str, pehe_threshold: float) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = TLearner(
        X_train.shape[1], binary_y=(len(np.unique(Y_train)) == 2), n_iter=500
    )

    model.train(X=X_train, y=Y_train, w=W_train)

    cate_pred = model(X_test).detach().numpy()

    pehe = sqrt_PEHE(Y_test, cate_pred)

    print(f"PEHE score for model torch.TLearner(NN) on {dataset} = {pehe}")
    assert pehe < pehe_threshold


@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4)])
@pytest.mark.parametrize(
    "po_estimator",
    [
        XGBClassifier(
            n_estimators=100,
            reg_lambda=1e-3,
            reg_alpha=1e-3,
            colsample_bytree=0.1,
            colsample_bynode=0.1,
            colsample_bylevel=0.1,
            max_depth=6,
            tree_method="hist",
            learning_rate=1e-2,
            min_child_weight=0,
            max_bin=256,
            random_state=0,
            eval_metric="logloss",
        ),
        RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
        ),
        LogisticRegression(
            C=1.0,
            solver="sag",
            max_iter=10000,
            penalty="l2",
        ),
    ],
)
def test_sklearn_model_sanity_binary_output(
    dataset: str, pehe_threshold: float, po_estimator: Any
) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = TLearner(
        X_train.shape[1],
        binary_y=True,
        po_estimator=po_estimator,
    )

    model.train(X=X_train, y=Y_train, w=W_train)

    cate_pred = model(X_test).detach().numpy()

    pehe = sqrt_PEHE(Y_test, cate_pred)

    print(
        f"PEHE score for model torch.TLearner {type(po_estimator)} on {dataset} = {pehe}"
    )
    assert pehe < pehe_threshold


@pytest.mark.parametrize("dataset, pehe_threshold", [("ihdp", 1.5)])
@pytest.mark.parametrize(
    "po_estimator",
    [
        XGBRegressor(
            n_estimators=500,
            reg_lambda=1e-3,
            reg_alpha=1e-3,
            colsample_bytree=0.1,
            colsample_bynode=0.1,
            colsample_bylevel=0.1,
            max_depth=6,
            tree_method="hist",
            learning_rate=1e-2,
            min_child_weight=0,
            max_bin=256,
            random_state=0,
            eval_metric="logloss",
        ),
        RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
        ),
    ],
)
def test_sklearn_model_sanity_regression(
    dataset: str, pehe_threshold: float, po_estimator: Any
) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = TLearner(
        X_train.shape[1],
        binary_y=False,
        po_estimator=po_estimator,
    )

    model.train(X=X_train, y=Y_train, w=W_train)

    cate_pred = model(X_test).detach().numpy()

    pehe = sqrt_PEHE(Y_test, cate_pred)

    print(
        f"PEHE score for model torch.TLearner {type(po_estimator)} on {dataset} = {pehe}"
    )
    assert pehe < pehe_threshold
