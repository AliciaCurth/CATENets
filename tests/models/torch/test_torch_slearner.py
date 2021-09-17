from typing import Any, Optional

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from torch import nn
from xgboost import XGBClassifier

from catenets.datasets import load
from catenets.experiment_utils.tester import evaluate_treatments_model
from catenets.models.torch import SLearner


def test_nn_model_params() -> None:
    model = SLearner(
        2,
        binary_y=True,
        n_layers_out=1,
        n_units_out=2,
        n_units_out_prop=33,
        n_layers_out_prop=12,
        weight_decay=0.5,
        lr=0.6,
        n_iter=700,
        batch_size=80,
        val_split_prop=0.9,
        n_iter_print=10,
        seed=11,
        weighting_strategy="ipw",
    )

    assert model._weighting_strategy == "ipw"
    assert model._propensity_estimator is not None
    assert model._po_estimator is not None

    assert model._po_estimator.n_iter == 700
    assert model._po_estimator.batch_size == 80
    assert model._po_estimator.n_iter_print == 10
    assert model._po_estimator.seed == 11
    assert model._po_estimator.val_split_prop == 0.9
    assert (
        len(model._po_estimator.model) == 5
    )  # 1 in + NL + 3 * (n_layers_hidden -1) + out + Sigmoid

    assert model._propensity_estimator.n_iter == 700
    assert model._propensity_estimator.batch_size == 80
    assert model._propensity_estimator.n_iter_print == 10
    assert model._propensity_estimator.seed == 11
    assert model._propensity_estimator.val_split_prop == 0.9
    assert (
        len(model._propensity_estimator.model) == 38
    )  # 1 in + NL + 3 * (n_layers_hidden - 1) + out + Softmax


@pytest.mark.parametrize("nonlin", ["elu", "relu", "sigmoid"])
def test_nn_model_params_nonlin(nonlin: str) -> None:
    model = SLearner(2, True, nonlin=nonlin, weighting_strategy="ipw")

    nonlins = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
    }

    for mod in [model._propensity_estimator, model._po_estimator]:
        assert isinstance(mod.model[2], nonlins[nonlin])


@pytest.mark.slow
@pytest.mark.parametrize("weighting_strategy", ["ipw", None])
@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
def test_nn_model_sanity(
    dataset: str, pehe_threshold: float, weighting_strategy: Optional[str]
) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = SLearner(
        X_train.shape[1],
        binary_y=(len(np.unique(Y_train)) == 2),
        weighting_strategy=weighting_strategy,
    )

    score = evaluate_treatments_model(model, X_train, Y_train, Y_train_full, W_train)

    print(
        f"Evaluation for model torch.SLearner(NN)(weighting_strategy={weighting_strategy}) on {dataset} = {score['str']}"
    )
    assert score["raw"]["pehe"][0] < pehe_threshold


@pytest.mark.slow
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
            use_label_encoder=False,
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

    model = SLearner(
        X_train.shape[1],
        binary_y=True,
        po_estimator=po_estimator,
    )

    score = evaluate_treatments_model(model, X_train, Y_train, Y_train_full, W_train)

    print(
        f"Evaluation for model torch.SLearner with {po_estimator.__class__} on {dataset} = {score['str']}"
    )
    assert score["raw"]["pehe"][0] < pehe_threshold


@pytest.mark.slow
@pytest.mark.parametrize("exp", [1, 10, 40, 50, 99])
@pytest.mark.parametrize(
    "po_estimator",
    [
        RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
        ),
    ],
)
def test_slearner_sklearn_model_ihdp(po_estimator: Any, exp: int) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(
        "ihdp", exp=exp, rescale=True
    )
    W_train = W_train.ravel()

    model = SLearner(
        X_train.shape[1],
        binary_y=False,
        po_estimator=po_estimator,
    )
    score = evaluate_treatments_model(model, X_train, Y_train, Y_train_full, W_train)

    print(
        f"Evaluation for model torch.SLearner with {po_estimator.__class__} on ihdp[{exp}] = {score['str']}"
    )
    assert score["raw"]["pehe"][0] < 1.5


def test_model_predict_api() -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("ihdp")
    W_train = W_train.ravel()

    model = SLearner(X_train.shape[1], binary_y=False, batch_size=1024, n_iter=100)
    model.fit(X_train, Y_train, W_train)

    out = model.predict(X_test)

    assert len(out) == len(X_test)

    out, p0, p1 = model.predict(X_test, return_po=True)
    assert len(out) == len(X_test)
    assert len(p0) == len(X_test)
    assert len(p1) == len(X_test)

    score = model.score(X_test, Y_test)

    assert score > 0
