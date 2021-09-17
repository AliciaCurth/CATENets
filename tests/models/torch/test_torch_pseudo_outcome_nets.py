from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from torch import nn
from xgboost import XGBClassifier

from catenets.datasets import load
from catenets.experiment_utils.tester import evaluate_treatments_model
from catenets.models.torch import (
    DRLearner,
    PWLearner,
    RALearner,
    RLearner,
    ULearner,
    XLearner,
)


@pytest.mark.parametrize(
    "model_t", [DRLearner, PWLearner, RALearner, RLearner, ULearner, XLearner]
)
def test_nn_model_params(model_t: Any) -> None:
    model = model_t(
        2,
        binary_y=True,
    )

    assert model._te_estimator is not None
    assert model._po_estimator is not None
    assert model._propensity_estimator is not None


@pytest.mark.parametrize("nonlin", ["elu", "relu", "sigmoid"])
@pytest.mark.parametrize(
    "model_t", [DRLearner, PWLearner, RALearner, RLearner, ULearner, XLearner]
)
def test_nn_model_params_nonlin(nonlin: str, model_t: Any) -> None:
    model = model_t(2, binary_y=True, nonlin=nonlin)

    nonlins = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
    }

    for mod in [model._te_estimator, model._po_estimator, model._propensity_estimator]:
        assert isinstance(mod.model[2], nonlins[nonlin])


@pytest.mark.slow
@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 4)])
@pytest.mark.parametrize("model_t", [DRLearner, RALearner, XLearner])
def test_nn_model_sanity(dataset: str, pehe_threshold: float, model_t: Any) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = model_t(X_train.shape[1], binary_y=(len(np.unique(Y_train)) == 2))

    score = evaluate_treatments_model(model, X_train, Y_train, Y_train_full, W_train)

    print(
        f"Evaluation for model torch.{model_t} with NNs on {dataset} = {score['str']}"
    )


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
    ],
)
@pytest.mark.parametrize(
    "te_estimator",
    [
        RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
        ),
    ],
)
@pytest.mark.parametrize("model_t", [DRLearner, RALearner])
def test_sklearn_model_pseudo_outcome_binary(
    dataset: str,
    pehe_threshold: float,
    po_estimator: Any,
    te_estimator: Any,
    model_t: Any,
) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = model_t(
        X_train.shape[1],
        binary_y=True,
        po_estimator=po_estimator,
        te_estimator=te_estimator,
        batch_size=1024,
    )

    score = evaluate_treatments_model(
        model, X_train, Y_train, Y_train_full, W_train, n_folds=3
    )

    print(
        f"Evaluation for model {model_t} with po_estimator = {type(po_estimator)},"
        f"te_estimator = {type(te_estimator)} on {dataset} = {score['str']}"
    )


def test_model_predict_api() -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("ihdp")
    W_train = W_train.ravel()

    model = XLearner(X_train.shape[1], binary_y=False, batch_size=1024, n_iter=100)
    model.fit(X_train, Y_train, W_train)

    out = model.predict(X_test)

    assert len(out) == len(X_test)

    score = model.score(X_test, Y_test)

    assert score > 0
