from typing import Type

import pytest
from torch import nn

from catenets.datasets import load
from catenets.experiment_utils.tester import evaluate_treatments_model
from catenets.models.torch import DragonNet, TARNet


@pytest.mark.parametrize("snet", [TARNet, DragonNet])
def test_model_params(snet: Type) -> None:
    model = snet(
        2,
        binary_y=True,
        n_layers_out=1,
        n_units_out=2,
        n_layers_r=3,
        n_units_r=4,
        weight_decay=0.5,
        lr=0.6,
        n_iter=700,
        batch_size=80,
        val_split_prop=0.9,
        n_iter_print=10,
        seed=11,
    )

    assert model._repr_estimator is not None
    assert model._propensity_estimator is not None
    assert len(model._po_estimators) == 2

    for mod in model._po_estimators:
        assert len(mod.model) == 5  # 1 in + NL + 4 * (n_layers_out - 1) + 1 out + NL

    assert len(model._repr_estimator.model) == 9


@pytest.mark.parametrize("nonlin", ["elu", "relu", "sigmoid"])
@pytest.mark.parametrize("snet", [TARNet, DragonNet])
def test_model_params_nonlin(nonlin: str, snet: Type) -> None:
    model = snet(2, nonlin=nonlin)

    nonlins = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
    }

    for mod in [
        model._repr_estimator,
        model._po_estimators[0],
        model._po_estimators[1],
        model._propensity_estimator,
    ]:
        assert isinstance(mod.model[2], nonlins[nonlin])


@pytest.mark.slow
@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
@pytest.mark.parametrize("snet", [TARNet, DragonNet])
def test_model_sanity(dataset: str, pehe_threshold: float, snet: Type) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = snet(X_train.shape[1], batch_size=1024)

    score = evaluate_treatments_model(
        model, X_train, Y_train, Y_train_full, W_train, n_folds=3
    )

    print(f"Evaluation for model {snet} on {dataset} = {score['str']}")
    assert score["raw"]["pehe"][0] < pehe_threshold


def test_model_predict_api() -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("ihdp")
    W_train = W_train.ravel()

    model = TARNet(X_train.shape[1], batch_size=1024, n_iter=100)
    model.fit(X_train, Y_train, W_train)

    out = model.predict(X_test)

    assert len(out) == len(X_test)

    out, p0, p1 = model.predict(X_test, return_po=True)
    assert len(out) == len(X_test)
    assert len(p0) == len(X_test)
    assert len(p1) == len(X_test)

    score = model.score(X_test, Y_test)

    assert score > 0
