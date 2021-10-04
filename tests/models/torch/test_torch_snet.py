import numpy as np
import pytest
from torch import nn

from catenets.datasets import load
from catenets.experiment_utils.tester import evaluate_treatments_model
from catenets.models.torch import SNet


def test_model_params() -> None:
    # with propensity estimator
    model = SNet(
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

    assert model._reps_c is not None
    assert model._reps_o is not None
    assert model._reps_mu0 is not None
    assert model._reps_mu1 is not None
    assert model._reps_prop is not None
    assert model._propensity_estimator is not None
    assert len(model._po_estimators) == 2

    for mod in model._po_estimators:
        assert len(mod.model) == 5  # 1 in + NL + 4 * (n_layers_out - 1) + 1 out + NL

    assert len(model._reps_c.model) == 9
    assert len(model._reps_o.model) == 9
    assert len(model._reps_mu0.model) == 9
    assert len(model._reps_mu1.model) == 9
    assert len(model._propensity_estimator.model) == 8

    # remove propensity estimator
    model = SNet(
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
        with_prop=False
    )

    with np.testing.assert_raises(AttributeError):
        model._reps_c
    with np.testing.assert_raises(AttributeError):
        model._reps_prop
    with np.testing.assert_raises(AttributeError):
        model._propensity_estimator
    assert model._reps_o is not None
    assert model._reps_mu0 is not None
    assert model._reps_mu1 is not None
    assert len(model._po_estimators) == 2

    for mod in model._po_estimators:
        assert len(mod.model) == 5  # 1 in + NL + 4 * (n_layers_out - 1) + 1 out + NL

    assert len(model._reps_o.model) == 9
    assert len(model._reps_mu0.model) == 9
    assert len(model._reps_mu1.model) == 9


@pytest.mark.parametrize("nonlin", ["elu", "relu", "sigmoid", "selu", "leaky_relu"])
def test_model_params_nonlin(nonlin: str) -> None:
    model = SNet(2, nonlin=nonlin)

    nonlins = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "selu": nn.SELU,
        "leaky_relu": nn.LeakyReLU,
    }

    for mod in [
        model._reps_c,
        model._reps_o,
        model._reps_mu0,
        model._reps_mu1,
        model._reps_prop,
        model._po_estimators[0],
        model._po_estimators[1],
        model._propensity_estimator,
    ]:
        assert isinstance(mod.model[2], nonlins[nonlin])


@pytest.mark.slow
@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
def test_model_sanity(dataset: str, pehe_threshold: float) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    # with propensity estimator
    model = SNet(
        X_train.shape[1],
        binary_y=(len(np.unique(Y_train)) == 2),
        batch_size=1024,
        n_iter=1500,
    )

    score = evaluate_treatments_model(
        model, X_train, Y_train, Y_train_full, W_train, n_folds=3
    )

    print(f"Evaluation for model SNet on {dataset} = {score['str']}")
    assert score["raw"]["pehe"][0] < pehe_threshold

    model = SNet(
        X_train.shape[1],
        binary_y=(len(np.unique(Y_train)) == 2),
        batch_size=1024,
        n_iter=1500,
        with_prop=False
    )

    score = evaluate_treatments_model(
        model, X_train, Y_train, Y_train_full, W_train, n_folds=3
    )

    print(f"Evaluation for model SNet (with_prop=False) on {dataset} = {score['str']}")
    assert score["raw"]["pehe"][0] < pehe_threshold


def test_model_predict_api() -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("ihdp")
    W_train = W_train.ravel()

    model = SNet(X_train.shape[1], batch_size=1024, n_iter=100)
    model.fit(X_train, Y_train, W_train)

    out = model.predict(X_test)

    assert len(out) == len(X_test)

    out, p0, p1 = model.predict(X_test, return_po=True)
    assert len(out) == len(X_test)
    assert len(p0) == len(X_test)
    assert len(p1) == len(X_test)

    score = model.score(X_test, Y_test)

    assert score > 0
