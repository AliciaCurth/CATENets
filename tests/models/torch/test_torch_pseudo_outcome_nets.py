from typing import Any

import pytest
from torch import nn

from catenets.datasets import load
from catenets.experiments.torch.metrics import sqrt_PEHE
from catenets.models.torch import RANet, XNet


@pytest.mark.parametrize("model_t", [RANet, XNet])
def test_model_params(model_t: Any) -> None:
    model = model_t(
        2,
    )

    assert model._te_estimator is not None
    assert model._po_estimator is not None
    assert model._propensity_estimator is not None


@pytest.mark.parametrize("nonlin", ["elu", "relu", "sigmoid"])
@pytest.mark.parametrize("model_t", [RANet, XNet])
def test_model_params_nonlin(nonlin: str, model_t: Any) -> None:
    model = model_t(2, nonlin=nonlin)

    nonlins = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
    }

    for mod in [model._te_estimator, model._po_estimator, model._propensity_estimator]:
        assert isinstance(mod.model[1], nonlins[nonlin])


@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
@pytest.mark.parametrize("model_t", [RANet, XNet])
def test_model_sanity(dataset: str, pehe_threshold: float, model_t: Any) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = model_t(X_train.shape[1], n_iter=500)

    model.train(X=X_train, y=Y_train, w=W_train)

    cate_pred = model(X_test).detach().numpy()

    pehe = sqrt_PEHE(Y_test, cate_pred)

    print(f"PEHE score for model torch.{model_t} on {dataset} = {pehe}")
    assert pehe < pehe_threshold
