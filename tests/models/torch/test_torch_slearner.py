from typing import Optional

import pytest
from torch import nn

from catenets.datasets import load
from catenets.experiments.torch.metrics import sqrt_PEHE
from catenets.models.torch import SLearner


def test_model_params() -> None:
    model = SLearner(
        2,
        binary_y=True,
        n_layers_out=1,
        n_units_out=2,
        n_layers_r=3,
        n_units_r=4,
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

    assert model.weighting_strategy == "ipw"
    assert model._propensity_estimator is not None
    assert model._output_estimator is not None

    assert model._output_estimator.n_iter == 700
    assert model._output_estimator.batch_size == 80
    assert model._output_estimator.n_iter_print == 10
    assert model._output_estimator.seed == 11
    assert model._output_estimator.val_split_prop == 0.9
    assert (
        len(model._output_estimator.model) == 10
    )  # 1 in + NL + 2 * n_layers_r + 2 * n_layers_out + 1 out

    assert model._propensity_estimator.n_iter == 700
    assert model._propensity_estimator.batch_size == 80
    assert model._propensity_estimator.n_iter_print == 10
    assert model._propensity_estimator.seed == 11
    assert model._propensity_estimator.val_split_prop == 0.9
    assert (
        len(model._propensity_estimator.model) == 32
    )  # 1 in + NL + 2 * n_layers_r + 2 * n_layers_out + 1 out


@pytest.mark.parametrize("nonlin", ["elu", "relu", "sigmoid"])
def test_model_params_nonlin(nonlin: str) -> None:
    model = SLearner(2, nonlin=nonlin)

    nonlins = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
    }

    for mod in [model._propensity_estimator, model._output_estimator]:
        assert isinstance(mod.model[1], nonlins[nonlin])


@pytest.mark.parametrize("weighting_strategy", ["ipw", None])
@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
def test_model_sanity(
    dataset: str, pehe_threshold: float, weighting_strategy: Optional[str]
) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = SLearner(
        X_train.shape[1], n_iter=250, weighting_strategy=weighting_strategy
    )

    model.train(X=X_train, y=Y_train, w=W_train)

    cate_pred = model(X_test).detach().numpy()

    pehe = sqrt_PEHE(Y_test, cate_pred)

    print(
        f"PEHE score for model torch.SLearner(weighting_strategy={weighting_strategy}) on {dataset} = {pehe}"
    )
    assert pehe < pehe_threshold
