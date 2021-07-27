import sys

import numpy as np
import pytest
from torch import nn

import catenets.logger as log
from catenets.datasets import load
from catenets.models.torch import TNet

log.add(sink=sys.stderr, level="DEBUG")


def sqrt_PEHE(y: np.ndarray, hat_y: np.ndarray) -> float:
    return np.sqrt(np.mean(((y[:, 1] - y[:, 0]) - hat_y) ** 2))


def test_model_params() -> None:
    model = TNet(
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

    assert model._plug_in_0 is not None
    assert model._plug_in_1 is not None

    for mod in [model._plug_in_0, model._plug_in_1]:
        assert mod.n_iter == 700
        assert mod.batch_size == 80
        assert mod.n_iter_print == 10
        assert mod.seed == 11
        assert mod.val_split_prop == 0.9
        assert (
            len(mod.model) == 10
        )  # 1 in + NL + 2 * n_layers_r + 2 * n_layers_out + 1 out


@pytest.mark.parametrize("nonlin", ["elu", "relu", "sigmoid"])
def test_model_params_nonlin(nonlin: str) -> None:
    model = TNet(2, nonlin=nonlin)

    assert model._plug_in_0 is not None
    assert model._plug_in_1 is not None

    nonlins = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
    }

    for mod in [model._plug_in_0, model._plug_in_1]:
        assert isinstance(mod.model[1], nonlins[nonlin])


@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
def test_model_sanity(dataset: str, pehe_threshold: float) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = TNet(X_train.shape[1], n_iter=250)

    model.train(X=X_train, y=Y_train, w=W_train)

    cate_pred = model(X_test).detach().numpy()

    pehe = sqrt_PEHE(Y_test, cate_pred)

    print(f"PEHE score for model torch.TNet on {dataset} = {pehe}")
    assert pehe < pehe_threshold
