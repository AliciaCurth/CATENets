import numpy as np
import pytest

from catenets.datasets import load
from catenets.experiment_utils.tester import evaluate_treatments_model
from catenets.models.torch import FlexTENet


def test_flextenet_model_params() -> None:
    model = FlexTENet(
        2,
        binary_y=True,
        n_layers_out=1,
        n_layers_r=2,
        n_units_s_out=20,
        n_units_p_out=30,
        n_units_s_r=40,
        n_units_p_r=50,
        private_out=True,
        weight_decay=1e-5,
        penalty_orthogonal=1e-7,
        lr=1e-2,
        n_iter=123,
        batch_size=234,
        early_stopping=True,
        patience=5,
        n_iter_min=13,
        n_iter_print=7,
        seed=42,
        shared_repr=False,
        normalize_ortho=False,
        mode=1,
    )

    assert model.binary_y is True
    assert model.n_layers_out == 1
    assert model.n_layers_r == 2
    assert model.n_units_s_out == 20
    assert model.n_units_p_out == 30
    assert model.n_units_s_r == 40
    assert model.n_units_p_r == 50
    assert model.private_out is True
    assert model.weight_decay == 1e-5
    assert model.penalty_orthogonal == 1e-7
    assert model.lr == 1e-2
    assert model.n_iter == 123
    assert model.batch_size == 234
    assert model.early_stopping is True
    assert model.patience == 5
    assert model.n_iter_min == 13
    assert model.n_iter_print == 7
    assert model.seed == 42
    assert model.shared_repr is False
    assert model.normalize_ortho is False
    assert model.mode == 1


@pytest.mark.slow
@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
def test_flextenet_model_sanity(dataset: str, pehe_threshold: float) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = FlexTENet(
        X_train.shape[1],
        binary_y=(len(np.unique(Y_train)) == 2),
        batch_size=1024,
        lr=1e-3,
    )

    score = evaluate_treatments_model(
        model, X_train, Y_train, Y_train_full, W_train, n_folds=2
    )

    print(f"Evaluation for model FlexTENet on {dataset} = {score['str']}")
    assert score["raw"]["pehe"][0] < pehe_threshold


@pytest.mark.parametrize("shared_repr", [False, True])
@pytest.mark.parametrize("private_out", [False, True])
@pytest.mark.parametrize("n_units_p_r", [50, 150])
def test_flextenet_model_predict_api(
    shared_repr: bool, private_out: bool, n_units_p_r: int
) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("ihdp")
    W_train = W_train.ravel()

    model = FlexTENet(
        X_train.shape[1],
        binary_y=(len(np.unique(Y_train)) == 2),
        batch_size=1024,
        n_iter=100,
        lr=1e-3,
        shared_repr=shared_repr,
        private_out=private_out,
        n_units_p_r=n_units_p_r,
    )
    model.fit(X_train, Y_train, W_train)

    out = model.predict(X_test)

    assert len(out) == len(X_test)

    out, p0, p1 = model.predict(X_test, return_po=True)
    assert len(out) == len(X_test)
    assert len(p0) == len(X_test)
    assert len(p1) == len(X_test)

    score = model.score(X_test, Y_test)

    assert score > 0
