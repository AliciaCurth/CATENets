import pytest
from torch import nn

from catenets.datasets import load
from catenets.experiments.torch.metrics import sqrt_PEHE
from catenets.models.torch import TARNet


def test_model_params() -> None:
    model = TARNet(
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

    assert model._representation_block is not None
    assert len(model._hypothesis_block) == 2

    for mod in model._hypothesis_block:
        assert mod.n_iter == 700
        assert mod.batch_size == 80
        assert mod.n_iter_print == 10
        assert mod.seed == 11
        assert mod.val_split_prop == 0.9
        assert (
            len(mod.model) == 10
        )  # 1 in + NL + 2 * n_layers_r + 2 * n_layers_out + 1 out

    assert len(model._representation_block.model) == 6


@pytest.mark.parametrize("nonlin", ["elu", "relu", "sigmoid"])
def test_model_params_nonlin(nonlin: str) -> None:
    model = TARNet(2, nonlin=nonlin)

    nonlins = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
    }

    for mod in [
        model._representation_block,
        model._hypothesis_block[0],
        model._hypothesis_block[1],
    ]:
        assert isinstance(mod.model[1], nonlins[nonlin])  # type: ignore


@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
def test_model_sanity(dataset: str, pehe_threshold: float) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()

    model = TARNet(X_train.shape[1], n_iter=300)

    model.train(X=X_train, y=Y_train, w=W_train)

    cate_pred = model(X_test).detach().numpy()

    pehe = sqrt_PEHE(Y_test, cate_pred)

    print(f"PEHE score for model torch.TARNet on {dataset} = {pehe}")
    assert pehe < pehe_threshold
