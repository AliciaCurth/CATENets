from copy import deepcopy

import pytest

from catenets.datasets import load
from catenets.experiment_utils.tester import evaluate_treatments_model
from catenets.models.jax import FLEXTE_NAME, OFFSET_NAME, FlexTENet, OffsetNet

LAYERS_OUT = 2
LAYERS_R = 3
PENALTY_L2 = 0.01 / 100
PENALTY_ORTHOGONAL_IHDP = 0

PARAMS_DEPTH: dict = {"n_layers_r": 2, "n_layers_out": 2, "n_iter": 10}
PENALTY_DIFF = 0.01
PENALTY_ORTHOGONAL = 0.1

ALL_MODELS = {
    OFFSET_NAME: OffsetNet(penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH),
    FLEXTE_NAME: FlexTENet(
        penalty_orthogonal=PENALTY_ORTHOGONAL, penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH
    ),
}

models = list(ALL_MODELS.keys())


@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 3)])
@pytest.mark.parametrize("model_name", models)
def test_model_sanity(dataset: str, pehe_threshold: float, model_name: str) -> None:
    model = deepcopy(ALL_MODELS[model_name])

    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)

    score = evaluate_treatments_model(model, X_train, Y_train, Y_train_full, W_train)
    print(f"Evaluation for model jax.{model_name} on {dataset} = {score['str']}")


def test_model_score() -> None:
    model = OffsetNet(n_iter=10)

    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("ihdp")

    model.fit(X_train[:10], Y_train[:10], W_train[:10])

    result = model.score(X_test, Y_test)

    assert result > 0

    with pytest.raises(ValueError):
        model.score(X_train, Y_train)  # Y_train has just one outcome
