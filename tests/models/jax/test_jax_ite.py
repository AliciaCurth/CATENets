from copy import deepcopy

import numpy as np
import pytest

from catenets.datasets import load
from catenets.experiment_utils.base import get_model_set
from catenets.experiment_utils.tester import evaluate_treatments_model

LAYERS_OUT = 2
LAYERS_R = 3
PENALTY_L2 = 0.01 / 100
PENALTY_ORTHOGONAL_IHDP = 0

MODEL_PARAMS = {
    "n_layers_out": LAYERS_OUT,
    "n_layers_r": LAYERS_R,
    "penalty_l2": PENALTY_L2,
    "penalty_orthogonal": PENALTY_ORTHOGONAL_IHDP,
    "n_layers_out_t": LAYERS_OUT,
    "n_layers_r_t": LAYERS_R,
    "penalty_l2_t": PENALTY_L2,
}

ALL_MODELS = get_model_set(model_selection="all", model_params=MODEL_PARAMS)


def sqrt_PEHE(y: np.ndarray, hat_y: np.ndarray) -> float:
    return np.sqrt(np.mean(((y[:, 1] - y[:, 0]) - hat_y) ** 2))


models = list(ALL_MODELS.keys())
models.remove("PseudoOutcomeNet_PW")


@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 1.5)])
@pytest.mark.parametrize("model_name", models)
def test_model_sanity(dataset: str, pehe_threshold: float, model_name: str) -> None:
    model = deepcopy(ALL_MODELS[model_name])

    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)

    score = evaluate_treatments_model(model, X_train, Y_train, Y_train_full, W_train)
    print(f"Evaluation for model jax.{model_name} on {dataset} = {score['str']}")
    assert score["raw"]["pehe"][0] < pehe_threshold
