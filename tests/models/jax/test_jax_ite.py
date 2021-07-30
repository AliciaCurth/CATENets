from copy import deepcopy

import pytest

from catenets.datasets import load
from catenets.experiment_utils.tester import evaluate_treatments_model
from catenets.models.jax import (
    DRNET_NAME,
    FLEXTE_NAME,
    OFFSET_NAME,
    T_NAME,
    TARNET_NAME,
    DRNet,
    FlexTENet,
    OffsetNet,
    TARNet,
    TNet,
)

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
PARAMS_DEPTH: dict = {"n_layers_r": 2, "n_layers_out": 2}
PARAMS_DEPTH_2: dict = {
    "n_layers_r": 2,
    "n_layers_out": 2,
    "n_layers_r_t": 2,
    "n_layers_out_t": 2,
}
PENALTY_DIFF = 0.01
PENALTY_ORTHOGONAL = 0.1

ALL_MODELS = {
    T_NAME: TNet(**PARAMS_DEPTH),
    T_NAME
    + "_reg": TNet(train_separate=False, penalty_diff=PENALTY_DIFF, **PARAMS_DEPTH),
    TARNET_NAME: TARNet(**PARAMS_DEPTH),
    TARNET_NAME
    + "_reg": TARNet(
        reg_diff=True, penalty_diff=PENALTY_DIFF, same_init=True, **PARAMS_DEPTH
    ),
    OFFSET_NAME: OffsetNet(penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH),
    FLEXTE_NAME: FlexTENet(
        penalty_orthogonal=PENALTY_ORTHOGONAL, penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH
    ),
    FLEXTE_NAME + "_noortho_reg_same": FlexTENet(penalty_orthogonal=0, **PARAMS_DEPTH),
    DRNET_NAME: DRNet(**PARAMS_DEPTH_2),
    DRNET_NAME + "_TAR": DRNet(first_stage_strategy="Tar", **PARAMS_DEPTH_2),
}

models = list(ALL_MODELS.keys())


@pytest.mark.parametrize("dataset, pehe_threshold", [("twins", 0.4), ("ihdp", 3)])
@pytest.mark.parametrize("model_name", models)
def test_model_sanity(dataset: str, pehe_threshold: float, model_name: str) -> None:
    model = deepcopy(ALL_MODELS[model_name])

    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)

    score = evaluate_treatments_model(model, X_train, Y_train, Y_train_full, W_train)
    print(f"Evaluation for model jax.{model_name} on {dataset} = {score['str']}")
    assert score["raw"]["pehe"][0] < pehe_threshold
