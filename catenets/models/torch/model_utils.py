"""
Author: Alicia Curth
Model utils shared across different nets
"""
from typing import Any, Optional

import torch
from sklearn.model_selection import train_test_split

from catenets.models.constants import DEFAULT_SEED, DEFAULT_VAL_SPLIT

TRAIN_STRING = "training"
VALIDATION_STRING = "validation"


def make_val_split(
    X: torch.Tensor,
    y: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    stratify_w: bool = True,
) -> Any:
    if val_split_prop == 0:
        # return original data
        if w is None:
            return X, y, X, y, TRAIN_STRING

        return X, y, w, X, y, w, TRAIN_STRING

    # make actual split
    if w is None:
        X_t, X_val, y_t, y_val = train_test_split(
            X, y, test_size=val_split_prop, random_state=seed, shuffle=True
        )
        return X_t, y_t, X_val, y_val, VALIDATION_STRING

    if stratify_w:
        # split to stratify by group
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X,
            y,
            w,
            test_size=val_split_prop,
            random_state=seed,
            stratify=w,
            shuffle=True,
        )
    else:
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X, y, w, test_size=val_split_prop, random_state=seed, shuffle=True
        )

    return X_t, y_t, w_t, X_val, y_val, w_val, VALIDATION_STRING


def heads_l2_penalty(
    params_0: torch.Tensor,
    params_1: torch.Tensor,
    n_layers_out: torch.Tensor,
    reg_diff: torch.Tensor,
    penalty_0: torch.Tensor,
    penalty_1: torch.Tensor,
) -> torch.Tensor:
    # Compute l2 penalty for output heads. Either seperately, or regularizing their difference

    # get l2-penalty for first head
    weightsq_0 = penalty_0 * sum(
        [torch.sum(params_0[i][0] ** 2) for i in range(0, 2 * n_layers_out + 1, 2)]
    )

    # get l2-penalty for second head
    if reg_diff:
        weightsq_1 = penalty_1 * sum(
            [
                torch.sum((params_1[i][0] - params_0[i][0]) ** 2)
                for i in range(0, 2 * n_layers_out + 1, 2)
            ]
        )
    else:
        weightsq_1 = penalty_1 * sum(
            [torch.sum(params_1[i][0] ** 2) for i in range(0, 2 * n_layers_out + 1, 2)]
        )
    return weightsq_1 + weightsq_0
