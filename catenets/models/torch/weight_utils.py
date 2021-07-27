"""
Author: Alicia Curth
Implement different reweighting/balancing strategies as in Li et al (2018)
"""
from typing import Optional

import numpy as np

IPW_NAME = "ipw"
TRUNC_IPW_NAME = "truncipw"
OVERLAP_NAME = "overlap"
MATCHING_NAME = "match"

ALL_WEIGHTING_STRATEGIES = [IPW_NAME, TRUNC_IPW_NAME, OVERLAP_NAME, MATCHING_NAME]


def compute_importance_weights(
    propensity: np.ndarray,
    w: np.ndarray,
    weighting_strategy: str,
    weight_args: Optional[dict] = None,
) -> np.ndarray:
    if weighting_strategy not in ALL_WEIGHTING_STRATEGIES:
        raise ValueError(
            "weighting_strategy should be in "
            "simbiote.utils.weight_utils.ALL_WEIGHTING_STRATEGIES. "
            "You passed {}".format(weighting_strategy)
        )
    if weight_args is None:
        weight_args = {}

    if weighting_strategy == IPW_NAME:
        return compute_ipw(propensity, w)
    elif weighting_strategy == TRUNC_IPW_NAME:
        return compute_trunc_ipw(propensity, w, **weight_args)
    elif weighting_strategy == OVERLAP_NAME:
        return compute_overlap_weights(propensity, w)
    elif weighting_strategy == MATCHING_NAME:
        return compute_matching_weights(propensity, w)


def compute_ipw(propensity: np.ndarray, w: np.ndarray) -> np.ndarray:
    p_hat = np.average(w)
    return w * p_hat / propensity + (1 - w) * (1 - p_hat) / (1 - propensity)


def compute_trunc_ipw(
    propensity: np.ndarray, w: np.ndarray, cutoff: float = 0.05
) -> np.ndarray:
    ipw = compute_ipw(propensity, w)
    return np.where((propensity > cutoff) & (propensity < 1 - cutoff), ipw, 0)


# TODO check normalizing these weights
def compute_matching_weights(propensity: np.ndarray, w: np.ndarray) -> np.ndarray:
    ipw = compute_ipw(propensity, w)
    return np.minimum(ipw, 1 - ipw) * ipw


def compute_overlap_weights(propensity: np.ndarray, w: np.ndarray) -> np.ndarray:
    ipw = compute_ipw(propensity, w)
    return propensity * (1 - propensity) * ipw
