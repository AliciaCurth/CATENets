"""
Author: Alicia Curth
Some utils for experiments
"""
import jax.numpy as jnp

from catenets.models.base import check_shape_1d_data


def eval_mse_model(inputs, targets, predict_fun, params):
    # evaluate the mse of a model given its function and params
    preds = predict_fun(params, inputs)
    return jnp.mean((preds - targets) ** 2)


def eval_mse(preds, targets):
    preds = check_shape_1d_data(preds)
    targets = check_shape_1d_data(targets)
    return jnp.mean((preds - targets) ** 2)


def eval_root_mse(cate_pred, cate_true):
    cate_true = check_shape_1d_data(cate_true)
    cate_pred = check_shape_1d_data(cate_pred)
    return jnp.sqrt(eval_mse(cate_pred, cate_true))