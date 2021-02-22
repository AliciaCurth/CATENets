"""
Author: Alicia Curth
Some utils for experiments
"""
import jax.numpy as jnp

from catenets.models.base import check_shape_1d_data

from catenets.models import T_NAME, SNET1_NAME, SNET2_NAME, SNET3_NAME, \
    SNET_NAME, TWOSTEP_NAME, TNet, SNet1, SNet2, SNet3, SNet, TwoStepNet, get_catenet
from catenets.models.transformation_utils import AIPW_TRANSFORMATION, HT_TRANSFORMATION, \
    RA_TRANSFORMATION

SEP = "_"


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


def get_model_set(model_selection = 'all', model_params: dict = None):
    """ Helper function to retrieve a set of models """
    # get model selection
    if type(model_selection) is str:
        if model_selection == 'plug':
            models = ALL_PLUGIN_MODELS
        elif model_selection == 'twostep':
            models = ALL_TWOSTEP_MODELS
        elif model_selection == 'all':
            models = dict(**ALL_PLUGIN_MODELS, **ALL_TWOSTEP_MODELS)
        else:
            models = {model_selection: get_catenet(model_selection)()}
    elif type(model_selection) is list:
        models = {}
        for model in model_selection:
            models.update({model: get_catenet(model)()})
    else:
        raise ValueError("model_selection should be string or list.")

    # set hyperparameters
    if model_params is not None:
        for model in models.values():
            existing_params = model.get_params()
            new_params = {key: val for key, val in model_params.items() if
                          key in existing_params.keys()}
            model.set_params(**new_params)

    return models


ALL_PLUGIN_MODELS = {T_NAME: TNet(),  SNET1_NAME: SNet1(), SNET2_NAME: SNet2(),
                     SNET3_NAME: SNet3(), SNET_NAME: SNet()}

ALL_TWOSTEP_MODELS = {TWOSTEP_NAME + SEP + AIPW_TRANSFORMATION:
    TwoStepNet(transformation=AIPW_TRANSFORMATION),
    TWOSTEP_NAME + SEP + HT_TRANSFORMATION: TwoStepNet(transformation=HT_TRANSFORMATION),
    TWOSTEP_NAME + SEP + RA_TRANSFORMATION: TwoStepNet(transformation=RA_TRANSFORMATION)
}
