"""
Author: Alicia Curth
Script to generate synthetic simulations in AISTATS paper
"""
import os
import csv
from sklearn import clone

from catenets.models import TWOSTEP_NAME, TwoStepNet
from catenets.models.snet import DEFAULT_UNITS_R_BIG_S, DEFAULT_UNITS_R_SMALL_S
from catenets.models.twostep_nets import S_STRATEGY, S1_STRATEGY
from catenets.models.transformation_utils import AIPW_TRANSFORMATION, RA_TRANSFORMATION
from catenets.experiments.experiment_utils import eval_root_mse, get_model_set
from catenets.experiments.simulation_utils import simulate_treatment_setup

# some constants
RESULT_DIR = 'results/simulations/'
CSV_STRING = '.csv'
SEP = '_'

# hyperparameters for experiments
LAYERS_OUT = 2
LAYERS_R = 3
PENALTY_L2 = 0.01 / 100
PENALTY_ORTHOGONAL = 1 / 100

MODEL_PARAMS = {'n_layers_out': LAYERS_OUT, 'n_layers_r': LAYERS_R, 'penalty_l2': PENALTY_L2,
                'penalty_orthogonal': PENALTY_ORTHOGONAL, 'n_layers_out_t': LAYERS_OUT,
                'n_layers_r_t': LAYERS_R, 'penalty_l2_t': PENALTY_L2}

# get basic models
ALL_MODELS = get_model_set(model_selection='all', model_params=MODEL_PARAMS)

# model-twostep combinations
COMBINED_MODELS = {TWOSTEP_NAME + SEP + AIPW_TRANSFORMATION + SEP + S_STRATEGY:
    TwoStepNet(
        transformation=AIPW_TRANSFORMATION, first_stage_strategy=S_STRATEGY,
        n_units_r=DEFAULT_UNITS_R_BIG_S, n_units_r_small=DEFAULT_UNITS_R_SMALL_S,
        n_layers_out=LAYERS_OUT, n_layers_r=LAYERS_R, penalty_l2_t=PENALTY_L2,
        penalty_l2=PENALTY_L2, n_layers_out_t=LAYERS_OUT,
        n_layers_r_t=LAYERS_R, penalty_orthogonal=PENALTY_ORTHOGONAL),
    TWOSTEP_NAME + SEP + RA_TRANSFORMATION + SEP + S_STRATEGY:
        TwoStepNet(
            transformation=RA_TRANSFORMATION, first_stage_strategy=S_STRATEGY,
            n_units_r=DEFAULT_UNITS_R_BIG_S, n_units_r_small=DEFAULT_UNITS_R_SMALL_S,
            penalty_orthogonal=PENALTY_ORTHOGONAL, n_layers_out=LAYERS_OUT, n_layers_r=LAYERS_R,
            penalty_l2_t=PENALTY_L2, penalty_l2=PENALTY_L2, n_layers_out_t=LAYERS_OUT,
            n_layers_r_t=LAYERS_R),
    TWOSTEP_NAME + SEP + AIPW_TRANSFORMATION + SEP + S1_STRATEGY:
        TwoStepNet(
            transformation=AIPW_TRANSFORMATION, first_stage_strategy=S1_STRATEGY,
            n_layers_out=LAYERS_OUT, n_layers_r=LAYERS_R, penalty_l2_t=PENALTY_L2,
            penalty_l2=PENALTY_L2, n_layers_out_t=LAYERS_OUT, n_layers_r_t=LAYERS_R),
    TWOSTEP_NAME + SEP + RA_TRANSFORMATION + SEP + S1_STRATEGY:
        TwoStepNet(
            transformation=RA_TRANSFORMATION, first_stage_strategy=S1_STRATEGY,
            n_layers_out=LAYERS_OUT, n_layers_r=LAYERS_R, penalty_l2_t=PENALTY_L2,
            penalty_l2=PENALTY_L2, n_layers_out_t=LAYERS_OUT, n_layers_r_t=LAYERS_R)
}

FULL_MODEL_SET = dict(**ALL_MODELS, ** COMBINED_MODELS)

# some more constants for experiments
NTRAIN_BASE = 2000
NTEST_BASE = 500
D_BASE = 25
BASE_XI = 3
TARGET_PROP_BASE = None

XI_STRING = 'xi'
N_STRING = 'n'
D_T_STRING = 'dim_t'
PROPENSITY_CONSTANT_STRING = 'p'
TARGET_STRING = 'target_p'


def simulation_experiment_loop(range_change, change_dim: str = N_STRING,
                               n_train: int = NTRAIN_BASE, n_test: int = NTEST_BASE,
                               n_repeats: int = 10, d: int = D_BASE, n_w: int = 0,
                               n_c: int = 5, n_o: int = 5, n_t: int = 0,
                               file_base: str = 'results', verbose: int = 1,
                               xi: float = BASE_XI, mu_1_model=None,
                               correlated_x: bool = False,
                               mu_1_model_params: dict = None,
                               mu_0_model_params: dict = None, models=None,
                               nonlinear_prop: bool = True,
                               prop_offset: float = 'center',
                               target_prop: float = TARGET_PROP_BASE):
    if change_dim is N_STRING:
        for n in range_change:
            if verbose > 0:
                print('Running experiments for {} set to {}'.format(N_STRING, n))
            do_one_experiment_repeat(n_train=n, n_test=n_test, n_repeats=n_repeats, d=d,
                                     n_w=n_w, n_c=n_c, n_o=n_o, n_t=n_t, file_base=file_base,
                                     verbose=verbose, xi=xi, mu_1_model=mu_1_model,
                                     correlated_x=correlated_x,
                                     models=models, mu_1_model_params=mu_1_model_params,
                                     mu_0_model_params=mu_0_model_params,
                                     nonlinear_prop=nonlinear_prop, prop_offset=prop_offset,
                                     target_prop=target_prop)
    elif change_dim is XI_STRING:
        for xi_temp in range_change:
            if verbose > 0:
                print('Running experiments for {} set to {}'.format(XI_STRING, xi_temp))
            do_one_experiment_repeat(n_train=n_train, n_test=n_test, n_repeats=n_repeats, d=d,
                                     n_w=n_w, n_c=n_c, n_o=n_o, n_t=n_t, file_base=file_base,
                                     verbose=verbose, xi=xi_temp, mu_1_model=mu_1_model,
                                     correlated_x=correlated_x,
                                     models=models, mu_1_model_params=mu_1_model_params,
                                     mu_0_model_params=mu_0_model_params,
                                     nonlinear_prop=nonlinear_prop, prop_offset=prop_offset,
                                     target_prop=target_prop)

    elif change_dim is D_T_STRING:
        for d_t_temp in range_change:
            if verbose > 0:
                print('Running experiments for {} set to {}'.format(D_T_STRING, d_t_temp))
            do_one_experiment_repeat(n_train=n_train, n_test=n_test, n_repeats=n_repeats, d=d,
                                     n_w=n_w, n_c=n_c, n_o=n_o, n_t=d_t_temp, file_base=file_base,
                                     verbose=verbose, xi=xi, mu_1_model=mu_1_model,
                                     correlated_x=correlated_x,
                                     models=models, mu_1_model_params=mu_1_model_params,
                                     mu_0_model_params=mu_0_model_params,
                                     nonlinear_prop=nonlinear_prop, prop_offset=prop_offset,
                                     target_prop=target_prop)

    elif change_dim is TARGET_STRING:
        for target_prop_temp in range_change:
            if verbose > 0:
                print(
                    'Running experiments for {} set to {}'.format(TARGET_STRING, target_prop_temp))
            do_one_experiment_repeat(n_train=n_train, n_test=n_test, n_repeats=n_repeats, d=d,
                                     n_w=n_w, n_c=n_c, n_o=n_o, n_t=n_t, file_base=file_base,
                                     verbose=verbose, xi=xi, mu_1_model=mu_1_model,
                                     correlated_x=correlated_x,
                                     models=models, mu_1_model_params=mu_1_model_params,
                                     mu_0_model_params=mu_0_model_params,
                                     nonlinear_prop=nonlinear_prop, prop_offset=prop_offset,
                                     target_prop=target_prop_temp)


def do_one_experiment_repeat(n_train: int = NTRAIN_BASE, n_test: int = NTEST_BASE,
                             n_repeats: int = 10, d: int = D_BASE, n_w: int = 0,
                             n_c: int = 0, n_o: int = 0, n_t: int = 0,
                             file_base: str = 'results', verbose: int = 1,
                             xi: float = BASE_XI, mu_1_model=None,
                             correlated_x: bool = True,
                             mu_1_model_params: dict = None,
                             mu_0_model_params: dict = None, models=None,
                             nonlinear_prop: bool = True, range_exp: list = None,
                             prop_offset: float = 0, target_prop: float = None):
    # make path
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if range_exp is None:
        range_exp = range(1, n_repeats + 1)

    if models is None:
        models = FULL_MODEL_SET

    if target_prop is None:
        prop_string = str(prop_offset)
    else:
        prop_string = str(target_prop)

    # create file name and file
    file_name = file_base + SEP + str(n_train) + SEP + str(d) + SEP + str(n_w) + SEP + str(n_c) + \
                SEP + str(n_o) + SEP + str(n_t) + SEP + str(xi) + SEP + prop_string

    out_file = open(RESULT_DIR + file_name + CSV_STRING, 'w', buffering=1)
    writer = csv.writer(out_file)
    header = [name for name in models.keys()]
    writer.writerow(header)

    for i in range_exp:
        if verbose > 0:
            print('Running experiment {}.'.format(i))
        mses = one_simulation_experiment(n_train=n_train, n_test=n_test, d=d, n_w=n_w,
                                         n_c=n_c, n_o=n_o, n_t=n_t, seed=i, xi=xi,
                                         mu_1_model=mu_1_model, correlated_x=correlated_x,
                                         models=models, nonlinear_prop=nonlinear_prop,
                                         verbose=verbose,
                                         mu_0_model_params=mu_0_model_params,
                                         mu_1_model_params=mu_1_model_params,
                                         prop_offset=prop_offset, target_prop=target_prop
                                         )
        writer.writerow(mses)

    out_file.close()
    return None


def one_simulation_experiment(n_train, n_test: int = NTEST_BASE, d: int = D_BASE, n_w: int = 0,
                              n_c: int = 0, n_o: int = 0, n_t: int = 0, xi: float = BASE_XI,
                              seed: int = 42, mu_1_model=None, propensity_model=None,
                              correlated_x: bool = False, verbose: int = 1,
                              mu_1_model_params: dict = None,
                              mu_0_model_params: dict = None, models=None,
                              nonlinear_prop: bool = False, prop_offset: float = 0,
                              target_prop: float = None):
    if models is None:
        models = FULL_MODEL_SET

    # get data
    X, y, w, p, t = simulate_treatment_setup(n_train + n_test, d=d, n_w=n_w, n_c=n_c, n_o=n_o,
                                             n_t=n_t,
                                             propensity_model=propensity_model,
                                             propensity_model_params={'xi': xi,
                                                                      'nonlinear': nonlinear_prop,
                                                                      'offset': prop_offset,
                                                                      'target_prop': target_prop},
                                             seed=seed,
                                             mu_1_model=mu_1_model,
                                             mu_0_model_params=mu_0_model_params,
                                             mu_1_model_params=mu_1_model_params,
                                             covariate_model_params={'correlated': correlated_x})
    # split data
    X_train, y_train, w_train, p_train = X[:n_train, :], y[:n_train], w[:n_train], p[:n_train]
    X_test, t_test = X[n_train:, :], t[n_train:]

    rmses = []
    for model_name, model in models.items():
        if verbose > 0:
            print('Training model {}'.format(model_name))

        estimator = clone(model)
        estimator.fit(X=X_train, y=y_train, w=w_train)

        cate_test = estimator.predict(X_test, return_po=False)
        rmses.append(eval_root_mse(cate_test, t_test))

    return rmses


def main_AISTATS(setting=1, models=None, file_name='res', n_repeats: int = 10):
    if type(models) is list or type(models) is str:
        models = get_model_set(models)
    if setting == 1:
        # no treatment effect, with confounding, by n
        simulation_experiment_loop([1000, 2000, 5000, 10000], change_dim='n', n_t=0, n_w=0,
                                   n_c=5, n_o=5, file_base=file_name, models=models,
                                   n_repeats=n_repeats)
    elif setting == 2:
        # with treatment effect, with confounding, by n
        simulation_experiment_loop([1000, 2000, 5000, 10000], change_dim='n', n_t=5, n_w=0,
                                   n_c=5, n_o=5, file_base=file_name, models=models,
                                   n_repeats=n_repeats)
    elif setting == 3:
        # Potential outcomes are supported on independent covariates, no confounding, by n
        simulation_experiment_loop([1000, 2000, 5000, 10000], change_dim='n', n_t=10, n_w=0,
                                   n_c=0, n_o=10, file_base=file_name, models=models, xi=0.5,
                                   mu_1_model_params={'withbase': False}, n_repeats=n_repeats)
    elif setting == 4:
        # vary number of predictive features at n=2000
        simulation_experiment_loop([0, 1, 3, 5, 7, 10], change_dim=D_T_STRING, n_train=2000, n_c=5,
                                   n_o=5, file_base=file_name, models=models, n_repeats=n_repeats)
    elif setting == 5:
        # vary percentage treated at n=2000
        simulation_experiment_loop([0.1, 0.2, 0.3, 0.4, 0.5], change_dim=TARGET_STRING,
                                   n_train=2000, n_c=5, n_o=5, n_t=0, n_repeats=n_repeats,
                                   file_base=file_name, models=models)
