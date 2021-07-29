import abc
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CF_FOLDS,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_OUT_T,
    DEFAULT_LAYERS_R,
    DEFAULT_LAYERS_R_T,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_STEP_SIZE_T,
    DEFAULT_UNITS_OUT,
    DEFAULT_UNITS_OUT_T,
    DEFAULT_UNITS_R,
    DEFAULT_UNITS_R_T,
    DEFAULT_VAL_SPLIT,
)
from catenets.models.torch.base import BaseCATEEstimator, BasicNet
from catenets.models.torch.transformations import ra_transformation_cate


class PseudoOutcomeNet(BaseCATEEstimator):
    """
    Class implements TwoStepLearners based on pseudo-outcome regression as discussed in
    Curth &vd Schaar (2021): RA-learner, PW-learner and DR-learner

    Parameters
    ----------
    n_unit_in: int
        Number of features
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        First stage Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out: int
        First stage Number of hidden units in each hypothesis layer
    n_layers_r: int
        First stage Number of representation layers before hypothesis layers (distinction between
        hypothesis layers and representation layers is made to match TARNet & SNets)
    n_units_r: int
        First stage Number of hidden units in each representation layer
    n_layers_out_t: int
        Second stage Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out_t: int
        Second stage Number of hidden units in each hypothesis layer
    n_layers_r_t: int
        Second stage Number of representation layers before hypothesis layers (distinction between
        hypothesis layers and representation layers is made to match TARNet & SNets)
    n_units_r_t: int
        Second stage Number of hidden units in each representation layer
    n_layers_out_prop: int
        Number of hypothesis layers for propensity score(n_layers_out x n_units_out + 1 x Dense
        layer)
    n_units_out_prop: int
        Number of hidden units in each propensity score hypothesis layer
    penalty_l2: float
        First stage l2 (ridge) penalty
    penalty_l2_t: float
        Second stage l2 (ridge) penalty
    step_size: float
        First stage learning rate for optimizer
    step_size_t: float
        Second stage learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    early_stopping: bool, default True
        Whether to use early stopping
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    nonlin: string, default 'elu'
        Nonlinearity to use in NN
    n_folds: int, default 1
        Number of cross-fitting folds. If 1, no cross-fitting
    """

    def __init__(
        self,
        n_unit_in: int,
        n_folds: int = DEFAULT_CF_FOLDS,
        binary_y: bool = True,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
        n_layers_r_t: int = DEFAULT_LAYERS_R_T,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_units_r: int = DEFAULT_UNITS_R,
        n_units_out_t: int = DEFAULT_UNITS_OUT_T,
        n_units_r_t: int = DEFAULT_UNITS_R_T,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
        weight_decay: float = DEFAULT_PENALTY_L2,
        weight_decay_t: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        lr_t: float = DEFAULT_STEP_SIZE_T,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        weighting_strategy: str = "prop",
    ):
        super(PseudoOutcomeNet, self).__init__()
        self.n_unit_in = n_unit_in
        self.binary_y = binary_y
        self.n_layers_out = n_layers_out
        self.n_units_out = n_units_out
        self.n_layers_r = n_layers_r
        self.n_units_r = n_units_r
        self.weight_decay_t = weight_decay_t
        self.weight_decay = weight_decay_t
        self.lr = lr_t
        self.lr_t = lr_t
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.nonlin = nonlin
        self.lr = lr

        # set other arguments
        self.n_folds = n_folds
        self.random_state = seed
        self.binary_y = binary_y

        # set estimators
        self._te_estimator = self._generate_te_estimator()
        self._po_estimator = self._generate_te_estimator()

    def _generate_te_estimator(self, name: str = "te_estimator") -> nn.Module:
        return BasicNet(
            name,
            self.n_unit_in,
            binary_y=False,
            n_layers_out=self.n_layers_out,
            n_units_out=self.n_units_out,
            weight_decay=self.weight_decay_t,
            lr=self.lr_t,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
        )

    def _generate_po_estimator(self, name: str = "po_estimator") -> nn.Module:
        return BasicNet(
            name,
            self.n_unit_in,
            binary_y=False,
            n_layers_out=self.n_layers_out,
            n_units_out=self.n_units_out,
            weight_decay=self.weight_decay,
            lr=self.lr,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            n_iter_print=self.n_iter_print,
            seed=self.seed,
            nonlin=self.nonlin,
        )

    def train(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor
    ) -> "PseudoOutcomeNet":
        X = torch.Tensor(X)
        y = torch.Tensor(y).squeeze()
        w = torch.Tensor(w).squeeze()

        n = len(y)

        # STEP 1: fit plug-in estimators via cross-fitting
        mu_0_pred, mu_1_pred, p_pred = torch.zeros(n), torch.zeros(n), torch.zeros(n)

        # create folds stratified by treatment assignment to ensure balance
        splitter = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        for train_index, test_index in splitter.split(X, w):
            # create masks
            pred_mask = torch.zeros(n, dtype=bool)
            pred_mask[test_index] = 1

            # fit plug-in te_estimator
            (
                mu_0_pred[pred_mask],
                mu_1_pred[pred_mask],
                p_pred[pred_mask],
            ) = self._first_step(X, y, w, ~pred_mask, pred_mask)

        # use estimated propensity scores
        p = p_pred

        # STEP 2: direct TE estimation
        self._second_step(X, y, w, p, mu_0_pred, mu_1_pred)

        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        X = torch.Tensor(X)
        return self._te_estimator(X)

    @abc.abstractmethod
    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pass

    def _impute_pos(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # split sample
        X_fit, Y_fit, W_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]

        # fit two separate (standard) models
        # untreated model
        temp_model_0 = self._generate_po_estimator("po_estimator_0_impute_pos")
        temp_model_0.train(X_fit[W_fit == 0], Y_fit[W_fit == 0])

        # treated model
        temp_model_1 = self._generate_po_estimator("po_estimator_1_impute_pos")
        temp_model_1.train(X_fit[W_fit == 1], Y_fit[W_fit == 1])

        mu_0_pred = temp_model_0(X[pred_mask, :])
        mu_1_pred = temp_model_1(X[pred_mask, :])

        return mu_0_pred, mu_1_pred


class RANet(PseudoOutcomeNet):
    """
    RA-learner for CATE estimation, based on singly robust regression-adjusted pseudo-outcome
    """

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = np.nan  # not needed
        return mu0_pred.squeeze(), mu1_pred.squeeze(), p_pred

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        pseudo_outcome = ra_transformation_cate(y, w, p, mu_0, mu_1)
        self._te_estimator.train(X, pseudo_outcome.detach())


class XNet(PseudoOutcomeNet):
    """
    X-learner for CATE estimation. Combines two CATE estimates via a weighting function g(x):
    tau(x) = g(x) tau_0(x) + (1-g(x)) tau_1(x)
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool = False,
        n_folds: int = DEFAULT_CF_FOLDS,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
        n_layers_r_t: int = DEFAULT_LAYERS_R_T,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_units_r: int = DEFAULT_UNITS_R,
        n_units_out_t: int = DEFAULT_UNITS_OUT_T,
        n_units_r_t: int = DEFAULT_UNITS_R_T,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
        weight_decay: float = DEFAULT_PENALTY_L2,
        weight_decay_t: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        lr_t: float = DEFAULT_STEP_SIZE_T,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        weighting_strategy: str = "prop",
    ) -> None:
        super().__init__(
            n_unit_in,
            binary_y=binary_y,
            n_folds=n_folds,
            n_layers_out=n_layers_out,
            n_layers_r=n_layers_r,
            n_layers_out_t=n_layers_out_t,
            n_layers_r_t=n_layers_r_t,
            n_units_out=n_units_out,
            n_units_r=n_units_r,
            n_units_out_t=n_units_out_t,
            n_units_r_t=n_units_r_t,
            n_units_out_prop=n_units_out_prop,
            n_layers_out_prop=n_layers_out_prop,
            weight_decay=weight_decay,
            weight_decay_t=weight_decay_t,
            lr=lr,
            lr_t=lr_t,
            n_iter=n_iter,
            batch_size=batch_size,
            val_split_prop=val_split_prop,
            n_iter_print=n_iter_print,
            seed=seed,
            nonlin=nonlin,
            weighting_strategy=weighting_strategy,
        )
        self.weighting_strategy = weighting_strategy

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = np.nan
        return mu0_pred.squeeze(), mu1_pred.squeeze(), p_pred

    def _second_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mu_0: torch.Tensor,
        mu_1: torch.Tensor,
    ) -> None:
        # split by treatment status, fit one model per group
        pseudo_0 = mu_1[w == 0] - y[w == 0]
        self._te_estimator_0 = self._generate_te_estimator("te_estimator_0_xnet")
        self._te_estimator_0.train(X[w == 0], pseudo_0.detach())

        pseudo_1 = y[w == 1] - mu_0[w == 1]
        self._te_estimator_1 = self._generate_te_estimator("te_estimator_1_xnet")
        self._te_estimator_1.train(X[w == 1], pseudo_1.detach())

        if self.weighting_strategy == "prop" or self.weighting_strategy == "1-prop":
            self._propensity_estimator.train(X, w)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions. Placeholder, can only accept False.
        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        X = torch.Tensor(X)
        tau0_pred = self._te_estimator_0(X)
        tau1_pred = self._te_estimator_1(X)

        if self.weighting_strategy == "prop" or self.weighting_strategy == "1-prop":
            prop_pred = self._propensity_estimator(X)

        if self.weighting_strategy == "prop":
            weight = prop_pred
        elif self.weighting_strategy == "1-prop":
            weight = 1 - prop_pred
        else:
            raise ValueError("invalid value for self.weighting_strategy")

        return weight * tau0_pred + (1 - weight) * tau1_pred
