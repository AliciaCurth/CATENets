from typing import Optional

import torch

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_R,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_UNITS_OUT,
    DEFAULT_UNITS_R_BIG_S,
    DEFAULT_VAL_SPLIT,
)
from catenets.models.torch.base import BaseCATEEstimator, BasicNet
from catenets.models.torch.weight_utils import compute_importance_weights


class SNet(BaseCATEEstimator):
    """

    Parameters
    ----------
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_layers_out_prop: int
        Number of hypothesis layers for propensity score(n_layers_out x n_units_out + 1 x Dense
        layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    n_units_out_prop: int
        Number of hidden units in each propensity score hypothesis layer
    n_layers_r: int
        Number of shared & private representation layers before hypothesis layers
    n_units_r: int
        If withprop=True: Number of hidden units in representation layer shared by propensity score
        and outcome  function (the 'confounding factor') and in the ('instrumental factor')
        If withprop=False: Number of hidden units in representation shared across PO function
    penalty_l2: float
        l2 (ridge) penalty
    step_size: float
        learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    nonlin: string, default 'elu'
        Nonlinearity to use in NN
    weighting_strategy: optional str, None
        Whether to include propensity head and which weightening strategy to use
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R_BIG_S,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        weighting_strategy: Optional[str] = None,
    ) -> None:
        super(SNet, self).__init__()

        self._weighting_strategy = weighting_strategy

        self._propensity_estimator = BasicNet(
            n_unit_in,
            binary_y=binary_y,
            n_layers_out=n_layers_out_prop,
            n_units_out=n_units_out_prop,
            n_layers_r=n_layers_r,
            n_units_r=n_units_r,
            weight_decay=weight_decay,
            lr=lr,
            n_iter=n_iter,
            batch_size=batch_size,
            val_split_prop=val_split_prop,
            n_iter_print=n_iter_print,
            seed=seed,
            nonlin=nonlin,
        )
        self._output_estimator = BasicNet(
            n_unit_in + 1,
            binary_y=binary_y,
            n_layers_out=n_layers_out,
            n_units_out=n_units_out,
            n_layers_r=n_layers_r,
            n_units_r=n_units_r,
            weight_decay=weight_decay,
            lr=lr,
            n_iter=n_iter,
            batch_size=batch_size,
            val_split_prop=val_split_prop,
            n_iter_print=n_iter_print,
            seed=seed,
            nonlin=nonlin,
        )

    def _get_importance_weights(self, X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self._propensity_estimator is None:
            raise ValueError(
                "Can only call get_importance_weights if propensity_estimator is not None."
            )
        if self._weighting_strategy is None:
            raise ValueError(
                "weighting_strategy must be valid for get_importance_weights"
            )

        p_pred = self._propensity_estimator(X)
        if p_pred.ndim > 1:
            if p_pred.shape[1] == 2:
                p_pred = p_pred[:, 1]
        return compute_importance_weights(p_pred, w, self._weighting_strategy, {})

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> "SNet":
        """
        Fit treatment models.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            The features to fit to
        y : torch.Tensor of shape (n_samples,) or (n_samples, )
            The outcome variable
        w: torch.Tensor of shape (n_samples,)
            The treatment indicator
        """

        X = torch.Tensor(X)
        y = torch.Tensor(y)
        w = torch.Tensor(w)

        # add indicator as additional variable
        X_ext = torch.cat((X, w.reshape((-1, 1))), dim=1)

        if self._weighting_strategy is None:
            # fit standard S-learner
            log.info("Fit the outcome estimator")
            self._output_estimator.train(X_ext, y)
        else:
            # use reweighting within the outcome model
            log.info("Fit the propensity estimator")
            self._propensity_estimator.train(X, w)
            weights = self._get_importance_weights(X, w)
            log.info("Fit the outcome estimator with weights")
            self._output_estimator.train(X_ext, y, weight=weights)

        return self

    def _create_extended_matrices(self, X: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        X = torch.Tensor(X)

        # create extended matrices
        w_1 = torch.ones((n, 1))
        w_0 = torch.zeros((n, 1))
        X_ext_0 = torch.cat((X, w_0), dim=1)
        X_ext_1 = torch.cat((X, w_1), dim=1)

        return [X_ext_0, X_ext_1]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict treatment effects and potential outcomes

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        Returns
        -------
        y: array-like of shape (n_treatments, n_samples,)
        """
        X = torch.Tensor(X)
        X_ext = self._create_extended_matrices(X)

        y = []
        for ext_mat in X_ext:
            y.append(self._output_estimator(ext_mat))

        return y[1] - y[0]
