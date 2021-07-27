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
    DEFAULT_UNITS_R,
    DEFAULT_VAL_SPLIT,
)
from catenets.models.torch.base import BaseCATEEstimator, BasicNet


class TNet(BaseCATEEstimator):
    """
    TNet class -- two separate functions learned for each Potential Outcome function

    Parameters
    ----------
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    n_layers_r: int
        Number of representation layers before hypothesis layers (distinction between
        hypothesis layers and representation layers is made to match TARNet & SNets)
    n_units_r: int
        Number of hidden units in each representation layer
    weight_decay: float
        l2 (ridge) penalty
    lr: float
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
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool = False,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
    ) -> None:
        super(TNet, self).__init__()

        self._plug_in_0 = BasicNet(
            n_unit_in,
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

        self._plug_in_1 = BasicNet(
            n_unit_in,
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict treatment effects and potential outcomes
        Parameters
        ----------
        X: torch.Tensor of shape (n_samples, n_features)
            Test-sample features

        Returns
        -------
        y: torch.Tensor of shape (n_treatments, n_samples,)
        """
        X = torch.Tensor(X)

        y_0 = self._plug_in_0(X)
        y_1 = self._plug_in_1(X)

        return y_1 - y_0

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: Optional[torch.Tensor] = None,
    ) -> "TNet":
        """
        Train plug-in models.

        Parameters
        ----------
        X : torch.Tensor (n_samples, n_features)
            The features to fit to
        y : torch.Tensor (n_samples,) or (n_samples, )
            The outcome variable
        w: torch.Tensor (n_samples,)
            The treatment indicator
        p: array-like of shape (n_samples,)
            The treatment propensity
        """
        log.info("Train first network")
        self._plug_in_0.train(X[w == 0], y[w == 0])

        log.info("Train second network")
        self._plug_in_1.train(X[w == 1], y[w == 1])

        return self
