import abc
from typing import Optional

import numpy as np
import torch
from torch import nn

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS_OUT,
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
from catenets.models.torch.model_utils import make_val_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicNet(nn.Module):
    def __init__(
        self,
        n_unit_in: int,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        binary_y: bool = False,
        n_layers_r: int = 0,
        n_units_r: int = DEFAULT_UNITS_R,
        nonlin: str = DEFAULT_NONLIN,
        lr: float = DEFAULT_STEP_SIZE,
        weight_decay: float = DEFAULT_PENALTY_L2,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
    ) -> None:
        super(BasicNet, self).__init__()

        if nonlin == "elu":
            NL = nn.ELU
        elif nonlin == "relu":
            NL = nn.ReLU
        elif nonlin == "sigmoid":
            NL = nn.Sigmoid
        else:
            raise ValueError("Unknown nonlinearity")

        layers = [nn.Linear(n_unit_in, n_units_r), NL()]

        for i in range(n_layers_r - 1):
            layers.extend([nn.Linear(n_units_r, n_units_r), NL()])

        # add output layers
        layers.extend([nn.Linear(n_units_r, n_units_out), NL()])

        # add required number of layers
        for i in range(n_layers_out - 1):
            layers.extend([nn.Linear(n_units_out, n_units_out), NL()])

        # add final layers
        layers.append(nn.Linear(n_units_out, 1))
        if binary_y:
            layers.append(nn.Sigmoid())

        # return final architecture
        self.model = nn.Sequential(*layers).to(DEVICE)

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.val_split_prop = val_split_prop

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        if binary_y:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.MSELoss()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def train(self, X: torch.Tensor, y: torch.Tensor) -> "BasicNet":
        X = torch.Tensor(X).to(DEVICE)
        y = torch.Tensor(y).to(DEVICE)

        # get validation split (can be none)
        X, y, X_val, y_val, val_string = make_val_split(
            X, y, val_split_prop=self.val_split_prop, seed=self.seed
        )
        y_val = y_val.squeeze()
        n = X.shape[0]  # could be different from before due to split

        # calculate number of batches per epoch
        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.round(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        # do training
        for i in range(self.n_iter):
            # shuffle data for minibatches
            np.random.shuffle(train_indices)
            for b in range(n_batches):
                self.optimizer.zero_grad()

                idx_next = train_indices[
                    (b * batch_size) : min((b + 1) * batch_size, n - 1)
                ]

                X_next = X[idx_next]
                y_next = y[idx_next].squeeze()

                preds = self.forward(X_next).squeeze()

                batch_loss = self.loss(preds, y_next)

                batch_loss.backward()

                self.optimizer.step()

            if i % self.n_iter_print == 0:
                with torch.no_grad():
                    preds = self.forward(X_val).squeeze()
                    val_loss = self.loss(preds, y_val)
                    log.info(f"Epoch: {i}, current {val_string} loss: {val_loss}")

        return self


class BaseCATEEstimator(nn.Module):
    """
    Base class for plug-in/ indirect estimators of CATE; such as S- and T-learners

    Parameters
    ----------
    po_estimator: nn.Module
        Estimator to be used for potential outcome regressions.
    binary_y: bool, default False
        Whether the outcome data is binary
    propensity_estimator: estimator, default None
        Estimator to be used for propensity score estimation (if needed)
    """

    def __init__(
        self,
        binary_y: bool = False,
    ) -> None:
        super(BaseCATEEstimator, self).__init__()
        self.binary_y = binary_y

    @abc.abstractmethod
    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        p: Optional[torch.Tensor] = None,
    ) -> "BaseCATEEstimator":
        """
        Train method for a CATEModel

        Parameters
        ----------
        X: torch.Tensor
            Covariate matrix
        y: torch.Tensor
            Outcome vector
        w: torch.Tensor
            Treatment indicator
        p: torch.Tensor
            Vector of treatment propensities.
        """
        ...

    @abc.abstractmethod
    def forward(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict treatment effect estimates using a CATEModel.

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Covariate matrix

        Returns
        -------
        potential outcomes probabilities
        """
        ...

    @staticmethod
    def _check_inputs(w: torch.Tensor, p: Optional[torch.Tensor]) -> None:
        if p is not None:
            if np.sum(p > 1) > 0 or np.sum(p < 0) > 0:
                raise ValueError("p should be in [0,1]")

        if not ((w == 0) | (w == 1)).all():
            raise ValueError("W should be binary")
