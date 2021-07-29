import abc
from typing import Optional

import numpy as np
import torch
from torch import nn

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
from catenets.models.torch.decorators import benchmark, check_input_train
from catenets.models.torch.model_utils import make_val_split
from catenets.models.torch.weight_utils import compute_importance_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}


class BasicNet(nn.Module):
    """
    Basic NN stub

    Parameters
    ----------
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    weight_decay: float
        l2 (ridge) penalty
    lr: float
        learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    nonlin: string, default 'elu'
        Nonlinearity to use in NN
    """

    def __init__(
        self,
        name: str,
        n_unit_in: int,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        binary_y: bool = False,
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

        self.name = name
        if nonlin not in ["elu", "relu", "sigmoid"]:
            raise ValueError("Unknown nonlinearity")

        NL = NONLIN[nonlin]
        layers = [nn.Linear(n_unit_in, n_units_out), NL()]

        # add required number of layers
        for i in range(n_layers_out):
            layers.extend([nn.Linear(n_units_out, n_units_out), NL()])

        # add final layers
        layers.append(nn.Linear(n_units_out, 1))
        if binary_y:
            layers.append(nn.Sigmoid())

        # return final architecture
        self.model = nn.Sequential(*layers).to(DEVICE)
        self.binary_y = binary_y

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.val_split_prop = val_split_prop

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def train(
        self, X: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> "BasicNet":
        self.loss = nn.BCELoss(weight=weight) if self.binary_y else nn.MSELoss()

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
            train_loss = []
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
                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss)

            if i % self.n_iter_print == 0:
                with torch.no_grad():
                    preds = self.forward(X_val).squeeze()
                    val_loss = self.loss(preds, y_val)
                    log.info(
                        f"[{self.name}] Epoch: {i}, current {val_string} loss: {val_loss}, train_loss: {torch.mean(train_loss)}"
                    )

        return self


class RepresentationNet(nn.Module):
    def __init__(
        self,
        n_unit_in: int,
        n_layers: int = DEFAULT_LAYERS_R,
        n_units: int = DEFAULT_UNITS_R,
        nonlin: str = DEFAULT_NONLIN,
    ) -> None:
        super(RepresentationNet, self).__init__()
        if nonlin not in ["elu", "relu", "sigmoid"]:
            raise ValueError("Unknown nonlinearity")

        NL = NONLIN[nonlin]

        layers = []

        layers = [nn.Linear(n_unit_in, n_units), NL()]
        # add required number of layers
        for i in range(n_layers - 1):
            layers.extend([nn.Linear(n_units, n_units), NL()])

        self.model = nn.Sequential(*layers).to(DEVICE)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class propensity_net(nn.Module):
    def __init__(
        self,
        n_unit_in: int,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        weighting_strategy: Optional[str] = None,
    ) -> None:
        super(propensity_net, self).__init__()

        NL = NONLIN[nonlin]

        layers = [
            nn.Linear(in_features=n_unit_in, out_features=n_units_out_prop),
            NL(),
        ]
        for idx in range(n_layers_out_prop):
            layers.extend(
                [
                    nn.Linear(
                        in_features=n_units_out_prop, out_features=n_units_out_prop
                    ),
                    NL(),
                ]
            )
        layers.extend(
            [
                nn.Linear(in_features=n_units_out_prop, out_features=n_units_out_prop),
                nn.Softmax(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def _get_importance_weights(self, X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.weighting_strategy is None:
            raise ValueError(
                "weighting_strategy must be valid for get_importance_weights"
            )

        p_pred = self._propensity_estimator(X)
        if p_pred.ndim > 1:
            if p_pred.shape[1] == 2:
                p_pred = p_pred[:, 1]
        return compute_importance_weights(p_pred, w, self.weighting_strategy, {})


class BaseCATEEstimator(nn.Module):
    """
    Interface for estimators of CATE
    """

    def __init__(
        self,
    ) -> None:
        super(BaseCATEEstimator, self).__init__()

    @abc.abstractmethod
    @check_input_train
    @benchmark
    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
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
        """
        ...

    @abc.abstractmethod
    @benchmark
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
