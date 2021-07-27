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
    DEFAULT_UNITS_R_BIG_S,
    DEFAULT_VAL_SPLIT,
)
from catenets.models.torch.model_utils import make_val_split
from catenets.models.torch.weight_utils import compute_importance_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}


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

        if nonlin not in ["elu", "relu", "sigmoid"]:
            raise ValueError("Unknown nonlinearity")

        NL = NONLIN[nonlin]
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


class RepresentationNet(nn.Module):
    def __init__(
        self,
        n_unit_in: int,
        n_layers: int = 3,
        n_units: int = 100,
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


class BaseCATEEstimator(nn.Module):
    """
    Base class for plug-in/ indirect estimators of CATE; such as S- and T-learners

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
        super(BaseCATEEstimator, self).__init__()
        self.binary_y = binary_y

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

    @abc.abstractmethod
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
