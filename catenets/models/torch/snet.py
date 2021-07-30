import abc
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import nn

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_R,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_UNITS_OUT,
    DEFAULT_UNITS_R,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from catenets.models.torch.base import (
    DEVICE,
    BaseCATEEstimator,
    BasicNet,
    PropensityNet,
    RepresentationNet,
)
from catenets.models.torch.utils.model_utils import make_val_split

EPS = 1e-8


class BaseSNet(BaseCATEEstimator):
    """

    Parameters
    ----------
    name: str
        Estimator name
    n_unit_in: int
        Number of features
    propensity_estimator: nn.Module
        Propensity estimator
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
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
        name: str,
        n_unit_in: int,
        propensity_estimator: nn.Module,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        weighting_strategy: Optional[str] = None,
    ) -> None:
        super(BaseSNet, self).__init__()

        self.name = name
        self.val_split_prop = val_split_prop
        self.seed = seed
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.lr = lr
        self.weight_decay = weight_decay

        self._repr_estimator = RepresentationNet(
            n_unit_in, n_units=n_units_r, n_layers=n_layers_r, nonlin=nonlin
        )
        self._po_estimators = []
        for idx in range(2):
            self._po_estimators.append(
                BasicNet(
                    f"{name}_po_estimator_{idx}",
                    n_units_r,
                    binary_y=binary_y,
                    n_layers_out=n_layers_out,
                    n_units_out=n_units_out,
                    nonlin=nonlin,
                )
            )
        self._propensity_estimator = propensity_estimator

    def loss(
        self,
        po_pred: torch.Tensor,
        t_pred: torch.Tensor,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
    ) -> torch.Tensor:
        def po_loss(
            po_pred: torch.Tensor, y_true: torch.Tensor, t_true: torch.Tensor
        ) -> torch.Tensor:
            y0_pred = po_pred[:, 0]
            y1_pred = po_pred[:, 1]

            loss0 = torch.mean((1.0 - t_true) * torch.square(y_true - y0_pred))
            loss1 = torch.mean(t_true * torch.square(y_true - y1_pred))

            return loss0 + loss1

        def prop_loss(t_pred: torch.Tensor, t_true: torch.Tensor) -> torch.Tensor:
            t_pred = t_pred + EPS
            return nn.CrossEntropyLoss()(t_pred, t_true)

        return po_loss(po_pred, y_true, t_true) + prop_loss(t_pred, t_true)

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> "BaseSNet":
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
        X = torch.Tensor(X).to(DEVICE)
        y = torch.Tensor(y).squeeze().to(DEVICE)
        w = torch.Tensor(w).squeeze().long().to(DEVICE)

        X, y, w, X_val, y_val, w_val, val_string = make_val_split(
            X, y, w=w, val_split_prop=self.val_split_prop, seed=self.seed
        )

        n = X.shape[0]  # could be different from before due to split

        # calculate number of batches per epoch
        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.round(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        params = (
            list(self._repr_estimator.parameters())
            + list(self._po_estimators[0].parameters())
            + list(self._po_estimators[1].parameters())
            + list(self._propensity_estimator.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # training
        val_loss_best = LARGE_VAL
        patience = 0
        for i in range(self.n_iter):
            # shuffle data for minibatches
            np.random.shuffle(train_indices)
            train_loss = []
            for b in range(n_batches):
                optimizer.zero_grad()

                idx_next = train_indices[
                    (b * batch_size) : min((b + 1) * batch_size, n - 1)
                ]

                X_next = X[idx_next]
                y_next = y[idx_next].squeeze()
                w_next = w[idx_next].squeeze()

                po_preds, prop_preds = self._step(X_next)
                batch_loss = self.loss(po_preds, prop_preds, y_next, w_next)

                batch_loss.backward()

                optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if i % self.n_iter_print == 0:
                with torch.no_grad():
                    po_preds, prop_preds = self._step(X_val)
                    val_loss = self.loss(po_preds, prop_preds, y_val, w_val)
                    if val_loss_best > val_loss:
                        val_loss_best = val_loss
                        patience = 0
                    else:
                        patience += 1
                    if patience > DEFAULT_PATIENCE and i > DEFAULT_N_ITER_MIN:
                        break

                    log.info(
                        f"[{self.name}] Epoch: {i}, current {val_string} loss: {val_loss} train_loss: {torch.mean(train_loss)}"
                    )

        return self

    @abc.abstractmethod
    def _step(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._check_tensor(X)
        repr_preds = self._repr_estimator(X).squeeze()
        y0_preds = self._po_estimators[0](repr_preds).squeeze()
        y1_preds = self._po_estimators[1](repr_preds).squeeze()

        return torch.vstack((y0_preds, y1_preds)).T

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict treatment effects and potential outcomes

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        Returns
        -------
        y: array-like of shape (n_samples,)
        """
        X = self._check_tensor(X).float()
        preds = self._forward(X)
        y0_preds = preds[:, 0]
        y1_preds = preds[:, 1]

        return y1_preds - y0_preds


class TARNet(BaseSNet):
    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool = False,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
        nonlin: str = DEFAULT_NONLIN,
        **kwargs: Any,
    ) -> None:
        propensity_estimator = PropensityNet(
            "tarnet_propensity_estimator",
            n_unit_in,
            2,
            "prop",
            n_layers_out_prop=n_layers_out_prop,
            n_units_out_prop=n_units_out_prop,
            nonlin=nonlin,
        ).to(DEVICE)
        super(TARNet, self).__init__(
            "TARNet",
            n_unit_in,
            propensity_estimator,
            binary_y=binary_y,
            n_layers_out_prop=n_layers_out_prop,
            n_units_out_prop=n_units_out_prop,
            nonlin=nonlin,
            **kwargs,
        )

    def _step(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        po_preds = self._forward(X)
        prop_preds = self._propensity_estimator(X)

        return po_preds, prop_preds


class DragonNet(BaseSNet):
    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool = False,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = 0,
        nonlin: str = DEFAULT_NONLIN,
        n_units_r: int = DEFAULT_UNITS_R,
        **kwargs: Any,
    ) -> None:
        propensity_estimator = PropensityNet(
            "dragonnet_propensity_estimator",
            n_units_r,
            2,
            "prop",
            n_layers_out_prop=n_layers_out_prop,
            n_units_out_prop=n_units_out_prop,
            nonlin=nonlin,
        ).to(DEVICE)
        super(DragonNet, self).__init__(
            "DragonNet",
            n_unit_in,
            propensity_estimator,
            binary_y=binary_y,
            n_layers_out_prop=n_layers_out_prop,
            n_units_out_prop=n_units_out_prop,
            nonlin=nonlin,
            **kwargs,
        )

    def _step(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        repr_preds = self._repr_estimator(X).squeeze()

        y0_preds = self._po_estimators[0](repr_preds).squeeze()
        y1_preds = self._po_estimators[1](repr_preds).squeeze()

        po_preds = torch.vstack((y0_preds, y1_preds)).T

        prop_preds = self._propensity_estimator(repr_preds)

        return po_preds, prop_preds
