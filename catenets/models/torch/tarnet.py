import numpy as np
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
from catenets.models.torch.base import BaseCATEEstimator, BasicNet, RepresentationNet
from catenets.models.torch.utils.model_utils import make_val_split


class TARNet(BaseCATEEstimator):
    """

    Parameters
    ----------
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
        n_unit_in: int,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R_BIG_S,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
    ) -> None:
        super(TARNet, self).__init__()

        self._representation_block = RepresentationNet(
            n_unit_in, n_units=n_units_r, nonlin=nonlin
        )
        self.val_split_prop = val_split_prop
        self.seed = seed
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print

        self._hypothesis_block = []
        for idx in range(2):
            self._hypothesis_block.append(
                BasicNet(
                    f"tarnet_hyp_{idx}",
                    n_units_r,
                    binary_y=binary_y,
                    n_layers_out=n_layers_out,
                    n_units_out=n_units_out,
                    weight_decay=weight_decay,
                    lr=lr,
                    n_iter=n_iter,
                    batch_size=batch_size,
                    val_split_prop=val_split_prop,
                    n_iter_print=n_iter_print,
                    seed=seed,
                    nonlin=nonlin,
                )
            )

        params = (
            list(self._representation_block.parameters())
            + list(self._hypothesis_block[0].parameters())
            + list(self._hypothesis_block[1].parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def loss(
        self,
        concat_pred: torch.Tensor,
        concat_true: torch.Tensor,
    ) -> torch.Tensor:
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]

        loss0 = torch.mean((1.0 - t_true) * torch.square(y_true - y0_pred))
        loss1 = torch.mean(t_true * torch.square(y_true - y1_pred))

        return loss0 + loss1

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> "TARNet":
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
        y = torch.Tensor(y).squeeze()
        w = torch.Tensor(w).squeeze()

        yw = torch.vstack((y, w)).T

        X, y, X_val, y_val, val_string = make_val_split(
            X, yw, val_split_prop=self.val_split_prop, seed=self.seed
        )
        y_val = y_val.squeeze()
        n = X.shape[0]  # could be different from before due to split

        # calculate number of batches per epoch
        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.round(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        # training
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

                preds = self._forward(X_next)
                batch_loss = self.loss(preds, y_next)

                batch_loss.backward()

                self.optimizer.step()

            if i % self.n_iter_print == 0:
                with torch.no_grad():
                    preds = self._forward(X_val)
                    val_loss = self.loss(preds, y_val)
                    log.info(f"Epoch: {i}, current {val_string} loss: {val_loss}")

        return self

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.Tensor(X)
        repr_preds = self._representation_block(X).squeeze()
        y0_preds = self._hypothesis_block[0](repr_preds).squeeze()
        y1_preds = self._hypothesis_block[1](repr_preds).squeeze()

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
        preds = self._forward(X)
        y0_preds = preds[:, 0]
        y1_preds = preds[:, 1]

        return y1_preds - y0_preds
