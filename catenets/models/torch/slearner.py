from typing import Any, Optional

import torch

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
    DEFAULT_VAL_SPLIT,
)
from catenets.models.torch.base import (
    DEVICE,
    BaseCATEEstimator,
    BasicNet,
    PropensityNet,
)


class SLearner(BaseCATEEstimator):
    """

    Parameters
    ----------
    n_unit_in: int
        Number of features
    binary_y: bool
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
        binary_y: bool,
        po_estimator: Any = None,
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
        super(SLearner, self).__init__()

        self._weighting_strategy = weighting_strategy
        if po_estimator is not None:
            self._po_estimator = po_estimator
        else:
            self._po_estimator = BasicNet(
                "slearner_po_estimator",
                n_unit_in + 1,
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
            ).to(DEVICE)
        if weighting_strategy is not None:
            self._propensity_estimator = PropensityNet(
                "slearner_prop_estimator",
                n_unit_in,
                2,  # number of treatments
                weighting_strategy,
                n_units_out_prop=n_units_out_prop,
                n_layers_out_prop=n_layers_out_prop,
                weight_decay=weight_decay,
                lr=lr,
                n_iter=n_iter,
                batch_size=batch_size,
                n_iter_print=n_iter_print,
                seed=seed,
                nonlin=nonlin,
                val_split_prop=val_split_prop,
            ).to(DEVICE)

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> "SLearner":
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
        y = torch.Tensor(y).to(DEVICE)
        w = torch.Tensor(w).to(DEVICE)

        # add indicator as additional variable
        X_ext = torch.cat((X, w.reshape((-1, 1))), dim=1).to(DEVICE)

        if not (
            hasattr(self._po_estimator, "train") or hasattr(self._po_estimator, "fit")
        ):
            raise NotImplementedError("invalid po_estimator for the slearner")

        if hasattr(self._po_estimator, "fit"):
            log.info("Fit the sklearn po_estimator")
            self._po_estimator.fit(X_ext.detach().numpy(), y.detach().numpy())
            return self

        if self._weighting_strategy is None:
            # fit standard S-learner
            log.info("Fit the PyTorch po_estimator")
            self._po_estimator.train(X_ext, y)
            return self

        # use reweighting within the outcome model
        log.info("Fit the PyTorch po_estimator with the propensity estimator")
        self._propensity_estimator.train(X, w)
        weights = self._propensity_estimator.get_importance_weights(X, w)
        self._po_estimator.train(X_ext, y, weight=weights)

        return self

    def _create_extended_matrices(self, X: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        X = self._check_tensor(X)

        # create extended matrices
        w_1 = torch.ones((n, 1))
        w_0 = torch.zeros((n, 1))
        X_ext_0 = torch.cat((X, w_0), dim=1).to(DEVICE)
        X_ext_1 = torch.cat((X, w_1), dim=1).to(DEVICE)

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
        y: array-like of shape (n_samples,)
        """
        X = self._check_tensor(X).float()
        X_ext = self._create_extended_matrices(X)

        y = []
        for ext_mat in X_ext:
            if hasattr(self._po_estimator, "forward"):
                y.append(self._po_estimator(ext_mat))
            elif hasattr(self._po_estimator, "predict_proba"):
                ext_mat_np = ext_mat.detach().numpy()
                no_event_proba = self._po_estimator.predict_proba(ext_mat_np)[
                    :, 0
                ]  # no event probability

                y.append(torch.Tensor(no_event_proba).to(DEVICE))
            elif hasattr(self._po_estimator, "predict"):
                ext_mat_np = ext_mat.detach().numpy()
                no_event_proba = self._po_estimator.predict(ext_mat_np)

                y.append(torch.Tensor(no_event_proba).to(DEVICE))
            else:
                raise NotImplementedError("Invalid po_estimator for slearner")

        return y[1] - y[0]
