"""Non‑parametric Bayesian clustering using a Dirichlet process mixture.

This module wraps scikit‑learn's BayesianGaussianMixture to provide a simple
interface for clustering observations into an unknown number of groups.  The
BayesianGaussianMixture implements a Dirichlet process Gaussian mixture model
by placing a Dirichlet process prior on the mixture weights and learning
component means and covariances from data.  For the purposes of this project
we use diagonal covariance matrices for stability.

The clustering results are used to assign customers to latent groups so that
the reinforcement learning algorithm can account for heterogeneous demand.
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import BayesianGaussianMixture


class DirichletProcessClustering:
    """Cluster data using a Dirichlet process Gaussian mixture model.

    Parameters
    ----------
    max_components : int
        Upper bound on the number of mixture components.  The model may use
        fewer components if the data support a more parsimonious clustering.
    covariance_type : str
        Type of covariance matrix used for each component.  Options are
        'diag', 'full', 'tied' and 'spherical'.  We recommend 'diag' for
        high‑dimensional data.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, max_components: int = 10, covariance_type: str = "diag", random_state: int | None = None) -> None:
        self.max_components = max_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._model: BayesianGaussianMixture | None = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the mixture model and return cluster labels.

        Args:
            X: Data matrix of shape `(n_samples, n_features)`.

        Returns:
            Cluster labels for each sample, shape `(n_samples,)`.
        """
        self._model = BayesianGaussianMixture(
            n_components=self.max_components,
            covariance_type=self.covariance_type,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1.0,
            random_state=self.random_state,
        )
        self._model.fit(X)
        labels = self._model.predict(X)
        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data after fitting.

        Args:
            X: Data matrix of shape `(n_samples, n_features)`.

        Returns:
            Cluster labels, shape `(n_samples,)`.
        """
        if self._model is None:
            raise ValueError("Model must be fitted before prediction.")
        return self._model.predict(X)

    def get_num_clusters(self) -> int:
        """Return the number of components that have non‑zero weights.

        Returns:
            The count of active mixture components.
        """
        if self._model is None:
            raise ValueError("Model must be fitted before querying components.")
        # Count components whose weights exceed a small threshold
        weights = self._model.weights_
        return int(np.sum(weights > 1e-3))