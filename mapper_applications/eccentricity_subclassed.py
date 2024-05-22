import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


# Taken form giotto-tda and modified to allow for L^{\infty}-metric
class EccentricitySubclassed(BaseEstimator, TransformerMixin):
    """Eccentricities of points in a point cloud or abstract metric space.

    Let `D` be a square matrix representing distances between points in a
    point cloud, or directly defining an abstract metric (or metric-like)
    space. The eccentricity of point `i` in the point cloud or abstract
    metric space is the `p`-norm (for some `p`) of row `i` in `D`.

    Parameters
    ----------
    exponent : int or float or np.inf, optional, default: ``2``
        `p`-norm exponent used to calculate eccentricities from the distance
        matrix.

    metric : str or function, optional, default: ``'euclidean'``
        Metric to use to compute the distance matrix if point cloud data is
        passed as input, or ``'precomputed'`` to specify that the input is
        already a distance matrix. If not ``'precomputed'``, it may be
        anything allowed by :func:`scipy.spatial.distance.pdist`.

    metric_params : dict, optional, default: ``{}``
        Additional keyword arguments for the metric function.

    """
    def __init__(self, exponent=2, metric='euclidean', metric_params={}):
        self.exponent = exponent
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method exists to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, \
            n_samples)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        # TODO: Consider making this transformer stateful so that the
        #  eccentricities of new points relative to the data seen in fit
        #  may be computed. May be useful for supervised tasks with Mapper?
        #  Evaluate performance impact of doing this.
        check_array(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Compute the eccentricities of points (i.e. rows) in  `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, \
            n_samples)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, 1)
            Column vector of eccentricities of points in `X`.

        """
        check_is_fitted(self, '_is_fitted')
        Xt = check_array(X)

        if self.metric != 'precomputed':
            Xt = squareform(
                pdist(Xt, metric=self.metric, **self.metric_params)
                )
        if self.exponent != np.inf:
            Xt = np.linalg.norm(Xt, axis=1, ord=self.exponent, keepdims=True)
        else:
            Xt = np.max(Xt, axis=1, keepdims=True)
        return Xt
