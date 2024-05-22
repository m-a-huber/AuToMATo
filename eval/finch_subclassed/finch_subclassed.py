from sklearn.base import ClusterMixin, BaseEstimator
import numpy as np
from finch import FINCH


class FINCHSubclassed(ClusterMixin, BaseEstimator):
    """Subclassed version of FINCH clusterer from [1] to make it compatible
    with the API of scikit-learn.

    References:
        [1]: https://github.com/ssarfraz/FINCH-Clustering
    """
    def __init__(
        self,
        initial_rank=None,
        req_clust=None,
        distance='euclidean',
        verbose=False
    ):
        self.initial_rank = initial_rank
        self.req_clust = req_clust
        self.distance = distance
        self.verbose = verbose

    def fit(self, X, y=None):
        """Method that fits a FINCHSubclassed instance to the point cloud to
        be clustered.

        Args:
            X (numpy.ndarray of shape (n_samples, dim)): NumPy-array containing
                the point cloud data to be clustered. Must be of shape
                (n_samples, dim), where n_samples is the number of data points
                and dim is the dimensionality of the point cloud.
            y: Not used, present here for API consistency with scikit-learn.

        Returns:
            :class:`eval.finch_subclassed.finch_subclassed.FINCHSubclassed`:
                Fitted instance of the FINCH clusterer.
        """
        c, num_clust, req_c = FINCH(
            data=X,
            initial_rank=self.initial_rank,
            req_clust=self.req_clust,
            distance=self.distance,
            verbose=self.verbose
        )
        self.labels_ = c.T[-1]
        self.n_clusters_ = len(np.unique(self.labels_))
        return self
