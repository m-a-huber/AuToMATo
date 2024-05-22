from sklearn.base import ClusterMixin, BaseEstimator
import numpy as np
import pandas as pd
import subprocess
from gudhi.clustering.tomato import Tomato


class TTKSubclassed(ClusterMixin, BaseEstimator):
    """Subclassed version of the TTK clusterer from [1] to make it compatible
    with the API of scikit-learn.

    References:
        [1]: Cotsakis, R., Shaw, J., Tierny, J., & Levine, J. A. (2021).
            Implementing Persistence-Based Clustering of Point Clouds in the
            Topology ToolKit. In I. Hotz, T. Bin Masood, F. Sadlo, & J. Tierny
            (Eds.), Topological Methods in Data Analysis and Visualization VI
            (pp. 343–357). Springer International Publishing.
    """
    def __init__(
        self
    ):
        pass

    def fit(self, X, y=None):
        """Method that fits a TTKSubclassed instance to the point cloud to be
        clustered.

        Args:
            X (numpy.ndarray of shape (n_samples, dim)): NumPy-array containing
                the point cloud data to be clustered. Must be of shape
                (n_samples, dim), where n_samples is the number of data points
                and dim is the dimensionality of the point cloud.
            y: Not used, present here for API consistency with scikit-learn.

        Returns:
            :class:`eval.ttk_subclassed.ttk_subclassed.TTKSubclassed`:
                Fitted instance of the TTK clusterer.
        """
        df = _arr_to_df(X)
        df.to_csv("./eval/ttk_subclassed/tmp_in.csv", index=False)
        subprocess.run(
            ["pvpython", "./eval/ttk_subclassed/ttk_aux.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        df_out = pd.read_csv("./eval/ttk_subclassed/tmp_out.csv")
        subprocess.run(["rm", "./eval/ttk_subclassed/tmp_in.csv"])
        subprocess.run(["rm", "./eval/ttk_subclassed/tmp_out.csv"])
        persistence_pairs = df_out[["Points:0", "Points:1"]]
        lifetimes = np.diff(persistence_pairs.values, axis=1)
        lifetimes_sorted = np.sort(lifetimes.reshape(-1,))
        n_clusters = _get_n_clusters(lifetimes_sorted)
        t = Tomato(n_clusters=n_clusters).fit(X)
        self.labels_ = t.labels_
        return self


def _arr_to_df(arr):
    columns = ["X", "Y", "Z"]
    n_feat = arr.shape[1]
    df = pd.DataFrame(
        data=arr,
        columns=columns[:n_feat]
    )
    return df


def _get_n_clusters(lifetimes):
    a, b = 0.2, 0.025
    thresh = np.inf
    while thresh == np.inf and len(lifetimes) > 2:
        p_n = lifetimes[-1]
        cond_1 = lifetimes[1] > (1+a)*lifetimes[0] + b*p_n
        cond_2 = lifetimes[2] > (1+2*a)*lifetimes[0] + 2*b*p_n
        if cond_1 and cond_2:
            thresh = (1+a)*lifetimes[0] + b*p_n
        lifetimes = lifetimes[1:]
    features = lifetimes[lifetimes > thresh]
    return len(features)
