import subprocess

import numpy as np
import pandas as pd
from gudhi.clustering.tomato import Tomato
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.manifold import TSNE


class TTKSubclassed(ClusterMixin, BaseEstimator):
    """Subclassed version of the TTK clusterer from [1] to make it compatible
    with the API of scikit-learn.

    Parameters:
        random_state (int, optional): If not None, this number will be used as
            random seed in the computation of any tSNE-embeddings, allowing for
            reproducibility of results. Defaults to None.

    References:
        [1]: Cotsakis, R., Shaw, J., Tierny, J., & Levine, J. A. (2021).
            Implementing Persistence-Based Clustering of Point Clouds in the
            Topology ToolKit. In I. Hotz, T. Bin Masood, F. Sadlo, & J. Tierny
            (Eds.), Topological Methods in Data Analysis and Visualization VI
            (pp. 343–357). Springer International Publishing.
    """
    def __init__(
        self,
        random_state=None
    ):
        self.random_state = random_state

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
        df = _X_to_df(X, self.random_state)
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


def _X_to_df(X, random_state):
    n_samples, n_feat = X.shape
    if n_feat > 2:
        perplexity = min(30, n_samples-1)
        X = TSNE(
            perplexity=perplexity,
            random_state=random_state
        ).fit_transform(X)
    df = pd.DataFrame(
        data=X,
        columns=["X", "Y"]
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
