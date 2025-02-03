import igraph as ig  # type: ignore
import numpy as np
from gudhi.clustering.tomato import Tomato  # type: ignore
from gudhi.point_cloud.knn import KNearestNeighbors as KNN  # type: ignore
from sklearn.base import BaseEstimator, ClusterMixin  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from bottleneck_bootstrap import BottleneckBootstrap
from persistence_plotting import plot_persistences


class _Customato(Tomato, ClusterMixin, BaseEstimator):
    """This is a helper class to facilitate the integration of
    bootstrap_persistence.BootstrapPersistence into AuToMATo.
    """
    def __init__(self, **params):
        Tomato.__init__(self, **params)

    def fit(self, X, **fit_params):
        Tomato.fit(self, X, **fit_params)
        self.persistence_ = self._get_persistence_()
        return self

    def _get_persistence_(self):
        self._min_weight_ = np.min(self.weights_)
        finite_pts = self.diagram_
        infinite_pts = np.array([
                [infinite_birth, -np.inf]
                for infinite_birth in self.max_weight_per_cc_
            ])
        all_pts = np.concatenate([finite_pts, infinite_pts])
        return [-all_pts]


class Automato(_Customato, ClusterMixin, BaseEstimator):
    """This class implements the AuToMATo clusterer, an automated version of
    the ToMATo clusterer defined by Chazal et al. in [1].
    Specifically, AuToMATo automates the selection of the number of clusters
    based on the persistence diagram produced by the ToMATo clusterer. It does
    so by separating noise from features in said persistence diagram by
    performing a bootstrap on the diagram as described in §6 of [2].

    Parameters:
        create_outliers (bool, optional): Experimental feature; whether or not
            the AuToMATo clusterer should be allowed to label certain points
            as outliers. If True, outliers will be assigned the label -1, and
            only the remaining points will be considered during clustering.
            Defaults to False.
        ratio_outliers (float, optional): The minimum percentage of edges of a
            point in the underlying neighborhood graph that must be outgoing
            for the point not to be considered an outlier. Must be a
            non-negative number less than or equal to 1. Defaults to 0.1.
        alpha (float, optional): Confidence level of the bootstrap. Writing to
            it automatically adjusts `n_clusters_` and `labels_`.
            Defaults to 0.35.
        n_bootstrap (int, optional): Number of bootstrap samples to use.
            Defaults to 1000.
        tomato_params (dict, optional): A dictionary containing arguments
            that are passed to the underlying instance of
            :class:`~gudhi.clustering.tomato.Tomato`, such as `density_type`,
            `metric` etc. Defaults to `dict()`.
        parallelize (int, optional): Whether or not to parallelize the
            computation of the estimator on the bootstrap samples by using
            the joblib-library. 0 means no parallelization, a positive integer
            i means parallelization using i processors, while a negative
            integer -i means parallelization using all but (i-1) processors so
            that e.g. -1 means using all processors. Defaults to -1.
        random_state (int, optional): If not None, this number will be used as
            random seed in the bootstrap procedure, allowing for
            reproducibility of results. Defaults to None.

    Attributes:
        points_ (numpy.ndarray of shape (n_samples, dim)): NumPy-array
            containing the point cloud data that the Automato instance was
            fitted on.
        n_clusters_ (int): The number of clusters consisting of points not
            labelled as outliers. Unlike when using the ToMATo clusterer, this
            attribute is not intended to be changed manually.
        labels_ (numpy.ndarray of shape (n_samples,)): Cluster labels for each
            data point, where outliers are assigned the label -1.
        width_conf_band_ (float): The width of the confidence band found by
            the bootstrapping procedure.
        diagram_ (numpy.ndarray of shape (n_generators, 2)): A NumPy-array
            whose entries are the birth and death times of each homological
            generator of the superlevel filtration found by the underlying
            ToMATo instance. The entries are sorted by decreasing lifetime,
            i.e. by increasing lifetime in absolute value.
        weights_ (numpy.ndarray of shape (n_generators,)): A NumPy-array that
            contains the density for each point as computed by the density
            estimator used. If `create_outliers` is True, the density assigned
            to any outlier is 0 or -np.inf according to whether `density_type`
            is `"DTM"` or `"KDE"`, or `"logDTM"` or `"logKDE"`, respectively.
        n_outliers_ (int): The number of points among X that are found to be
            outliers and assigned the label -1 by the clusterer.

    References:
        [1]: Frédéric Chazal, Leonidas J. Guibas, Steve Y. Oudot, and Primoz
            Skraba. 2013. Persistence-Based Clustering in Riemannian Manifolds.
            J. ACM 60, 6, Article 41 (November 2013), 38 pages.
            https://doi.org/10.1145/2535927
        [2]: Chazal, F., Fasy, B., Lecci, F., Michel, B., Rinaldo, A., &
            Wasserman, L. (2018). Robust Topological Inference: Distance To a
            Measure and Kernel Distance. Journal of Machine Learning Research,
            18(159), 1–40. http://jmlr.org/papers/v18/15-484.html

    Examples:
        >>> from automato import Automato
        >>> from sklearn.datasets import make_blobs
        >>> X, y = make_blobs(centers=2, random_state=42)
        >>> aut = Automato(random_state=42).fit(X)
        >>> aut.n_clusters_
        2
        >>> (aut.labels_ == y).all()
        True
    """
    def __init__(
            self,
            create_outliers=False,
            ratio_outliers=0.1,
            alpha=0.35,
            n_bootstrap=1000,
            tomato_params=dict(),
            parallelize=-1,
            random_state=None
    ):
        _Customato.__init__(self, **tomato_params)
        self.create_outliers = create_outliers
        self.ratio_outliers = ratio_outliers
        self._alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.tomato_params = tomato_params
        self.parallelize = parallelize
        self.random_state = random_state

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if alpha == self._alpha:
            pass
        else:
            check_is_fitted(self, attributes="_bootstrapper")
            self._alpha = alpha
            self._bootstrapper.alpha = alpha
            self._essential_pts_ = np.concatenate(
                self._bootstrapper.cleaned_persistence_
            )
            self.width_conf_band_ = self._bootstrapper.width_conf_band_
            self.n_clusters_ = self._bootstrapper.n_top_features_

    def fit(self, X, y=None, tomato_fit_params=dict()):
        """Method that fits an AuToMATo instance to the point cloud to be
        clustered.

        Args:
            X (numpy.ndarray of shape (n_samples, dim)): NumPy-array containing
                the point cloud data to be clustered. Must be of shape
                (n_samples, dim), where n_samples is the number of data points
                and dim is the dimensionality of the point cloud.
            y: Not used, present here for API consistency with scikit-learn.
            tomato_fit_params (dict, optional): A dictionary containing
                arguments that are passed to the fit()-method of the underlying
                instance of :class:`~gudhi.clustering.tomato.Tomato`.
                Defaults to `dict()`

        Returns:
            :class:`automato.Automato`: Fitted instance of the AuToMATo
                clusterer.
        """
        self.tomato_fit_params = tomato_fit_params
        if len(X) == 1:
            self._get_fitted_attributes(X, tomato_fit_params)
            self.labels_ = np.array([-1])
            self.points_ = X
            return self
        else:
            if self.create_outliers:
                k_knn = self.tomato_params.get("k", np.min([len(X), 10]))
                self._knn_graph_ = self._get_knn_graph(X, k_knn)
                _labels_ = np.full(len(X), -2)  # placeholder for no label
                if self.density_type_.startswith("log"):
                    _weights_ = np.full(len(X), -np.inf)
                else:
                    _weights_ = np.zeros(len(X))
                non_outlier_ixs = self._get_non_outliers_ixs(X)
                _labels_[~non_outlier_ixs] = -1
                X, _X = X[non_outlier_ixs], X
            self._get_fitted_attributes(X, tomato_fit_params)
            self.n_outliers_ = 0
            if self.create_outliers:
                self.points_ = _X
                _labels_[non_outlier_ixs] = self.labels_
                self.labels_ = _labels_
                _weights_[non_outlier_ixs] = self.weights_
                self.weights_ = _weights_
                self.n_outliers_ = len(self.labels_[self.labels_ == -1])
            return self

    def predict(self, X, y=None):
        """Returns the labels assigned to the data points by the fitted
        AuToMATo instance. Only works if the instance has been fitted before.

        Args:
            X (numpy.ndarray of shape (n_samples, dim)): NumPy-array containing
                the point cloud data to be clustered. Must be of shape
                (n_samples, dim), where n_samples is the number of data points
                and dim is the dimensionality of the point cloud.
            y: Not used, present here for API consistency with scikit-learn.

        Returns:
            numpy.ndarray of shape (n_samples,): NumPy-array containing the
                labels assigned to the data points.
        """
        check_is_fitted(self, attributes="labels_")
        return self.labels_

    def fit_predict(self, X, y=None, tomato_fit_params=dict()):
        """Equivalent to calling fit() followed by predict().

        Args:
            X (numpy.ndarray of shape (n_samples, dim)): NumPy-array containing
                the point cloud data to be clustered. Must be of shape
                (n_samples, dim), where n_samples is the number of data points
                and dim is the dimensionality of the point cloud.
            y: Not used, present here for API consistency with scikit-learn.
            tomato_fit_params (dict, optional): A dictionary containing
                arguments that are passed to the fit()-method of the underlying
                instance of :class:`~gudhi.clustering.tomato.Tomato`.
                Defaults to `dict()`

        Returns:
            numpy.ndarray of shape (n_samples,): NumPy-array containing the
                labels assigned to the data points.
        """
        return self.fit(X, y=None).labels_

    def _get_knn_graph(self, X, k):
        list_of_vertex_nbrs = KNN(k=k).fit_transform(X)
        knn_edges = np.array([
            [vertex, vertex_nbr]
            for vertex, vertex_nbrs in enumerate(list_of_vertex_nbrs)
            for vertex_nbr in vertex_nbrs
        ])
        adj = np.zeros((len(X), len(X)), dtype=int)
        adj[knn_edges[:, 0], knn_edges[:, 1]] = 1
        np.fill_diagonal(adj, 0)
        return ig.Graph.Adjacency(adj, mode="directed")

    def _get_non_outliers_ixs(self, X):
        degrees = np.array(self._knn_graph_.degree(mode="all"))
        indegrees = np.array(self._knn_graph_.degree(mode="in"))
        return indegrees > self.ratio_outliers * degrees

    def _get_fitted_attributes(self, X, tomato_fit_params):
        if len(X) == 1:
            self.weights_ = np.array([np.inf])
            self._min_weight_ = np.min(self.weights_)
            self.max_weight_per_cc_ = np.array([np.inf])
            self._finite_pts_ = np.array([]).reshape(-1, 2)
            self._infinite_pts_ = np.array([]).reshape(-1, 2)
            self.diagram_ = np.concatenate([
                    self._finite_pts_,
                    self._infinite_pts_
                ])
            self._essential_pts_ = np.array([]).reshape(-1, 2)
            self.n_clusters_ = 1
        else:
            _Customato.fit(self, X, **self.tomato_fit_params)
            self._finite_pts_, self._infinite_pts_ = self._get_points()
            self.diagram_ = np.concatenate([
                    self._finite_pts_,
                    self._infinite_pts_
                ])
            self._essential_pts_ = self._cancel_noise_automato()
            self.n_clusters_ = np.max([
                len(self._essential_pts_),
                len(self.max_weight_per_cc_)
            ])

    def _get_points(self):
        def _sort_by_lifetimes(persistence):
            lifetimes = -np.diff(persistence, axis=1).reshape(-1,)
            return persistence[np.argsort(lifetimes)]
        finite_pts = self.diagram_
        infinite_pts = np.array([
            [infinite_birth, -np.inf]
            for infinite_birth in self.max_weight_per_cc_
        ])
        return _sort_by_lifetimes(finite_pts), _sort_by_lifetimes(infinite_pts)

    def _cancel_noise_automato(self):
        self._bootstrapper = BottleneckBootstrap(
            estimator=_Customato,
            alpha=self.alpha,
            n_bootstrap=self.n_bootstrap,
            parallelize=self.parallelize,
            estimator_params=self.tomato_params,
            random_state=self.random_state
        )
        self._bootstrapper.fit(
            X=self.points_,
            estimator_fit_params=self.tomato_fit_params
        )
        cleaned_pts = np.concatenate(
            self._bootstrapper.cleaned_persistence_
        )
        self.width_conf_band_ = self._bootstrapper.width_conf_band_
        return cleaned_pts

    def plot_diagram(
        self,
        to_scale=False,
        display_plot=False
    ):
        """Method to plot the persistence diagram of the AuToMATo instance
        resp. its underlying ToMATo instance.
        In the resulting plot, points are colored according to whether or not
        they correspond to clusters that survive until infinity in the
        superlevel filtration. Moreover, the corresponding cutoff line is
        displayed.

        Args:
            to_scale (bool, optional): Whether or not to use the same scale on
                both axes of the plot. Defaults to False.
            display_plot (bool, optional): Whether or not to call show() on
                the resulting Plotly figure, as opposed to just returning
                the figure. Defaults to False.

        Returns:
            :class:`plotly.graph_objs._figure.Figure`: A plot of the
                persistence diagram.

        """
        fig = plot_persistences(
            [[self._finite_pts_, self._infinite_pts_]],
            bandwidths=[-self.width_conf_band_],
            to_scale=to_scale,
            display_plot=display_plot
        )
        fig._data_objs[0].name = "y=x"
        fig._data_objs[1].name = "Finite points"
        fig._data_objs[2].name = "Infinite points"
        fig._data_objs[3].name = u"y=\u2212\u221E"
        fig._data_objs[4].name = "Bootstrap cutoff"
        fig._data_objs[5].name = "Finite points"
        fig._data_objs[6].name = "Infinite points"
        fig._data_objs[7].name = "y=x"
        fig._data_objs[8].name = u"y=\u2212\u221E"
        fig._data_objs[9].name = "Bootstrap cutoff"
        return fig
