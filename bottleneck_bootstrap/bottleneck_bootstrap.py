from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import numpy as np
from copy import deepcopy
from gudhi import bottleneck_distance
from functools import partial
from joblib import delayed, Parallel
from persistence_plotting import plot_persistences


class BottleneckBootstrap(BaseEstimator):
    """This transformer is desgined to perform a bootstrapping procedure on a
    persistence diagram in order to separate topological features from noise,
    as explained in §6 of [1].

    Parameters:
        estimator: An estimator computing persistent homology that
            implements a `fit`-method. The estimator must be such that any
            fitted instance of it has an attribute named `persistence_`
            containing data of a persistence diagram. The format of this
            data must be a list of NumPy-arrays of shape (n_generators, 2),
            where the i-th entry of the list is an array containing the
            birth and death times of the homological generators in
            dimension i-1. In particular, the list must start with
            0-dimensional homology and contains information from
            consecutive homological dimensions.
        estimator_params (dict, optional): A dictionary containing
            arguments that are passed to the underlying estimator instance.
            Defaults to dict().
        alpha (float, optional): Confidence level of the bootstrap. Writing to
            it after fitting automatically adjusts `width_conf_band_`,
            `cleaned_persistence_` and `n_top_features_` accordingly (without
            performing the entire subsampling process anew). Defaults to 0.05.
        n_bootstrap (int, optional): Number of bootstrap samples to use.
            Defaults to 1000.
        parallelize (int, optional): Whether or not to parallelize the
            computation of the estimator on the bootstrap samples by using
            the joblib-library. 0 means no parallelization, a positive
            integer i means parallelization using i processors, while a
            negative integer -i means parallelization using all but (i-1)
            processors so that e.g. -1 means using all processors.
            Defaults to -1.
        random_state (int, optional): If not None, this number will be
            used as random seed in the bootstrap procedure, allowing for
            reproducibility of results. Defaults to None.

    Attributes:
        points_ (numpy.ndarray of shape (n_samples, dim)): NumPy-array
            containing the point cloud data that the BootstrapPersistence
            instance was fitted on.
        persistence_ (list[numpy.ndarray of shape (n_generators, 2)]): The
            persistence data of the underlying point cloud. The format of this
            data is a list of NumPy-arrays of shape (n_generators, 2), where
            the i-th entry of the list is an array containing the birth and
            death times of the homological generators in dimension i-1. In
            particular, the list starts with 0-dimensional homology and
            contains information from consecutive homological dimensions.
        width_conf_band_ (float): The width of a confidence band found by
            fitting an instance to a point cloud resp. its persistence. That
            is, al generators whose lifetime is less than this value are to be
            interpreted as noise.
        cleaned_persistence_ (list[numpy.ndarray of shape (n_generators, 2)]):
            The homological generators that are considered topological
            features. The format of this data is the same as that of
            `persistence_`.
        n_top_features_ (int): The number of homological generators that are
            considered topological features.

    References:
        [1]: Chazal, F., Fasy, B., Lecci, F., Michel, B., Rinaldo, A., &
            Wasserman, L. (2018). Robust Topological Inference: Distance To a
            Measure and Kernel Distance. Journal of Machine Learning Research,
            18(159), 1–40. http://jmlr.org/papers/v18/15-484.html
    """
    def __init__(
        self,
        estimator,
        estimator_params=dict(),
        alpha=0.05,
        n_bootstrap=1000,
        parallelize=-1,
        random_state=None
    ):
        self.estimator = estimator
        self.estimator_params = estimator_params
        self._alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.parallelize = parallelize
        self.random_state = random_state

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if alpha == self._alpha:
            return
        else:
            self._alpha = alpha
            t_value_new = self._t_value_fcn(alpha)
            self.width_conf_band_ = (2*t_value_new)/np.sqrt(len(self.points_))
            self.cleaned_persistence_ = self._clean_persistence(
                self.persistence_,
                self._lifetimes_,
                self.width_conf_band_
            )
            self.n_top_features_ = len(
                np.concatenate(self.cleaned_persistence_)
            )
            return

    def fit(
        self,
        X,
        y=None,
        persistence=None,
        estimator_fit_params=dict()
    ):
        """Method that fits a BootstrapPersistence instance to the point cloud
        whose persistence the bootstrapping procedure is to be applied to.
        This sets the attributes `width_conf_band_`, `cleaned_persistence_` and
        `n_top_features_`.

        Args:
            X (numpy.ndarray of shape (n_samples, dim)): NumPy-array
                containing the point cloud data underlying the persistence
                information.
            y: Not used, present here for API consistency with scikit-learn.
            persistence (list[numpy.ndarray], optional):
                If available, the persistence data of X that the bootstrap is
                to be applied to. The format of this data must be a list of
                NumPy-arrays of shape (n_generators, 2), where the i-th entry
                of the list is an array containing the birth and death times
                of the homological generators in dimension i-1. In particular,
                the list must start with 0-dimensional homology and contains
                information from consecutive homological dimensions.
                Defaults to None.
            estimator_fit_params (dict, optional): A dictionary containing
                arguments that are passed to the fit()-method of the underlying
                estimator instance. Defaults to dict().

        Returns:
            :class:`bootstrap_persistence.BootstrapPersistence`: Fitted
                instance of the BootstrapPersistence clusterer.
        """
        self.estimator_fit_params = estimator_fit_params
        self.points_ = X
        if persistence is None:
            est = self.estimator(**self.estimator_params)
            est.fit(self.points_, **self.estimator_fit_params)
            self.persistence_ = est.persistence_
        else:
            self.persistence_ = persistence
        self._lifetimes_ = [
            np.diff(dim, axis=1).reshape(-1,)
            for dim in self.persistence_
        ]
        persistence_flattened = np.concatenate(deepcopy(self.persistence_))
        # Add column with lifetimes
        lifetimes_flattened = np.concatenate(self._lifetimes_)
        persistence_flattened = np.concatenate([
            persistence_flattened, lifetimes_flattened.reshape(-1, 1)
        ], axis=1)
        # Drop points at infinity
        persistence_flattened = persistence_flattened[
            np.isfinite(persistence_flattened[:, -1])
        ]
        self._t_value_fcn = get_bootstrap_t_value(
            X=self.points_,
            statistic=partial(
                self._bottleneck_distance_to_reference,
                n=len(self.points_),
                estimator=self.estimator,
                estimator_params=self.estimator_params,
                estimator_fit_params=self.estimator_fit_params,
                dgm_ref=persistence_flattened[:, :2]
            ),
            n_bootstrap=self.n_bootstrap,
            parallelize=self.parallelize,
            random_state=self.random_state
        )
        t_value = self._t_value_fcn(self.alpha)
        self.width_conf_band_ = (2 * t_value) / np.sqrt(len(self.points_))
        self.cleaned_persistence_ = self._clean_persistence(
            self.persistence_,
            self._lifetimes_,
            self.width_conf_band_
        )
        self.n_top_features_ = len(
            np.concatenate(self.cleaned_persistence_)
        )
        return self

    @staticmethod
    def _bottleneck_distance_to_reference(
        subsample,
        n,
        estimator,
        estimator_params,
        estimator_fit_params,
        dgm_ref
    ):
        est = estimator(**estimator_params)
        est.fit(subsample, **estimator_fit_params)
        dgm = np.concatenate(est.persistence_)
        # Drop points at infinity
        dgm = dgm[np.isfinite(np.diff(dgm, axis=1).reshape(-1,))]
        return np.sqrt(n)*bottleneck_distance(dgm, dgm_ref)

    @staticmethod
    def _clean_persistence(persistence, lifetimes, width_conf_band_):
        return [
            dim[lifetimes[i] >= width_conf_band_]
            for i, dim in enumerate(persistence)
        ]

    def plot_persistence(
            self,
            without_infty=False,
            with_band=True,
            to_scale=False,
            marker_size=5.0,
            display_plot=False,
            plotly_params=None
    ):
        """Method to plot the persistence diagram of the point cloud
        underlying a fitted instance of BottleneckBootstrap.

        Args:
            without_infty (bool, optional): Whether or not to plot a horizontal
                line corresponding to generators dying at infinity.
                Defaults to False.
            with_band (bool, optional): Whether or not to plot a line at
                distance `width_conf_band_` from the diagonal, representing
                the confidence band found by bootstrapping. Defaults to True.
            to_scale (bool, optional): Whether or not to use the same scale
                across all axes of the plot. Defaults to False.
            marker_size (float, optional): The size of the markers used to
                plot homological generators. Defaults to 5.0.
            display_plot (bool, optional): Whether or not to call show() on
                the resulting Plotly figure, as opposed to just returning the
                figure. Defaults to False.
            plotly_params (_type_, optional):  Custom parameters to configure
                the plotly figure. Allowed keys are ``"trace"`` and
                ``"layout"``, and the corresponding values should be
                dictionaries containing keyword arguments as would be fed to
                the :meth:`update_traces` and :meth:`update_layout` methods of
                :class:`plotly.graph_objects.Figure`. Defaults to None.

        Returns:
            :class:`plotly.graph_objs._figure.Figure`: Figure containing the
                plotted persistence diagrams.
        """
        check_is_fitted(self, attributes="width_conf_band_")
        if with_band:
            bandwidth = self.width_conf_band_
        else:
            bandwidth = None
        return plot_persistences(
            [self.persistence_],
            without_infty=without_infty,
            bandwidths=[bandwidth],
            to_scale=to_scale,
            marker_size=marker_size,
            display_plot=display_plot,
            plotly_params=plotly_params
        )


def get_bootstrap_t_value(
    X,
    statistic,
    n_bootstrap=1000,
    parallelize=-1,
    random_state=None
):
    """Function performing the bootstrap on given data.

    Args:
        X (numpy.ndarray of shape (n_samples, dim)): NumPy-array containing
            the point cloud data to perform the bootstrap on. Must be of shape
            (n_samples, dim), where n_samples is the number of data points and
            dim is the dimensionality of the point cloud.
        statistic (Callable): The bootstrapping statistic, i.e. a callable
            that is defined for X and each subsample and returns a real number.
        n_bootstrap (int, optional): Number of bootstrap samples to use.
            Defaults to 1000.
        parallelize (int, optional): Whether or not to parallelize the
            computation of the estimator on the bootstrap samples by using
            the joblib-library. 0 means no parallelization, a positive
            integer i means parallelization using i processors, while a
            negative integer -i means parallelization using all but (i-1)
            processors so that e.g. -1 means using all processors.
            Defaults to -1.
        random_state (int, optional): If not None, this number will be used as
            random seed in the bootstrap procedure, allowing for
            reproducibility of results. Defaults to None.

    Returns:
        Callable: A callable that represents the inverse of the function
            F(t), where F(t) is the approximated CDF of the test statistic.
    """
    bootstrap_samples = X[
        np.random.RandomState(random_state).choice(
            a=len(X),
            size=(n_bootstrap, len(X)),
            replace=True
        )
    ]
    if parallelize != 0:
        theta_stars = np.array(Parallel(parallelize)(
            delayed(statistic)(subsample)
            for subsample in bootstrap_samples
        ))
    else:
        theta_stars = np.array([
            statistic(subsample)
            for subsample in bootstrap_samples
        ])
    theta_stars_sorted = np.sort(theta_stars)
    return partial(
        t_value_fcn,
        n_bootstrap=n_bootstrap,
        theta_stars_sorted=theta_stars_sorted
    )


def t_value_fcn(alpha, n_bootstrap, theta_stars_sorted):
    if alpha == 0:
        return np.inf
    elif alpha == 1:
        return 0
    else:
        k_hat = int(np.ceil(n_bootstrap*(1-alpha)))
        return theta_stars_sorted[k_hat-1]
