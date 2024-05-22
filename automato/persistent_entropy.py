import numpy as np
from sklearn.metrics import pairwise_distances
from copy import deepcopy


def _probabilities(lifetimes):
    """Scales lifetimes so that they sum to 1, while keeping their proportions.
    """
    return lifetimes / np.sum(lifetimes)


def _persistent_entropy(lifetimes):
    """Computes the persistent entropy of an array of lifetimes of generators.
    """
    return -np.dot(
        _probabilities(lifetimes),
        np.log(_probabilities(lifetimes))
    )


def _get_l_prime(lifetimes, i):
    """Computes the value of l_prime as described in Thm. 3 in [1]
        and Thm. 1 in [2].
    """
    return (np.sum(lifetimes[i+1:])
            / np.exp(_persistent_entropy(lifetimes[i+1:])))


def _neutralize(lifetimes, i):
    """Performs the neutralization operation on an array of lifetimes as
        described in Thm. 3 in [1] and Thm. 1 in [2].
    """
    if i < 0:
        return lifetimes
    elif i >= len(lifetimes) - 2:
        return lifetimes[-1] * np.ones(lifetimes.shape[0])
    l_prime = _get_l_prime(lifetimes, i)
    lifetimes_new = np.copy(lifetimes).astype(float)
    lifetimes_new[:i+1] = l_prime
    return lifetimes_new


def _probability_max_lifetime(lifetimes, i):
    """Computes the probability of the longest bar after the neutralization
        operation. Note: this implicitly assumes that the input array of
        lifetimes is sorted in descending order.
    """
    lifetimes = np.roll(lifetimes, -1)
    return _probabilities(_neutralize(lifetimes, i))[-1]


def _get_Q(lifetimes):
    """Computes the value of Q as described in Thm. 4 of [1]. Note, however,
        that, in contrast to [1], the function at hand computes this value
        simply by using the numpy.argmin function.
    """
    # Uncomment the following five lines to use the version from the paper [1]:
    # def _get_alpha(lifetimes):
    #     return np.min(lifetimes) / np.max(lifetimes)
    # n, alpha = len(lifetimes), get_alpha(lifetimes)
    # Q = (n / (1 - alpha)**2) * (alpha * (alpha - np.log(alpha) - 1)) + 1
    # return int(Q)
    n = len(lifetimes)
    max_lifetime, min_lifetime = np.max(lifetimes), np.min(lifetimes)
    entropies = np.array([
        _persistent_entropy(
            np.concatenate([
                max_lifetime * np.ones(i),
                min_lifetime * np.ones(n-i)
            ])
        )
        for i in range(1, n)
    ])
    return np.argmin(entropies) + 1


def _ratios(array):
    """Given a 1-dim. array, computes the array of consecutive ratios.
    """
    return array[1:] / array[:-1]


def cancel_noise(
        persistence,
        max_lifetime=None,
        X=None,
        homology_dimensions=None,
        use_advanced=False,
        beta=-1
):
    """Function to separate topological features from noise in a persistence
    diagram. This is an implementation of the algorithms introduced by Atienza
    et al. in Procedure 5 in [1] and §5 in [2]. Note that both algorithms are
    implemented with minor changes from their original definitions: the
    algorithm from [1] as implemented here contains an additional stopping
    condition in the while loop in order to guarantee termination, while the
    algorithm from [2] is adjusted to guarantee to return a list of
    consecutive lifetimes when ordered in decreasing order.

    Args:
        persistence (list[numpy.ndarray of shape (n_generators, 2)]): The
            persistence data that the function is to be applied to. Must be a
            list of NumPy-arrays of shape (n_generators, 2), where the i-th
            entry of the list is an array containing the birth and death times
            of the homological generators in dimension i-1. In particular, the
            list must start with 0-dimensional homology and contain information
            from consecutive homological dimensions.
        max_lifetime (float, optional): The value at which to cap lifetimes of
            homological generators, in order to ensure that all generators
            have a finite lifetime as the algorithms fail otherwise.
            If max_lifetime is None, a value of max_lifetime will be extracted
            from the parameter X, if present; see below.
            If X is None, too, then max_lifetime will be set as the maximal
            death time among all generators of persistence (which should be
            finite). Defaults to None.
        X (numpy.ndarray of shape (n_samples, dim), optional): NumPy-array
            containing the point cloud data underlying the persistence
            information, if available. If `max_lifetime` is not None, this is
            ignored, and otherwise used only to extract a finite value of
            `max_lifetime`, as described in Proposition 2 in [1].
            Defaults to None.
        homology_dimensions (list[int], optional): A list of the homological
            dimensions by which to filter the persistence data. If None, then
            all homological dimensions present are taken into account.
            Defaults to None.
        use_advanced (bool, optional): Whether or not to use the "advanced"
            version of cancelling noise in the persistence diagram, i.e.
            the algorithm described in [2]. Defaults to False.
        beta (float, optional): The exponent to which to raise the thresholds
            in the "non-advanced" version of separating noise from features in
            a persistence diagram. Must be a positive number or -1. If set to
            -1, then the base-2-log will be applied to the threshold (without
            raising them to any power first). Ignored if `use_advanced` is set
            to 0. Defaults to -1.

    Returns:
        tuple(numpy.ndarray of shape (n_features, 2)): A tuple whose first
            entry is a NumPy-array containing the birth and death times of the
            homological generators that are considered topological features. If
            `use_advanced` is True, the second entry is None. Otherwise, it is
            a NumPy-array of shape (n_generators, ) containing the relative
            entropies of the homological generators, where `n_generators` is
            their number.

    References:
        [1]: Atienza, N., Gonzalez-Diaz, R. & Rucco, M. Persistent entropy for
            separating topological features from noise in vietoris-rips
            complexes. J Intell Inf Syst 52, 637–655 (2019).
            https://doi.org/10.1007/s10844-017-0473-4
        [2]: Atienza, N., Gonzalez-Diaz, R., Rucco, M. (2016). Separating
            Topological Noise from Features Using Persistent Entropy.
            In: Milazzo, P., Varró, D., Wimmer, M. (eds) Software Technologies:
            Applications and Foundations. STAF 2016. Lecture Notes in Computer
            Science(), vol 9946. Springer, Cham.
            https://doi.org/10.1007/978-3-319-50230-4_1
    """
    if not homology_dimensions:
        homology_dimensions = list(range(len(persistence)))
    pers_data = deepcopy(persistence)
    # Filter by homology dimensions and concatenate:
    pers_data = np.concatenate([pers_data[dim] for dim in homology_dimensions])
    if len(pers_data) <= 1:
        return pers_data[:, :2], np.ones(len(pers_data))
    if max_lifetime is None:
        if X is not None:
            dm = pairwise_distances(X)
            max_lifetime = np.min(np.max(dm, axis=1))
        else:
            max_lifetime = np.max(persistence[0][:, 1])
        if not np.isfinite(max_lifetime):
            raise ValueError(
                "Could not determine a finite value for "
                "`max_lifetime` from the inputs."
                )
    # Cap lifetimes at max_lifetime:
    pers_data = np.where(pers_data > max_lifetime, max_lifetime, pers_data)
    # Add column with lifetimes:
    pers_data = np.concatenate([pers_data, np.diff(pers_data, axis=1)], axis=1)
    # Keep only bars with non-zero lifespan:
    pers_data = pers_data[np.where(pers_data[:, -1] > 0)]
    # Sort pers_data by lifetimes in descending order:
    pers_data = pers_data[np.argsort(pers_data[:, -1])][::-1]
    if len(pers_data) <= 1:
        return pers_data[:, :2], np.ones(len(pers_data))
    lifetimes = pers_data[:, -1]
    r = lifetimes[-1]
    if not use_advanced:
        n = len(lifetimes)
        entropies = np.array([
            _persistent_entropy(_neutralize(lifetimes, i))
            for i in range(-1, n)
        ])
        # Catch case where all lifetimes are equal:
        if np.log(n) == entropies[0]:
            rel_entropies = np.zeros(len(lifetimes))
            rel_entropies[0] = 1
        else:
            rel_entropies = np.diff(entropies) / (np.log(n) - entropies[0])
        lower_bounds = np.arange(1, n+1) / n
        if beta > 0 and beta != 1:
            lower_bounds = lower_bounds ** beta
        if beta == -1:
            lower_bounds = np.log2(lower_bounds + 1)
        feature_ixs = rel_entropies > lower_bounds
        # Uncomment the following for the method from paper [1], which
        # potentially yields a list of non-consecutive features:
        # return pers_data[feature_ixs][:, :2], rel_entropies
        if np.max(feature_ixs) == 0:
            m = 0
        else:
            m = n - np.argmax(feature_ixs[::-1])
        return pers_data[:m][:, :2], rel_entropies
    else:
        Q, Q_old = 1, 1
        m, m_old = len(lifetimes), np.inf
        # Second condition is to ensure termination
        while Q_old < m and m < m_old:
            Q, Q_old = _get_Q(lifetimes), Q
            C_values = _ratios(
                np.array([
                    _probability_max_lifetime(lifetimes, ix-1)
                    for ix in range(m+1)
                ])
            )
            m, m_old = np.argmax(C_values < 1) + 1, m
            lifetimes = np.concatenate([lifetimes[:m], [r]])
        return pers_data[:m, :2], None
