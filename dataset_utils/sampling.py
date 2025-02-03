import numpy as np
from scipy.optimize import root_scalar  # type: ignore


def sphere_sampling(n=1000, r=1.0, noise=0.0):
    """Uniformly and randomly samples points from a round
    sphere centered at the origin of 3-space. Inspired by and building upon
    code from the `giotto-tda` package [1].

    Args:
        n (int, optional): The number of points to be sampled. Must
            be a positive integer. Defaults to 1000.
        r (float, optional): The radius of the sphere to be sampled from. Must
            be a positive number. Defaults to 1.0.
        noise (float, optional): The noise of the sampling, which is
            introduced by adding Gaussian noise around each data point. Must
            be a non-negative number. Defaults to 0.0.

    Returns:
        numpy.ndarray of shape (n, 3): NumPy-array containing the sampled
            points.

    References:
        [1]: giotto-tda: A Topological Data Analysis Toolkit for Machine
            Learning and Data Exploration, Tauzin et al, J. Mach. Learn. Res.
            22.39 (2021): 1-6.
    """
    # Validate `n` parameter
    if not (
        isinstance(n, int) and
        n > 0 and
        n < float("inf")
    ):
        raise ValueError(
            "The `n` parameter must be a finite positive integer."
        )
    # Validate `r` parameter
    if not (
        isinstance(r, (int, float)) and
        r > 0 and
        r < float("inf")
    ):
        raise ValueError(
            "The `r` parameter must be a finite positive number."
        )
    # Validate `noise` parameter
    if not (
        isinstance(noise, (int, float)) and
        noise >= 0 and
        noise < float("inf")
    ):
        raise ValueError(
            "The `noise` parameter must be a finite non-negative number."
        )

    def parametrization(theta, phi):
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        return np.array([x, y, z])

    def U_phi_inverse(y): return np.arccos(1 - 2 * y)
    points = np.array([
        parametrization(
            np.random.uniform(low=0, high=2 * np.pi),
            U_phi_inverse(np.random.uniform())
        )
        for _ in range(n)
    ])
    if noise:
        points += noise * np.random.randn(n, 3)
    return points


def torus_sampling(n=1000, R=3.0, r=1.0, noise=0.0):
    """Uniformly and randomly samples points from a torus
    centered at the origin of 3-space and lying in it
    horizontally. Inspired by and building upon code from the `giotto-tda`
    package [1].

    Args:
        n (int, optional): The number of points to be sampled. Must
            be a positive integer. Defaults to 1000.
        R (float, optional): The inner radius of the torus to be sampled from,
            that is, the radius of the circle along which the "tube" follows.
            Must be a positive number. Defaults to 3.0.
        r (float, optional): The outer radius of the torus to be sampled from,
            that is, the radius of the "tube". Must be a positive number.
            Defaults to 1.0.
        noise (float, optional): The noise of the sampling, which is
            introduced by adding Gaussian noise around each data point. Must
            be a non-negative number. Defaults to 0.0.

    Returns:
        numpy.ndarray of shape (n, 3): NumPy-array containing the sampled
            points.

    References:
        [1]: giotto-tda: A Topological Data Analysis Toolkit for Machine
            Learning and Data Exploration, Tauzin et al, J. Mach. Learn. Res.
            22.39 (2021): 1-6.
    """
    # Validate `n` parameter
    if not (
        isinstance(n, int) and
        n > 0 and
        n < float("inf")
    ):
        raise ValueError(
            "The `n` parameter must be a finite positive integer."
        )
    # Validate `R` parameter
    if not (
        isinstance(R, (int, float)) and
        R > 0 and
        R < float("inf")
    ):
        raise ValueError(
            "The `R` parameter must be a finite positive number."
        )
    # Validate `r` parameter
    if not (
        isinstance(r, (int, float)) and
        r > 0 and
        r < float("inf")
    ):
        raise ValueError(
            "The `r` parameter must be a finite positive number."
        )
    # Validate `noise` parameter
    if not (
        isinstance(noise, (int, float)) and
        noise >= 0 and
        noise < float("inf")
    ):
        raise ValueError(
            "The `noise` parameter must be a finite non-negative number."
        )

    def parametrization(theta, phi):
        x = np.cos(theta) * (R + r * np.cos(phi))
        y = np.sin(theta) * (R + r * np.cos(phi))
        z = r * np.sin(phi)
        return np.array([x, y, z])

    def U_phi(x): return (0.5 / np.pi) * (x + r * np.sin(x) / R)

    def U_phi_inverse(y):
        def U_phi_shifted(x): return U_phi(x) - y
        sol = root_scalar(U_phi_shifted, bracket=[0, 2 * np.pi])
        return sol.root
    points = np.array([
        parametrization(
            np.random.uniform(low=0, high=2 * np.pi),
            U_phi_inverse(np.random.uniform())
        )
        for _ in range(n)
    ])
    if noise:
        points += noise * np.random.randn(n, 3)
    return points


def circle_sampling(n=1000, r=1.0, noise=0.0):
    """Uniformly and randomly samples points from a circle
    centered at the origin of 3-space and lying in it
    horizontally. Inspired by and building upon code from the `giotto-tda`
    package [1].

    Args:
        n (int, optional): The number of points to be sampled. Must
            be a positive integer. Defaults to 1000.
        r (float, optional): The radius of the circle to be sampled from. Must
            be a positive number. Defaults to 1.0.
        noise (float, optional): The noise of the sampling, which is
            introduced by adding Gaussian noise around each data point. Must
            be a non-negative number. Defaults to 0.0.

    Returns:
        numpy.ndarray of shape (n, 3): NumPy-array containing the sampled
            points.

    References:
        [1]: giotto-tda: A Topological Data Analysis Toolkit for Machine
            Learning and Data Exploration, Tauzin et al, J. Mach. Learn. Res.
            22.39 (2021): 1-6.
    """
    # Validate `n` parameter
    if not (
        isinstance(n, int) and
        n > 0 and
        n < float("inf")
    ):
        raise ValueError(
            "The `n` parameter must be a finite positive integer."
        )
    # Validate `r` parameter
    if not (
        isinstance(r, (int, float)) and
        r > 0 and
        r < float("inf")
    ):
        raise ValueError(
            "The `r` parameter must be a finite positive number."
        )
    # Validate `noise` parameter
    if not (
        isinstance(noise, (int, float)) and
        noise >= 0 and
        noise < float("inf")
    ):
        raise ValueError(
            "The `noise` parameter must be a finite non-negative number."
        )

    def parametrization(theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0
        return np.array([x, y, z])
    points = np.array([
        parametrization(
            np.random.uniform(low=0, high=2 * np.pi)
        )
        for _ in range(n)
    ])
    if noise:
        points += noise * np.random.randn(n, 3)
    return points
