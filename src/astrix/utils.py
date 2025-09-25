# pyright: reportAny=false

import pyproj

from ._backend_utils import (
    Array,
    ArrayNS,
    np,
    get_backend,
    warn_if_not_numpy,
    backend_jit,
    BackendArg,
    Backend,
    coerce_ns,
)

from scipy.spatial.transform import Rotation


def ensure_1d(x: Array | float | list[float], backend: Backend = None) -> Array:
    """Ensure the input array is 1-dimensional.
    Scalars are converted to shape (1,).
    """

    xp = coerce_ns(backend)
    x_arr = xp.asarray(x, dtype=xp.float64)
    if x_arr.ndim == 0:
        x_arr = xp.reshape(x_arr, (1,))
    elif x_arr.ndim > 1:
        raise ValueError("Input array must be 1-dimensional or scalar.")
    return x_arr


def ensure_2d(
    x: Array | float | list[float] | list[list[float]],
    n: int | None = None,
    backend: Backend = None,
) -> Array:
    """Ensure the input array is 2-dimensional.
    If n is given, ensure the second dimensionn has size n.
    """

    xp = coerce_ns(backend)
    x_arr = xp.asarray(x, dtype=xp.float64)
    if x_arr.ndim == 0:
        x_arr = xp.reshape(x_arr, (1, 1))
    elif x_arr.ndim == 1:
        x_arr = xp.reshape(x_arr, (1, -1))
    elif x_arr.ndim > 2:
        raise ValueError("Input array must be 2-dimensional or less.")
    if n is not None and x_arr.shape[1] != n:
        raise ValueError(f"Input array must have shape (m, {n}), found {x_arr.shape}.")
    return x_arr


def is_increasing(x: Array, backend: Backend = None) -> bool:
    """Check if the input array is strictly increasing along the first axis."""

    xp = coerce_ns(backend)
    x_arr = xp.asarray(x)
    if x_arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional.")
    return xp.all(x_arr[1:] > x_arr[:-1])


t_ecef2geodet = pyproj.Transformer.from_crs("epsg:4978", "epsg:4979")
t_geodet2ecef = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978")


def ecef2geodet(ecef: Array) -> np.ndarray:
    """
    Use pyproj to convert from WGS84 coordinates to geodetic
    (Cartesian Earth Centered Earth Fixed (ECEF) to lat, long, alt)

    Args:
        ecef (np.ndarray): 3xn array of ECEF x,y,z points in [m]

    Returns:
        np.ndarray: 3xn array of long, lat, alt [m] points

    """

    warn_if_not_numpy(ecef)
    ecef_np = np.array(ecef)
    geodet = np.array(
        t_ecef2geodet.transform(
            ecef_np[:, 0], ecef_np[:, 1], ecef_np[:, 2], radians=False
        )
    ).T
    return geodet


def geodet2ecef(geodet: Array) -> np.ndarray:
    """
    Use pyproj to convert from WGS84 coordinates to x,y,z points
    (long, lat, alt to Cartesian Earth Centered Earth Fixed(ECEF))

    Args:
        geodet (np.ndarray): 3xn array of long, lat, alt [m] points

    Returns:
        np.ndarray: 3xn array of ECEF x,y,z points in [m]

    """

    warn_if_not_numpy(geodet)
    geodet_np = np.array(geodet)
    ecef = np.array(
        t_geodet2ecef.transform(
            geodet_np[:, 0], geodet_np[:, 1], geodet_np[:, 2], radians=False
        )
    ).T

    return ecef


@backend_jit(["inverse", "xp"])
def apply_rot(r: Rotation, v: Array, inverse: bool = False, xp: ArrayNS = np) -> Array:
    """Apply a scipy Rotation to a set of vectors.

    Args:
        v (Array): Array of shape (m, 3) of m 3D vectors.
        r (Rotation): A scipy Rotation object.
        backend (BackendArg, optional): Backend to use. Defaults to None.

    Returns:
        Array: Array of shape (m, 3) of rotated vectors.

    Notes:
        - If inverse is True, apply the inverse rotation.
        - No checks performed on input shapes or backends
    """


    if inverse:
        v_rot = xp.einsum("ijk,ik->ij", r.as_matrix().reshape(-1, 3, 3).mT, v)
    else:
        v_rot = xp.einsum("ijk,ik->ij", r.as_matrix().reshape(-1, 3, 3), v)
    return v_rot


@backend_jit(["backend"])
def sort_by_time(
    times: Array,
    data: Array,
    backend: Backend = None,
) -> tuple[Array, Array]:
    """Sort data by increasing time.

    Args:
        times (Array): 1D array of times, len N.
        data (Array): NxM array of data to sort by time.
        backend (BackendArg, optional): Backend to use. Defaults to None.

    Returns:
        tuple[Array, Array]: Sorted times and data.
    """

    xp = coerce_ns(backend)
    times_1d = ensure_1d(times, backend=xp)
    if data.shape[0] != times_1d.shape[0]:
        raise ValueError(
            f"Data first dimension must match times length, found {data.shape} and {times_1d.shape}."
        )
    sort_idx = xp.argsort(times_1d)
    return times_1d[sort_idx], data[sort_idx]

