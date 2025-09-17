# pyright: reportAny=false

import pyproj

from ._backend_utils import Array, np, get_backend, warn_if_not_numpy


def ensure_1d(x: Array | float | list[float]) -> Array:
    """Ensure the input array is 1-dimensional.
    Scalars are converted to shape (1,).
    """

    xp = get_backend(x)
    x_arr = xp.asarray(x)
    if x_arr.ndim == 0:
        x_arr = xp.reshape(x_arr, (1,))
    elif x_arr.ndim > 1:
        raise ValueError("Input array must be 1-dimensional or scalar.")
    return x_arr


def ensure_2d(
    x: Array | float | list[float] | list[list[float]], n: int | None = None
) -> Array:
    """Ensure the input array is 2-dimensional.
    If n is given, ensure the second dimensionn has size n.
    """

    xp = get_backend(x)
    x_arr = xp.asarray(x)
    if x_arr.ndim == 0:
        x_arr = xp.reshape(x_arr, (1, 1))
    elif x_arr.ndim == 1:
        x_arr = xp.reshape(x_arr, (1, -1))
    elif x_arr.ndim > 2:
        raise ValueError("Input array must be 2-dimensional or less.")
    if n is not None and x_arr.shape[1] != n:
        raise ValueError(f"Input array must have shape (m, {n}), found {x_arr.shape}.")
    return x_arr


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
