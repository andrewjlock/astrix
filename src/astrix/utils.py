# pyright: reportAny=false

"""Utility functions for Astrix.

Should not be imported to core type modules to avoid circular dependencies.
"""

from .spatial.location import Point, Path
from ._backend_utils import (
    coerce_ns,
    BackendArg,
    Backend,
    resolve_backend,
    Array,
    np,
    warn_if_not_numpy,
)
from .time import TimeLike, TIME_INVARIANT


def dist(a: Point, b: Point, backend: BackendArg = None) -> Array:
    """Calculate the Euclidean distance between two points."""

    if backend is None:
        if a.backend == b.backend:
            backend = a.backend
    xp = resolve_backend(backend)

    if len(a) != len(b) and any([len(a) != 1, len(b) != 1]):
        raise ValueError(
            "To calculate distance point objects have be same length, or one singular."
        )

    d_arr = xp.linalg.norm(a.ecef - b.ecef, axis=1)
    return d_arr


def point_from_heading(
    start: Point,
    head: float,
    dist: float,
    time_new: TimeLike = TIME_INVARIANT,
    method: str = "haversine",
) -> Point:
    """Calculate a new point given a start point, heading (degrees) and distance (meters).
    If point has len > 1, uses the last point.

    Note: Applicable for NumPy backend only.

    """

    warn_if_not_numpy(start.backend, "point_from_heading")

    start = start[-1]
    a = np.deg2rad(head)
    lat1, lon1, alt = start.geodet.T
    r = np.linalg.norm(start.ecef)

    if method == "const":
        # Constant heading angle - rhumb line (inverse Gudermannian function)
        lat2_r = lat1 + (dist / r) * np.cos(a)
        d_psi = np.log(
            np.tan((np.pi / 4) + (lat2_r / 2)) / np.tan((np.pi / 4) + (lat1 / 2))
        )
        if d_psi < 1e-12:
            q = np.cos(lat1)
        else:
            q = (lat2_r - lat1) / d_psi
        lon2_r = lon1 + (dist / r) * np.sin(a) / q

    elif method == "haversine":
        # Based on Haversine formula (shortest ciruclar path)
        lat2_r = np.arcsin(
            np.sin(lat1) * np.cos(dist / r) + np.cos(lat1) * np.sin(dist / r) * np.cos(a)
        )
        lon2_r = lon1 + np.arctan2(
            np.sin(a) * np.sin(dist / r) * np.cos(lat1),
            np.cos(dist / r) - np.sin(lat1) * np.sin(lat2_r),
        )
    else:
        raise ValueError("Method must be 'const' or 'haversine'")

    lat2 = np.rad2deg(lat2_r)
    lon2 = np.rad2deg(lon2_r)
    point2 = Point.from_geodet(np.array([lat2, lon2, alt]), time=time_new, backend=np)
    return point2
