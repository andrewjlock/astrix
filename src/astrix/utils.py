# pyright: reportAny=false

"""Utility functions for Astrix.

Should not be imported to core type modules to avoid circular dependencies.
"""

from .spatial.location import Point, Path, Location
from ._backend_utils import (
    coerce_ns,
    BackendArg,
    Backend,
    resolve_backend,
    Array,
    np,
    warn_if_not_numpy,
)
from .time import TimeLike, TIME_INVARIANT, Time
from .functs import ned_rotation
from scipy.spatial.transform import Rotation


def dist(a: Point, b: Point, backend: BackendArg = None) -> Array:
    """Calculate the Euclidean distance between two points."""

    if backend is None:
        if a.backend == b.backend:
            backend = a.backend
    xp = resolve_backend(backend)

    if len(a) != len(b) and all([len(a) != 1, len(b) != 1]):
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
            np.sin(lat1) * np.cos(dist / r)
            + np.cos(lat1) * np.sin(dist / r) * np.cos(a)
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


def defeature_path(path: Path, tol: float = 50.0) -> Path:
    """Defeature a path by removing points that are within tol meters of a straight line
    between their neighbors. The first and last points are always kept. Useful for plotting.

    Args:
        path: Path
            The path to defeature.
        tol: float
            The tolerance in meters. Points that are within this distance of a straight line
            between their neighbors will be removed. Default is 50 meters.

    Returns:
        Path
            A new path with fewer points.
    """

    ecef = path.points.ecef
    n = len(ecef)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e <= s + 1:
            continue
        a, b = ecef[s], ecef[e]
        ab = b - a
        L2 = np.dot(ab, ab)
        P = ecef[s + 1 : e]
        if L2 == 0:
            d2 = np.sum((P - a) ** 2, axis=1)
        else:
            t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d2 = np.sum((P - proj) ** 2, axis=1)  # point-to-segment distance^2
        i = np.argmax(d2)
        if d2[i] > tol * tol:
            m = s + 1 + i
            keep[m] = True
            stack.extend([(s, m), (m, e)])  # pyright: ignore

    new_time = Time(path.time.unix[keep])
    points = Point(ecef[keep], time=new_time, backend=path.points.backend)
    new_path = Path(points)
    return new_path


def ground_path(path: Path, alt: float = 0.0) -> Path:
    """Return a new path with all points at the specified altitude (default 0 m).

    Args:
        path: Path
            The path to ground.
        alt: float
            The altitude in meters. Default is 0 m.
    Returns:
        Path
            A new path with all points at the specified altitude.
    """

    geodet = path.points.geodet
    new_geodet = np.c_[geodet[:, 0], geodet[:, 1], np.full(len(geodet), alt)]
    new_points = Point.from_geodet(
        new_geodet, time=path.time, backend=path.points.backend
    )
    new_path = Path(new_points)
    return new_path


def get_ned_rotation(loc: Location[TimeLike], backend: BackendArg = None) -> Rotation:
    """Calculate the rotation matrix from ECEF to NED frame at the given location.

    Args:
        loc: Location
            The location to calculate the NED rotation matrix for.
        backend: BackendArg
            The backend to use. If None, uses the backend of the location.
    Returns:
        Rotation
            The rotation matrix from ECEF to NED frame at the given location.
    """

    if backend is None:
        backend = loc.backend
    xp = resolve_backend(backend)
    rot = ned_rotation(loc.geodet, xp=xp)
    return rot


def solve_wahba(
    v: Array, w: Array, weights: Array | None = None
) -> tuple[Array, Array]:
    """Calculate the rotation matrix R which transforms vectors w to v.
    It is posed as Wahba's problem
    https://en.wikipedia.org/wiki/Wahba%27s_problem

    Parameters
    ----------
    v : nx3 dim numpy array
        vectors in the first coordiante frame
    w : nx3 dim numpy array
        vectors in the second coordinate frame
    weights : nx0 dim numpy array, optional
        weights associated with each vector. Otherwise assumed even weights

    Returns
    -------
    R : 3x3 numpy array
        The rotation matrix such that w ~ Rv


    Note these functions are very important for real-time evaluation.
    They need robust functional testing and good coding practices.
    """

    warn_if_not_numpy(v, "solve_wahba")

    # Check data
    if v.shape[0] != w.shape[0]:
        raise ValueError("Mismatch number of vectors in rotation estimation")
    if weights is None:
        a = np.ones(v.shape[0])
    elif len(weights) != v.shape[0]:
        raise ValueError(
            "Number of weights does not match number of vectors in \
                      rotation estimation"
        )
    else:
        a = np.array(weights)

    b = np.zeros((3, 3))
    for v_i, w_i, a_i in zip(v, w, a):
        b += a_i * w_i.reshape(3, 1) @ (v_i.reshape(3, 1).T)
    u, _, v_T = np.linalg.svd(b)
    m = np.diag([1, 1, np.linalg.det(u) * np.linalg.det(v_T.T)])
    r = u @ m @ v_T

    # Compute angle difference to estimate quality
    v_u = v / np.linalg.norm(v, axis=0)
    w2_u = r @ v
    w_u = w / np.linalg.norm(w, axis=0)
    dot_products = np.sum(w_u * w2_u, axis=0)
    delta_angle_deg = np.rad2deg(np.arccos(dot_products))
    return r, delta_angle_deg
