# pyright: reportAny=false
# pyright: standard

"""Functions for observation flight path planning.

Note: These functions are typically only applicable for the NumPy backend.
"""


from .spatial.location import Point
from ._backend_utils import (
    np,
    warn_if_not_numpy,
)
from .time import TIME_INVARIANT, Time
from .functs import ned_rotation


def point_from_heading(
    start: Point,
    head: float,
    dist: float,
    time_new: Time = TIME_INVARIANT,
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


def turn_from_radius(
    start_point: Point,
    start_heading: float,
    degrees: float,
    radius: float,
    speed: float,
    n: int | None = None,
):
    """Create a set of points representing a turn.

    Does not include the starting point (to reduce duplication when flight path
    constructed from points)

    Parameters
    ----------
    start_point : Point
        starting point of turn. If len(start_point) > 1, the last point is used.
    start_heading : float
        initial heading, in degrees
    degrees : float
        degrees of the turn, in degrees (i.e. 180 for half-turn)
    radius : float
        radius of the turn (in metres)
    speed : float
        air speed of the plan during turn (m/s)
    n :  int, optional
        number of points created to represent the turn, defualt is 1 per degree

    Notes:
    - There will be some small mis-match due to cartesian definition of turn,
    and haversine/rhumb line definition of distance.
    - I'm sure there are much nicer ways which are more analytically correct,
    which would be good to implement in the future.
    """

    point_start = start_point[-1]  # use last point if multiple
    if not isinstance(point_start.time, Time):
        raise ValueError("start_point must have associated Time for constructing turn.")
    if n is None:
        n = int(np.rint(np.abs(degrees)))
    turn_rate = np.sign(degrees) * speed / radius  # rad/s

    point_centre = point_from_heading(
        point_start,
        start_heading + 90 * np.sign(degrees),
        radius,
        method="haversine",
    )
    ned_rot = ned_rotation(point_centre.geodet)
    start_rel_ned = ned_rot.as_matrix().T @ (point_start.ecef - point_centre.ecef)
    start_angle = np.rad2deg(np.arctan2(start_rel_ned[1], start_rel_ned[0]))

    t = np.linspace(0.0, np.deg2rad(degrees), n + 1)[1:]  # skip first point
    xt = radius * np.cos(np.deg2rad(start_angle) + t * turn_rate * np.sign(degrees))
    yt = radius * np.sin(np.deg2rad(start_angle) + t * turn_rate * np.sign(degrees))
    zt = np.repeat(0.0, n)

    pts_rel = np.vstack((xt, yt, zt)).T
    pts_ecef = (ned_rot.as_matrix() @ pts_rel.T).T + point_centre.ecef
    unix = point_start.time.unix + t
    times = Time(unix)
    pts = Point(pts_ecef, time=times, backend=np)
    return pts
