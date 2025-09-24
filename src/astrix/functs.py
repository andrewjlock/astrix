# pyright: reportAny=false

from scipy.spatial.transform import Rotation as R

from .utils import ecef2geodet, geodet2ecef
from ._backend_utils import (
    Array,
    backend_jit,
    Backend,
    coerce_ns,
    safe_set,
    warn_if_not_numpy,
)


@backend_jit("backend")
def interp_nd(x: Array, xd: Array, fd: Array, backend: Backend = None) -> Array:
    """N-dimensional interpolation along first axis.
    Data is assumed to be already checked for monotonicity, shape, and bounds.
    This function is JIT compatible with JAX.

    Args:
        x (Array): 1D array of x-coordinates to nterpolate at.
        xd (Array): 1D array of x-coordinates of the data points.
        fd (Array): N-D array of data points to interpolate.

    Returns:
        Array: Interpolated values at x.
    """

    xp = coerce_ns(backend)
    x = xp.asarray(x)
    xd = xp.asarray(xd)
    fd = xp.asarray(fd)

    indices = xp.searchsorted(xd, x, side="right") - 1
    indices = xp.clip(indices, 0, xd.shape[0] - 2)

    x0, x1 = xd[indices], xd[indices + 1]
    f0, f1 = fd[indices], fd[indices + 1]

    slope = (f1 - f0) / (x1 - x0)[..., xp.newaxis]
    return f0 + slope * (x - x0)[..., xp.newaxis]


@backend_jit("backend")
def central_diff(xd: Array, fd: Array, backend: Backend = None) -> Array:
    """
    Derivative of fd w.r.t. xd at the sample nodes, using:
      - 3-point *non-uniform* central differences in the interior (O(h^2))
      - 3-point one-sided stencils at the ends (O(h))

    Args
    ----
    xd : (N,) strictly increasing array of x-locations, must have N>=3
    fd : (N, ...) array of function values; derivative is taken along axis 0
    backend : optional, backend to use (numpy, jax, etc.)

    Returns
    -------
    dfdx : (N, ...) array of derivatives at each xd[i]

    Notes
    -----
    Must have xd.shape[0] >= 3. No bounds checking is performed

    """

    xp = coerce_ns(backend)
    xd = xp.asarray(xd)
    fd = xp.asarray(fd)

    # Interior: i = 1..n-2
    a = xd[1:-1] - xd[:-2]  # (n-2,)
    b = xd[2:] - xd[1:-1]  # (n-2,)

    w_im1 = -b / (a * (a + b))  # (n-2,)
    w_i = (b - a) / (a * b)  # (n-2,)
    w_ip1 = a / (b * (a + b))  # (n-2,)

    # Broadcast weights over trailing dims of fd
    expand = (slice(None),) + (None,) * (fd.ndim - 1)
    interior = w_im1[expand] * fd[:-2] + w_i[expand] * fd[1:-1] + w_ip1[expand] * fd[2:]

    # Forward one-sided at i = 0 using x0,x1,x2
    h1 = xd[1] - xd[0]
    h2 = xd[2] - xd[1]
    c0 = -(2 * h1 + h2) / (h1 * (h1 + h2))
    c1 = (h1 + h2) / (h1 * h2)
    c2 = -h1 / (h2 * (h1 + h2))
    left = c0 * fd[0] + c1 * fd[1] + c2 * fd[2]

    # Backward one-sided at i = n-1 using x[n-3],x[n-2],x[n-1]
    h1b = xd[-2] - xd[-3]
    h2b = xd[-1] - xd[-2]
    d0 = h2b / (h1b * (h1b + h2b))
    d1 = -(h1b + h2b) / (h1b * h2b)
    d2 = (h1b + 2 * h2b) / (h2b * (h1b + h2b))
    right = d0 * fd[-3] + d1 * fd[-2] + d2 * fd[-1]

    # Assemble output
    dfdx = xp.empty_like(fd)
    dfdx = safe_set(dfdx, 0, left, backend)
    dfdx = safe_set(dfdx, slice(1, -1), interior, backend)
    dfdx = safe_set(dfdx, -1, right, backend)

    return dfdx

def finite_diff_2pt(xd: Array, fd: Array, backend: Backend = None) -> Array:
    """ Simple 2-point finite difference derivative of fd w.r.t. xd at the sample nodes.
    Uses forward difference at the first point, backward difference at the last point,
    and central difference in the interior.

    Args
    ----
    xd : (N,) strictly increasing array of x-locations, must have N>=2
    fd : (N, ...) array of function values; derivative is taken along axis 0
    backend : optional, backend to use (numpy, jax, etc.)

    Returns
    -------
    dfdx : (N, ...) array of derivatives at each xd[i]

    Notes
    -----
    Must have xd.shape[0] >= 2. No bounds checking is performed
    """

    xp = coerce_ns(backend)
    xd = xp.asarray(xd)
    fd = xp.asarray(fd)

    dfdx = xp.empty_like(fd)
    dfdx = safe_set(dfdx, 0, (fd[1] - fd[0]) / (xd[1] - xd[0]), backend)
    dfdx = safe_set(dfdx, -1, (fd[-1] - fd[-2]) / (xd[-1] - xd[-2]), backend)
    if xd.shape[0] > 2:
        dfdx = safe_set(dfdx, slice(1, -1), (fd[2:] - fd[:-2]) / (xd[2:] - xd[:-2])[:, None], backend)
    return dfdx






def great_circle_distance(
    geodet1: Array, geodet2: Array, backend: Backend = None, ignore_elev: bool = False
) -> Array:
    """Uses haversine formual for cirular distance
    Accounts for change in altitude using euclidian norm

    Ref: http://www.movable-type.co.uk/scripts/latlong.html

    Can choose to ignore change in elevation for great circle distance at
    constant altitude

    Args:
        geodet1 (Array): Nx3 array of lat, long, alt [deg, deg, m]
        geodet2 (Array): Nx3 array of lat, long, alt [deg, deg, m]
        backend (Backend, optional):
        ignore_elev (bool, optional): If True, ignore elevation change. Defaults to False.

    Returns:
        Array: Array of distances [m] between each pair of points
    """

    xp = coerce_ns(backend)
    warn_if_not_numpy(xp)

    rs1 = xp.linalg.norm(geodet2ecef(geodet1), axis=1)
    rs2 = xp.linalg.norm(geodet2ecef(geodet2), axis=1)

    lat1s = xp.deg2rad(geodet1[:, 0])
    lon1s = xp.deg2rad(geodet1[:, 1])
    lat2s = xp.deg2rad(geodet2[:, 0])
    lon2s = xp.deg2rad(geodet2[:, 1])

    alphas = xp.sin((lat2s - lat1s) / 2) ** 2 + xp.cos(lat1s) * (
        xp.sin((lon2s - lon1s) / 2) ** 2
    )

    cs = 2 * xp.arctan2(alphas**0.5, (1 - alphas) ** 0.5)
    ds_geodet = rs1 * cs
    if ignore_elev:
        return ds_geodet
    ds = (ds_geodet**2 + (rs2 - rs1) ** 2) ** 0.5
    return ds


def interp_haversine(
    secs: Array, secs_data: Array, ecef_data: Array, backend: Backend = None
) -> Array:
    """
    Interpolate ECEF trajectory data to given times using great-circle (haversine) interpolation.
    This is more accurate for long distances than linear interpolation in ECEF space.
    Approximates the Earth as a sphere (reasonable for interpolation).

    Args:
        secs (Array): 1D array of times [s] to interpolate at.
        secs_data (Array): 1D array of times [s] of the data points.
        ecef_data (Array): Nx3 array of ECEF x,y,z points [m] of the data points.
        backend (Backend, optional): Backend to use (numpy, jax, etc.). Defaults to None.

    Returns:
        Array: Mx3 array of interpolated ECEF x,y,z points [m] at times secs.

    Notes:
        - This function is not compatible with Jax JIT or grad due to the use of pyproj.
        - secs_data must be strictly increasing and within the bounds of secs.
        - ecef_data and secs_data must have the same length along axis 1.
        - Uses WGS84 Earth radius for haversine calculations.
    """

    xp = coerce_ns(backend)
    warn_if_not_numpy(xp)

    geodet = ecef2geodet(ecef_data)

    inds = xp.searchsorted(secs_data, secs) - 1

    r1 = xp.linalg.norm(ecef_data[inds, :], axis=1)

    geodet1 = geodet[inds, :]
    geodet2 = geodet[inds + 1, :]
    lat1s = xp.deg2rad(geodet1[:, 0])
    lon1s = xp.deg2rad(geodet1[:, 1])
    lat2s = xp.deg2rad(geodet2[:, 0])
    lon2s = xp.deg2rad(geodet2[:, 1])

    fs = (secs - secs_data[inds]) / (secs_data[inds + 1] - secs_data[inds])
    ds_geodet = great_circle_distance(geodet1, geodet2, ignore_elev=True)
    deltas = ds_geodet / r1
    a = xp.sin((1 - fs) * deltas) / xp.sin(deltas)
    b = xp.sin(fs * deltas) / xp.sin(deltas)
    x = a * xp.cos(lat1s) * xp.cos(lon1s) + b * xp.cos(lat2s) * xp.cos(lon2s)
    y = a * xp.cos(lat1s) * xp.sin(lon1s) + b * xp.cos(lat2s) * xp.sin(lon2s)
    z = a * xp.sin(lat1s) + b * xp.sin(lat2s)
    lat_i = xp.arctan2(z, (x**2 + y**2) ** 0.5)
    lon_i = xp.arctan2(y, x)
    z_i = xp.interp(secs, secs_data, geodet[:, 2])

    geodet = xp.array([xp.rad2deg(lat_i), xp.rad2deg(lon_i), z_i]).T
    ecef = geodet2ecef(geodet)
    return ecef


def ned_rotation(geodet: Array, xp: Backend = None) -> R:
    """Get the rotation from ECEF base to the North-East-Down frame at given 
    geodetic locations.

    Args:
        pos_geodet (Array): Nx3 array of lat, long, alt [deg, deg, m]

    Returns:
        Rotation: scipy Rotation object representing the NED frame rotation
    """

    xp = coerce_ns(xp)
    rot_ned0 = R.from_matrix(
        xp.asarray(
            [
                [0, 0, -1],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
    )
    rot_ned_ned0 = R.from_euler("XY", xp.asarray([geodet[:, 1], -geodet[:, 0]]).T, degrees=True)
    rot_ned = rot_ned0 * rot_ned_ned0
    return rot_ned
