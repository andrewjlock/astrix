# pyright: reportAny=false

"""
Array and logic functions for Astrix.

Should not have any dependencies on core Astrix types to avoid circular dependencies.
Backend utility imports are allowed.

Guiding principles:
- Functions should be backend-agnstic where possible.
- Functions should be JIT-compatible, where possible.
- Data validation is not performed here, assume inputs are valid.
"""

from scipy.spatial.transform import Rotation
import pyproj

from ._backend_utils import (
    Array,
    ArrayLike,
    ArrayNS,
    np,
    warn_if_not_numpy,
    backend_jit,
    Backend,
    coerce_ns,
    safe_set,
)
from astrix.constants import cam_to_frd_mat

def ensure_1d(x: ArrayLike | float | list[float], backend: Backend = None) -> Array:
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
    x: ArrayLike | float | list[float] | list[list[float]],
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
    elif n is not None and x_arr.shape[1] != n:
        if x_arr.shape[0] == n and x_arr.shape[1] != n:
            x_arr = x_arr.T
        if x_arr.shape[1] != n:
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
        xp (ArrayNS, optional): Backend to use. Defaults to None.

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


def sort_by_time(
    times: Array,
    data: Array,
    backend: Backend = None,
) -> tuple[Array, Array]:
    """Sort data by increasing time.

    Args:
        times (Array): 1D array of times, len N.
        data (Array): NxM array of data to sort by time.
        backend (Backend, optional): Backend to use. Defaults to None.

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
def interp_unit_vec(t: Array, td: Array, vecs: Array, backend: Backend = None) -> Array:
    """Interpolate unit vectors along the first axis using spherical linear interpolation (slerp).

    Args:
        t (Array): 1D array of times to interpolate at.
        td (Array): 1D array of times of the data points.
        vecs (Array): Nx3 array of unit vectors to interpolate.

    Returns:
        Array: Interpolated unit vectors at times t.
    """

    xp = coerce_ns(backend)
    t = xp.asarray(t)
    td = xp.asarray(td)
    vecs = xp.asarray(vecs)

    indices = xp.searchsorted(td, t, side="right") - 1
    indices = xp.clip(indices, 0, td.shape[0] - 2)

    t0, t1 = td[indices], td[indices + 1]
    v0, v1 = vecs[indices], vecs[indices + 1]

    dot = xp.sum(v0 * v1, axis=1)
    dot = xp.clip(dot, -1.0, 1.0)
    theta = xp.arccos(dot)

    sin_theta = xp.sin(theta)
    sin_theta = xp.where(sin_theta == 0, 1e-10, sin_theta)

    f0 = xp.sin((t1 - t) / (t1 - t0) * theta) / sin_theta
    f1 = xp.sin((t - t0) / (t1 - t0) * theta) / sin_theta

    interp_vecs = f0[:, xp.newaxis] * v0 + f1[:, xp.newaxis] * v1
    interp_vecs /= xp.linalg.norm(interp_vecs, axis=1, keepdims=True)

    return interp_vecs


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


def ned_rotation(geodet: Array, xp: Backend = None) -> Rotation:
    """Get the rotation from ECEF base to the North-East-Down frame at given 
    geodetic locations.

    Args:
        pos_geodet (Array): Nx3 array of lat, long, alt [deg, deg, m]

    Returns:
        Rotation: scipy Rotation object representing the NED frame rotation
    """

    xp = coerce_ns(xp)
    rot_ned0 = Rotation.from_matrix(
        xp.asarray(
            [
                [0, 0, -1],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
    )
    rot_ned_ned0 = Rotation.from_euler("XY", xp.asarray([geodet[:, 1], -geodet[:, 0]]).T, degrees=True)
    rot_ned = rot_ned0 * rot_ned_ned0
    return rot_ned


@backend_jit(["backend"])
def az_el_from_vec(v: Array, backend: Backend = None) -> Array:
    """Compute azimuth and elevation from a set of 3D vectors.
    
    Args:
        v (Array): Array of shape (m, 3) of m 3D vectors.

    Returns:
        Array: Array of shape (m, 2) of azimuth and elevation angles in degrees.
            Azimuth is in [-180, 180), elevation is in [-90, 90].

    Notes:
        Assumes vectors are:
            - Right-handed coordinate system
            - x points forward
            - y points right
            - z points down
    """
    xp = coerce_ns(backend)
    az = xp.rad2deg(xp.arctan2(v[:, 1], v[:, 0]))
    el = xp.rad2deg(xp.arctan2(-v[:, 2], xp.linalg.norm(v[:, 0:2], axis=1)))

    return xp.stack((az, el), axis=1)

@backend_jit("backend")
def vec_from_az_el(az_el: Array, backend: Backend = None) -> Array:
    """Compute 3D unit vectors from azimuth and elevation angles.

    Args:
        az_el (Array): Array of shape (m, 2) of azimuth and elevation angles in degrees.
            Azimuth is in [0, 360), elevation is in [-90, 90].
        backend (Backend, optional): Backend to use. Defaults to None.

    Returns:
        Array: Array of shape (m, 3) of m 3D unit vectors.

    Notes:
        Assumes vectors are:
            - Right-handed coordinate system
            - x points forward
            - y points right
            - z points down
    """
    xp = coerce_ns(backend)
    az = xp.deg2rad(az_el[:, 0])
    el = xp.deg2rad(az_el[:, 1])

    x = xp.cos(el) * xp.cos(az)
    y = xp.cos(el) * xp.sin(az)
    z = -xp.sin(el)

    return xp.stack((x, y, z), axis=1)

@backend_jit("backend")
def total_angle_from_vec(v: Array, backend: Backend = None) -> Array:
    """Compute the total angle from a set of 3D vectors from forard (1, 0, 0).

    Args:
        v (Array): Array of shape (N, 3) of m 3D vectors.
        backend (Backend, optional): Backend to use. Defaults to None.

    Returns:
        Array: Array of shape (N,) of total angles in degrees.
            Total angle is in [0, 180].
    """

    xp = coerce_ns(backend)
    v_unit = v / xp.linalg.norm(v, axis=1, keepdims=True)
    total_angle = xp.rad2deg(xp.arccos(v_unit[:, 0]))
    return total_angle

@backend_jit("backend")
def pixel_to_vec(pixels: Array, mat: Array, backend: Backend = None) -> Array:
    """Convert pixel coordinates to 3D unit vectors in camera frame.

    Args:
        pixels (Array): Nx2 array of pixel coordinates (u, v).
        mat (Array): 3x3 or (N,3,3) camera intrinsic matrix.
        backend (Backend, optional): Backend to use. Defaults to None.

    Returns:
        Array: Nx3 array of 3D unit vectors in camera frame.
    """
    xp = coerce_ns(backend)
    pixels_h = xp.concatenate((pixels, xp.ones((pixels.shape[0], 1), dtype=pixels.dtype)), axis=1)
    mat_inv = xp.linalg.inv(mat)
    vecs_cam = xp.einsum("ijk,ik->ji", mat_inv.reshape(-1, 3, 3), pixels_h) # shape (3, N)
    # vecs_cam = mat_inv @ (pixels_h.T) # shape (3, N)
    vecs_cam_unit = vecs_cam / xp.linalg.norm(vecs_cam, axis=0, keepdims=True) # shape (3, N)
    vecs_frd_unit = cam_to_frd_mat(xp) @ vecs_cam_unit # shape (3, N)
    return vecs_frd_unit.T # shape (N, 3)

@backend_jit("backend")
def vec_to_pixel(vecs: Array, mat: Array, backend: Backend = None) -> Array:
    """Convert 3D unit vectors in camera frame to pixel coordinates.

    Args:
        vecs (Array): Nx3 array of 3D unit vectors in camera frame.
        mat (Array): 3x3 camera intrinsic matrix.
        backend (Backend, optional): Backend to use. Defaults to None.

    Returns:
        Array: Nx2 array of pixel coordinates (u, v).
    """
    xp = coerce_ns(backend)
    vecs_cam = cam_to_frd_mat(xp).T @ vecs.T # shape (3, N)
    # pixels_h = mat @ vecs_cam # shape (3, N)
    pixels_h = xp.einsum("ijk,ki->ji", mat.reshape(-1, 3, 3), vecs_cam) # shape (3, N)
    pixels = pixels_h[:2, :] / pixels_h[2, :] # shape (2, N)
    return pixels.T # shape (N, 2)




