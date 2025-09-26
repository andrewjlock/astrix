# pyright: reportAny=false

from scipy.spatial.transform import Rotation as R
from astrix._backend_utils import Backend, Array, backend_jit, resolve_backend


@backend_jit("backend")
def cam_to_frd_mat(backend: Backend) -> Array:
    """Rotation matrix to convert from camera frame to NED frame."""
    xp = resolve_backend(backend)
    return xp.asarray(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

@backend_jit("backend")
def window_rh_rot(backend: Backend) -> Array:
    """Right-hand window matrix."""
    xp = resolve_backend(backend)
    return R.from_euler("Z", xp.asarray([[90]]), degrees=True).as_matrix()

@backend_jit("backend")
def window_lh_rot(backend: Backend) -> Array:
    """Left-hand window matrix."""
    xp = resolve_backend(backend)
    return R.from_euler("Z", xp.asarray([[-90]]), degrees=True).as_matrix()

