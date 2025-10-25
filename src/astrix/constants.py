# pyright: reportAny=false

from scipy.spatial.transform import Rotation
from astrix._backend_utils import Backend, Array, backend_jit, resolve_backend


@backend_jit("backend")
def cam_to_frd_mat(backend: Backend = None) -> Array:
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
def rot_window_rh(backend: Backend = None) -> Rotation:
    """Right-hand window matrix."""
    xp = resolve_backend(backend)
    return Rotation.from_euler("Z", xp.asarray([90]), degrees=True)

@backend_jit("backend")
def rot_window_lh(backend: Backend = None) -> Rotation:
    """Left-hand window matrix."""
    xp = resolve_backend(backend)
    return Rotation.from_euler("Z", xp.asarray([-90]), degrees=True)

@backend_jit("backend")
def rot_identity(backend: Backend = None) -> Rotation:
    """Identity rotation matrix."""
    xp = resolve_backend(backend)
    return Rotation.from_quat(xp.array([[0, 0, 0, 1]]))

