# pyright: reportAny=false

from ._backend_utils import (
    Array,
    backend_jit,
    Backend,
    coerce_ns
)

@backend_jit("backend")
def interp_nd(x: Array, xd: Array, fd: Array, backend: Backend = None) -> Array:
    """N-dimensional interpolation along first axis.
    Data is assumed to be already checked for monotonicity, shape, and bounds.
    This function is JIT compatible with JAX.

    Args:
        x (Array): 1D array of x-coordinates to interpolate at.
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
    
