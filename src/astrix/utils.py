from ._backend_utils import Array, ArrayNS, coerce_ns


def ensure_1d(x: Array, xp: ArrayNS | None = None) -> Array:
    """Ensure the input array is 1-dimensional. 
    Scalars are converted to shape (1,).
    If not specified, the default backend ('xp') is NumPy.
"""
    xp = coerce_ns(xp)
    x = xp.asarray(x)
    if x.ndim == 0:
        x = xp.reshape(x, (1,))
    elif x.ndim > 1:
        raise ValueError("Input array must be 1-dimensional or scalar.")
    return x


def ensure_2d(x: Array, n: int | None = None, xp: ArrayNS | None = None) -> Array:
    """Ensure the input array is 2-dimensional. 
    If n is given, ensure the second dimensionn has size n.
    if not specified, the default backend ('xp') is NumPy.
    """
    xp = coerce_ns(xp)
    x = xp.asarray(x)
    if x.ndim == 0:
        x = xp.reshape(x, (1, 1))
    elif x.ndim == 1:
        x = xp.reshape(x, (1, -1))
    elif x.ndim > 2:
        raise ValueError("Input array must be 2-dimensional or less.")
    if n is not None and x.shape[1] != n:
        raise ValueError(f"Input array must have shape (m, {n}), found {x.shape}.")
    return x
