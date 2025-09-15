from types import ModuleType
from typing import Optional
from ._backend_utils import resolve_backend, Array, ArrayNamespace, BackendArg
import numpy as np


def ensure_1d(x: Array, xp: ModuleType = np) -> Array:
    """Ensure the input array is 1-dimensional."""
    x = xp.asarray(x)
    if x.ndim == 0:
        x = xp.reshape(x, (1,))
    elif x.ndim > 1:
        raise ValueError("Input array must be 1-dimensional or scalar.")
    return x


def ensure_2d(x: Array, n: Optional[int] = None, xp: ModuleType = np) -> Array:
    """Ensure the input array is 2-dimensional."""
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
