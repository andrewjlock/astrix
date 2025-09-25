# pyright: reportExplicitAny=false, reportAny=false, reportAttributeAccessIssue=false

from __future__ import annotations
import os
import sys
import warnings
from typing import Final, Any, TypeAlias
from typing import TYPE_CHECKING
from types import ModuleType
from functools import lru_cache
from importlib.util import find_spec
from array_api_compat import numpy as array_namespace
import numpy as np
from numpy.typing import NDArray

os.environ["SCIPY_ARRAY_API"] = "True"
from scipy.spatial.transform import Rotation


if TYPE_CHECKING:
    """ Soon hopefully there will be a release of the array-api-typing package.
    These can then be replaced with proper types for better type checking."""
    Array: TypeAlias = NDArray[Any]
    ArrayNS: TypeAlias = ModuleType
else:
    ArrayNS = Any
    Array = Any


HAS_JAX: Final = (find_spec("jax") is not None) and (find_spec("jaxlib") is not None)

BackendArg = str | ArrayNS | None
Backend: TypeAlias = ArrayNS | None


def get_backend(*args: Any) -> ArrayNS:
    """Get the backend (Array Namespace) of the input array(s).
    If multiple arrays are given, ensure they all have the same backend.
    If not arrags are given (or non-array arguments), return NumPy as default."""

    if len(args) == 1:
        if hasattr(args[0], "__array_namespace__"):
            return args[0].__array_namespace__()
        return np
    if len(args) == 0:
        return np
    try:
        return array_namespace(
            *args,
        )
    except TypeError:
        return np


# Default to Numpy in case of None
def coerce_ns(xp: ArrayNS | None) -> ArrayNS:
    """Coerce the input to an Array Namespace (ArrayNS)."""
    if xp is not None:
        return xp
    if xp is None:
        return np


@lru_cache(None)
def require_jax():
    if not HAS_JAX:
        raise ImportError(
            "This feature requires JAX. \n  \
                          Please install it and try again"
        )


def warn_if_not_numpy(arg: ModuleType | Array, fun: str = ""):
    if isinstance(arg, ModuleType):
        if arg is not np:  # pyright: ignore[reportUnnecessaryComparison]
            warnings.warn(
                "Force converting backend to NumPy array for compatibility. "
                + "This is incompatible with JAX's JIT and autograd features. "
                + f"Function '{fun}' only supports NumPy backend.",
                stacklevel=2,
            )
    if hasattr(arg, "__array_namespace__"):
        if not isinstance(arg, np.ndarray):
            warnings.warn(
                "Force converting JaX array to NumPy array for compatibility. "
                + "This is incompatible with JAX's JIT and autograd features. "
                + f"Function '{fun}' only supports NumPy arrays.",
                stacklevel=2,
            )


@lru_cache(None)
def resolve_backend(
    name_or_mod: str | ModuleType | None = None,
) -> ArrayNS:
    """Resolve the backend (array namespace) from a string or module."""
    if name_or_mod in (None, "np", "numpy"):
        return np
    if name_or_mod in ("jax", "jax.numpy", "jnp"):
        require_jax()
        import jax.numpy as jnp

        return jnp
    if isinstance(name_or_mod, ModuleType):
        return name_or_mod
    raise ValueError(
        f"Unknown backend '{name_or_mod}'. Supported backends are None/'np'/'numpy' (NumPy) and 'jax'/'jnp' (JAX)."
    )


def enforce_cpu_x64():
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "jax" in sys.modules:
        import jax

        x64 = bool(jax.config.read("jax_enable_x64"))
        backend = jax.default_backend()
        devs = [d.platform for d in jax.devices()]
        if (not x64) or backend != "cpu" or any(p != "cpu" for p in devs):
            warnings.warn(
                f"JAX not in CPU+x64 mode (x64={x64}, backend= \
                {backend}, devices={devs}). "
                + "Set JAX_ENABLE_X64=1 and JAX_PLATFORMS=cpu before importing JAX.",
                stacklevel=2,
            )


# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
def backend_jit(static_argnames: str | list[str] | None =None): #pyright: ignore
    def decorator(func):
        if not HAS_JAX:
            return func
        import jax    
        jitted_func = jax.jit(func, static_argnames=static_argnames)
        
        def wrapper(*args, **kwargs):
            backend = kwargs.get('backend')
            # Only use JIT if backend is JAX
            if backend and hasattr(backend, '__name__') and 'jax' in str(backend):
                return jitted_func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def safe_set(
    array: Array,
    index: int | slice | tuple[int, ...] | tuple[slice, ...] | Array,
    value: float | int | Array,
    backend: Backend,
) -> Array:
    """Set value(s) in an array at the given index/indices, returning a new array.
    This is safe for JAX arrays (which are immutable) and works with NumPy arrays too.

    Args:
        array (Array): Input array.
        index (int | slice | tuple | Array): Index or indices to set.
        value (float | int | Array): Value(s) to set.

    Returns:
        Array: New array with the value(s) set.
    """
    xp = coerce_ns(backend)
    if xp.__name__ == "jax.numpy":
        return xp.array(array).at[index].set(value)
    else:
        array[index] = value
        return array

def _convert_rot_backend(rot: Rotation, backend: Backend) -> Rotation:
    """Convert a scipy Rotation object to the given backend.
    This is a no-op for NumPy backend, and converts to JAX arrays for JAX backend.
    """
    xp = coerce_ns(backend)
    if xp is rot._xp:
        return rot
    quat = xp.asarray(rot._quat, copy=True)
    return Rotation.from_quat(quat)
