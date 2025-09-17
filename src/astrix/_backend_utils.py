# pyright: reportExplicitAny=false, reportAny=false

from __future__ import annotations
import os
import sys
import warnings
from typing import Final, Any, TypeAlias
from typing import TYPE_CHECKING
from types import ModuleType
from functools import lru_cache
from importlib.util import find_spec
from array_api_compat import numpy as np, array_namespace
from numpy.typing import NDArray


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


def get_backend(*args: Any) -> ArrayNS:
    """Get the backend (Array Namespace) of the input array(s). 
    If multiple arrays are given, ensure they all have the same backend.
    If not arrags are given (or non-array arguments), return NumPy as default."""

    if len(args) == 1:
        if hasattr(args[0], "__array_namespace__"):
            return args[0].__array_namespace__() # pyright: ignore
        return np
    if len(args) == 0:
        return np
    try:
        return array_namespace(*args,)
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
        if arg is not np:
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
    if name_or_mod in ("jax", "jnp"):
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
