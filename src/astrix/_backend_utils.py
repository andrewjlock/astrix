# pyright: reportExplicitAny=false

from __future__ import annotations
import os
import sys
import warnings
from typing import Final, Any, cast
from typing import TYPE_CHECKING, TypeAlias
from types import ModuleType
from functools import lru_cache
from importlib.util import find_spec
import array_api_compat.numpy as np

# Create as NameSpace for type hints and LSP
if TYPE_CHECKING:
    from array_api._2024_12 import Array as _Array, ArrayNamespace as _ANS

    # Parameterize the generics so pyright stops asking for args
    Array: TypeAlias = _Array[Any, Any]
    ArrayNS: TypeAlias = _ANS[Array, Any, Any]
else:
    # No runtime dependency on stubs
    from typing import Any as ArrayNS, Any as Array


HAS_JAX: Final = (find_spec("jax") is not None) and (find_spec("jaxlib") is not None)

BackendArg = str | ArrayNS | ModuleType | None


# Default to Numpy in case of None
@lru_cache(None)
def coerce_ns(xp: ArrayNS | None) -> ArrayNS:
    """Coerce the input to an Array Namespace (ArrayNS)."""
    if xp is not None:
        return xp
    if xp is None:
        return cast(ArrayNS, cast(Any, np))


@lru_cache(None)
def require_jax():
    if not HAS_JAX:
        raise ImportError(
            "This feature requires JAX. \n  \
                          Please install it and try again"
        )


@lru_cache(None)
def resolve_backend(
    name_or_mod: str | ArrayNS | None = None,
) -> ArrayNS:
    if name_or_mod in (None, "np", "numpy"):
        return cast(ArrayNS, cast(Any, np))
    if name_or_mod in ("jax", "jnp"):
        require_jax()
        import jax.numpy as jnp

        return cast(ArrayNS, cast(Any, jnp))
    if isinstance(name_or_mod, ModuleType):
        return cast(ArrayNS, name_or_mod)
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
