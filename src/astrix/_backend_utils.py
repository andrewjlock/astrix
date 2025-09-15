import os
import sys
import warnings
from typing import Final, Optional, Union
from types import ModuleType
from functools import lru_cache
from importlib.util import find_spec
import array_api_compat
from array_api._2024_12 import Array, ArrayNamespace, ShapedAnyArray

HAS_JAX: Final = (find_spec("jax") is not None) and (find_spec("jaxlib") is not None)

BackendArg = Optional[Union[str, ModuleType]]


@lru_cache(None)
def require_jax():
    if not HAS_JAX:
        raise ImportError(
            "This feature requires JAX. \n  \
                          Please install it and try again"
        )


@lru_cache(None)
def resolve_backend(
    name_or_mod: Optional[Union[str, ModuleType]] = None,
) -> ArrayNamespace:
    if name_or_mod in (None, "np", "numpy"):
        import array_api_compat.numpy as np

        return np
    if name_or_mod in ("jax", "jnp"):
        require_jax()
        import jax.numpy as jnp

        return jnp
    return name_or_mod


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
                f"JAX not in CPU+x64 mode (x64={x64}, backend={
                    backend}, devices={devs}). "
                "Set JAX_ENABLE_X64=1 and JAX_PLATFORMS=cpu before importing JAX.",
                stacklevel=2,
            )
