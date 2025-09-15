import importlib.util as spu
import pytest
import numpy as np

_HAS_JAX = (spu.find_spec("jax") is not None) and (spu.find_spec("jaxlib") is not None)

_backends = [np]
if _HAS_JAX:
    from astrix._backend_utils import enforce_cpu_x64
    enforce_cpu_x64()
    import jax.numpy as jnp
    _backends.append(jnp)


@pytest.fixture(params=_backends, ids=lambda m: m.__name__.split(".")[0])
def xp(request):
    """Backend module (np or jnp if available)."""
    return request.param
