import pytest
from astrix._backend_utils import HAS_JAX

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not available")

# Override the xp fixture for this directory
@pytest.fixture
def xp():
    """JAX backend only for tests in jax/ directory."""
    from astrix._backend_utils import enforce_cpu_x64
    enforce_cpu_x64()
    import jax.numpy as jnp
    return jnp

