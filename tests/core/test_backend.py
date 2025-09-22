import pytest

from astrix._backend_utils import resolve_backend, get_backend
import numpy as np

def test_backend_import():
    import astrix as at
    at._backend_utils.resolve_backend("np")
    if at._backend_utils.HAS_JAX:
        at._backend_utils.resolve_backend("jnp")

def test_get_backend():
    assert resolve_backend("np") is np
    assert resolve_backend(None) is np
    assert resolve_backend("numpy") is np
    assert resolve_backend("jax") is resolve_backend("jnp")


    assert get_backend(np.asarray([1,2,3]), np.asarray([4,5,6])).__name__ is np.__name__
    assert get_backend(np.asarray([1., 2., 3.])).__name__ is np.__name__

    with pytest.raises(ValueError):
        resolve_backend("unknown_backend") 






