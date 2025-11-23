# pyright: basic

import pytest
import jax
from astrix import Time
from astrix.functs import interp_nd, vec_from_az_el, central_diff


def _require_jax(xp):
    if not xp.__name__.startswith("jax"):
        pytest.skip("JAX-only test")


def test_interp_nd_jit(xp):
    _require_jax(xp)
    xd = xp.asarray([0.0, 1.0, 2.0])
    fd = xp.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    f = jax.jit(lambda x: interp_nd(x, xd, fd, backend=xp))
    out = f(xp.asarray([0.5, 1.5]))
    expected = xp.asarray([[0.5, 0.5], [1.5, 0.5]])
    assert xp.allclose(out, expected, atol=1e-6)


def test_vec_from_az_el_grad(xp):
    _require_jax(xp)
    angles = xp.asarray([[10.0, 20.0]])
    func = lambda a: xp.sum(vec_from_az_el(a, backend=xp))
    grad = jax.grad(func)(angles)
    # Finite difference check
    eps = 1e-3
    fd = []
    for i in range(2):
        plus = angles.copy()
        minus = angles.copy()
        plus = plus.at[0, i].add(eps)
        minus = minus.at[0, i].add(-eps)
        fd_val = (func(plus) - func(minus)) / (2 * eps)
        fd.append(fd_val)
    fd = xp.stack(fd, axis=-1)
    assert xp.allclose(grad, fd, atol=1e-3)


def test_central_diff_jit(xp):
    _require_jax(xp)
    xd = xp.asarray([0.0, 1.0, 3.0, 6.0])
    fd = xp.asarray([[0.0], [1.0], [3.0], [6.0]])
    jit_cd = jax.jit(lambda arr: central_diff(xd, arr, backend=xp))
    out = jit_cd(fd)
    expected = central_diff(xd, fd, backend=xp)
    assert xp.allclose(out, expected, atol=1e-6)
