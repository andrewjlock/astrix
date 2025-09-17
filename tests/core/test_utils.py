from astrix.utils import ensure_1d
from astrix.utils import ensure_2d

def test_enforce1d(xp):
    x = 1.0
    y = ensure_1d(x, xp)
    assert y.shape == (1,)
    assert y[0] == x

    x = xp.asarray([1.0, 2.0, 3.0])
    y = ensure_1d(x, xp)
    assert y.shape == (3,)

def test_enforce2d(xp):
    x = 1.0
    y = ensure_2d(x,  backend=xp)
    assert y.shape == (1, 1)
    assert y[0, 0] == x

    x = [1.0, 2.0, 3.0]
    y = ensure_2d(x, 3, xp)
    assert y.shape == (1, 3)
    assert (y[0, :] == xp.asarray(x)).all()

    x = [[1.0, 2.0], [3.0, 4.0]]
    y = ensure_2d(x, 2, xp)
    assert y.shape == (2, 2)
    assert (y[:, :] == xp.asarray(x)).all()

    x = xp.asarray([1.0, 2.0, 3.0])
    y = ensure_2d(x, 3, xp)
    assert y.shape == (1, 3)
    assert (y[0, :] == x).all()

