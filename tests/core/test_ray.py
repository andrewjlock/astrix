# pyright: basic

import pytest
import numpy as np
from .helpers import to_text
from astrix import Ray, Time

def test_single_ray_init(xp):
    origin = xp.array([1.0, 2.0, 3.0])
    direction = xp.array([0.0, 1.0, 0.0])
    ray1 = Ray(origin, direction, backend=xp)

    time = Time(0.0, backend=xp)
    ray2 = Ray(origin, direction, time=time, backend=xp)
    ray2.convert_to(np)


def test_bad_init(xp):
    origin = xp.array([1.0, 2.0, 3.0])
    direction = xp.array([0.0, 1.0])  # Incorrect shape

    with pytest.raises(ValueError):
        Ray(origin, direction, backend=xp)

    direction = xp.array([0.0, 0.0, 0.0])  # Zero vector

    with pytest.raises(ValueError):
        Ray(origin, direction, backend=xp)

    direction = xp.array([0.0, 1.0, 0.0])
    time = Time([0.0, 1.0], backend=xp)
    with pytest.raises(ValueError):
        Ray(origin, direction, time=time, backend=xp)


@pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_multiple_ray_init(xp):
    origins = xp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    directions = xp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    time = Time([0.0, 1.0], backend=xp)
    ray1 = Ray(origins, directions, time, backend=xp)
    to_text(ray1)
    assert len(ray1) == 2
    assert ray1.origin.shape == (2, 3)
    assert ray1.unit.shape == (2, 3)
    assert ray1.has_time is True

    time_interp = Time(0.5, backend=xp)
    ray2 = ray1.interp(time_interp)
    assert len(ray2) == 1
    assert ray2.origin.shape == (1, 3)
    assert ray2.unit.shape == (1, 3)
    assert ray2.has_time is True
    assert xp.allclose(ray2.origin, xp.array([[2.5, 3.5, 4.5]]))
    assert xp.allclose(ray2.unit, xp.array([[0.70710678, 0.70710678, 0.0]]))

    time_bad = Time([0.0, 1.0, 2.0], backend=xp)
    with pytest.warns():
        ray3 = ray1.interp(time_bad)

    ray4 = Ray(origins, directions, backend=xp)
    assert ray4.has_time is False
    with pytest.raises(ValueError):
        ray4.interp(time_interp)




    




