# pyright: basic

import pytest
import jax
from astrix.primitives import Path, Point, Time

def test_path_interpolate(xp):
    
    ecef = xp.array([
        [3877000.0, 350000.0, 5027000.0],
        [3875000.0, 348000.0, 5029000.0],
        [3876000.0, 349000.0, 5028000.0],
    ])
    posix = xp.array([1577836920.0, 1577836800.0, 1577836860.0])  # 2020-01-01 00:02:00, 2020-01-01 00:00:00, 2020-01-01 00:01:00
    time = Time(posix, backend=xp)
    point = Point(ecef, time=time, backend=xp)
    path = Path(point, backend=xp)

    t_interp = Time(xp.array([1577836830.0]), backend=xp)  # 2020-01-01 00:00:30

    point_interp = path.interpolate(t_interp)
    
    def myfunction(t):
        time = Time(t, backend=xp)
        return path.interpolate(time).ecef

    jacobian = jax.jacobian(myfunction)(t_interp.secs)
    
    assert jacobian.shape == (1, 3, 1)

    assert point_interp.ecef.shape == (1,3)
    correct = ecef[1] + (ecef[2] - ecef[1]) * 0.5
    assert xp.allclose(point_interp.ecef[0], correct, atol=1e-1)

