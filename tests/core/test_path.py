# pyright: basic

import pytest
from astrix.primitives import Path, Point, Time

def test_path_create(xp):

    ecef = xp.array([
        [3877000.0, 350000.0, 5027000.0],
        [3875000.0, 348000.0, 5029000.0],
        [3876000.0, 349000.0, 5028000.0],
    ])
    posix = xp.array([1577836920.0, 1577836800.0, 1577836860.0])  # 2020-01-01 00:02:00, 2020-01-01 00:00:00, 2020-01-01 00:01:00
    time = Time(posix, backend=xp)
    point = Point(ecef, time=time, backend=xp)
    path = Path(point)

    assert path.ecef.shape == (3,3)
    assert path.start_time.secs == posix[1]
    assert path.end_time.secs == posix[0]

def test_path_interp(xp):
    
    ecef = xp.array([
        [3877000.0, 350000.0, 5027000.0],
        [3875000.0, 348000.0, 5029000.0],
        [3876000.0, 349000.0, 5028000.0],
    ])
    posix = xp.array([1577836920.0, 1577836800.0, 1577836860.0])  # 2020-01-01 00:02:00, 2020-01-01 00:00:00, 2020-01-01 00:01:00
    time = Time(posix, backend=xp)
    point = Point(ecef, time=time, backend=xp)
    path = Path(point)

    t_interp = Time(xp.array([1577836830.0]), backend=xp)  # 2020-01-01 00:00:30
    point_interp = path.interp(t_interp)

    assert point_interp.ecef.shape == (1,3)
    correct = ecef[1] + (ecef[2] - ecef[1]) * 0.5
    assert xp.allclose(point_interp.ecef[0], correct, atol=1e-1)

def test_path_vel_interp(xp):

    ecef = xp.array([
        [3877000.0, 350000.0, 5027000.0],
        [3875000.0, 348000.0, 5029000.0],
        [3876000.0, 349000.0, 5028000.0],
    ])
    posix = xp.array([1577836920.0, 1577836800.0, 1577836860.0])  # 2020-01-01 00:02:00, 2020-01-01 00:00:00, 2020-01-01 00:01:00
    time = Time(posix, backend=xp)
    point = Point(ecef, time=time, backend=xp)
    path = Path(point)

    t_interp = Time(xp.array([1577836830.0]), backend=xp)  # 2020-01-01 00:00:30
    vel_interp = path.interp_vel(t_interp)
    assert vel_interp.vec.shape == (1,3)
    correct = (ecef[2] - ecef[1]) / (posix[2] - posix[1])  # velocity between point 1 and 2
    assert xp.allclose(vel_interp.vec[0], correct, atol=1e-6)


