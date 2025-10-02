# pyright: basic

import pytest
from astrix import Path, Point, Time
from .helpers import to_text
import numpy as np


@pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_path_create(xp):
    ecef = xp.array(
        [
            [3877000.0, 350000.0, 5027000.0],
            [3875000.0, 348000.0, 5029000.0],
            [3876000.0, 349000.0, 5028000.0],
        ]
    )
    posix = xp.array(
        [1577836920.0, 1577836800.0, 1577836860.0]
    )  # 2020-01-01 00:02:00, 2020-01-01 00:00:00, 2020-01-01 00:01:00

    time = Time(posix, backend=xp)
    point = Point(ecef, time=time, backend=xp)
    path = Path(point)
    to_text(path)

    assert path.ecef.shape == (3, 3)
    assert path.start_time.unix == posix[1]
    assert path.end_time.unix == posix[0]
    assert path.geodet.shape == (3, 3)
    assert path.vel.vec.shape == (3, 3)

    path.convert_to(np)

    point1 = Point(ecef[0], time=time[0], backend=xp)
    point2 = Point(ecef[1], time=time[1], backend=xp)
    path2 = Path([point1, point2], backend=xp)



def test_bad_init(xp):
    ecef = xp.array(
        [
            [3877000.0, 350000.0, 5027000.0],
            [3875000.0, 348000.0, 5029000.0],
            [3876000.0, 349000.0, 5028000.0],
        ]
    )
    posix = xp.array(
        [1577836920.0, 1577836800.0, 1577836860.0]
    )  # 2020-01-01 00:02:00, 2020-01-01 00:00:00, 2020-01-01 00:01:00
    time = Time(posix, backend=xp)

    with pytest.raises(AttributeError):
        Path(ecef)  # not a Point

    point_no_time = Point(ecef, backend=xp)
    with pytest.raises(ValueError):
        Path(point_no_time)  # Point has no time

    with pytest.raises(ValueError):
        point_bad_time = Point(ecef, time=time[::2], backend=xp)  # time length mismatch
        Path(point_bad_time)


@pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_path_interp(xp):
    ecef = xp.array(
        [
            [3877000.0, 350000.0, 5027000.0],
            [3875000.0, 348000.0, 5029000.0],
            [3876000.0, 349000.0, 5028000.0],
        ]
    )
    posix = xp.array(
        [1577836920.0, 1577836800.0, 1577836860.0]
    )  # 2020-01-01 00:02:00, 2020-01-01 00:00:00, 2020-01-01 00:01:00
    time = Time(posix, backend=xp)
    point = Point(ecef, time=time, backend=xp)
    path = Path(point)

    t_interp = Time(xp.array([1577836830.0]), backend=xp)  # 2020-01-01 00:00:30
    point_interp = path.interp(t_interp)
    point_interp_hav = path.interp(t_interp, method="haversine")
    assert xp.allclose(point_interp.ecef, point_interp_hav.ecef, atol=1e0)
    with pytest.raises(ValueError):
        path.interp(t_interp, method="bad")

    assert point_interp.ecef.shape == (1, 3)
    correct = ecef[1] + (ecef[2] - ecef[1]) * 0.5
    assert xp.allclose(point_interp.ecef[0], correct, atol=1e-1)

    with pytest.warns():
        t_bad = Time(xp.array([1577836790.0]), backend=xp)  # before start time
        path.interp(t_bad)

    ecef2 = xp.array(
        [
            [3878000.0, 351000.0, 5026000.0],
            [3879000.0, 352000.0, 5025000.0],
        ]
    )
    posix2 = xp.array(
        [1577836980.0, 1577837040.0]
    )  # 2020-01-01 00:03:00, 2020-01-01 00:04:00
    time2 = Time(posix2, backend=xp)
    point2 = Point(ecef2, time=time2, backend=xp)
    path2 = Path(point2, backend=xp)
    t_interp2 = Time(xp.array([1577837000.0]), backend=xp)  # 2020-01-01 00:03:20
    point_interp2 = path2.interp(t_interp2)
    assert point_interp2.ecef.shape == (1, 3)
    correct2 = ecef2[0] + (ecef2[1] - ecef2[0]) * (20.0 / 60.0)
    assert xp.allclose(point_interp2.ecef[0], correct2, atol=1e-1)
    assert path2.vel.vec.shape == (2, 3)
    assert xp.allclose(path2.vel.vec[0], (ecef2[1] - ecef2[0]) / 60.0, atol=1e-6)
    assert xp.allclose(path2.vel.vec[1], (ecef2[1] - ecef2[0]) / 60.0, atol=1e-6)


def test_path_vel_interp(xp):
    ecef = xp.array(
        [
            [3877000.0, 350000.0, 5027000.0],
            [3875000.0, 348000.0, 5029000.0],
            [3876000.0, 349000.0, 5028000.0],
        ]
    )
    posix = xp.array(
        [1577836920.0, 1577836800.0, 1577836860.0]
    )  # 2020-01-01 00:02:00, 2020-01-01 00:00:00, 2020-01-01 00:01:00
    time = Time(posix, backend=xp)
    point = Point(ecef, time=time, backend=xp)
    path = Path(point)

    t_interp = Time(xp.array([1577836830.0]), backend=xp)  # 2020-01-01 00:00:30
    vel_interp = path.interp_vel(t_interp)
    to_text(vel_interp)
    assert vel_interp.vec.shape == (1, 3)
    correct = (ecef[2] - ecef[1]) / (
        posix[2] - posix[1]
    )  # velocity between point 1 and 2
    assert xp.allclose(vel_interp.vec[0], correct, atol=1e-6)
    assert xp.allclose(vel_interp.magnitude, xp.linalg.norm(correct), atol=1e-6)
    assert xp.allclose(vel_interp.unit, correct / xp.linalg.norm(correct), atol=1e-6)
    assert len(vel_interp) == 1
    vel_interp.convert_to(np)

    with pytest.warns():
        t_bad = Time(xp.array([1577836790.0]), backend=xp)  # before start time
        path.interp_vel(t_bad)
