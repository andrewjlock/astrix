# pyright: basic

import pytest
import warnings
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
    vel_interp = path.vel.interp(t_interp)
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
        path.vel.interp(t_bad)


def test_path_downsample_max_step(xp):
    ecef = xp.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [25.0, 0.0, 0.0],
        ]
    )
    times = Time(xp.array([0.0, 30.0, 90.0]), backend=xp)
    path = Path(Point(ecef, time=times, backend=xp))

    if xp.__name__.startswith("jax"):
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        warnings.simplefilter("ignore", UserWarning)
    try:
        downsampled = path.downsample(dt_max=20.0)
    finally:
        if xp.__name__.startswith("jax"):
            ctx.__exit__(None, None, None)

    unix = np.asarray(downsampled.time.unix)
    assert unix[0] == 0.0 and unix[-1] == 90.0
    assert np.all(np.diff(unix) <= 20.0 + 1e-9)
    assert len(unix) >= len(times)


def test_path_acceleration_constant_velocity(xp):
    ecef = xp.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]
    )
    times = Time(xp.array([0.0, 1.0, 2.0]), backend=xp)
    path = Path(Point(ecef, time=times, backend=xp))

    assert xp.allclose(path.vel.vec[:, 0], xp.asarray([10.0, 10.0, 10.0]))
    assert xp.allclose(path.vel.vec[:, 1:], 0.0)
    assert xp.allclose(path.acc.vec, 0.0, atol=1e-12)


def test_path_truncate(xp):
    ecef = xp.array(
        [
            [3877000.0, 350000.0, 5027000.0],
            [3875000.0, 348000.0, 5029000.0],
            [3876000.0, 349000.0, 5028000.0],
        ]
    )
    posix = xp.array([0.0, 30.0, 90.0])
    time = Time(posix, backend=xp)
    path = Path(Point(ecef, time=time, backend=xp))

    truncated = path.truncate(
        Time(xp.array([10.0]), backend=xp),
        Time(xp.array([60.0]), backend=xp),
    )
    assert xp.allclose(truncated.start_time.unix, xp.asarray([10.0]))
    assert xp.allclose(truncated.end_time.unix, xp.asarray([60.0]))
    assert truncated.ecef.shape[0] >= 2


def test_path_time_at_altitude(xp):
    geodetic = xp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0],
            [0.0, 0.0, 200.0],
        ]
    )
    times = Time(xp.array([0.0, 10.0, 20.0]), backend=xp)

    if xp.__name__.startswith("jax"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            points = Point.from_geodet(geodetic, time=times, backend=xp)
            path = Path(points, backend=xp)
            crossing_time = path.time_at_alt(50.0)
    else:
        points = Point.from_geodet(geodetic, time=times, backend=xp)
        path = Path(points, backend=xp)
        crossing_time = path.time_at_alt(50.0)
    assert crossing_time.unix.shape == (1,)
    assert xp.allclose(crossing_time.unix, xp.asarray([5.0]), atol=1e-3)


def test_path_truncate_errors(xp):
    ecef = xp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    times = Time(xp.array([0.0, 10.0]), backend=xp)
    path = Path(Point(ecef, time=times, backend=xp))

    with pytest.raises(ValueError):
        path.truncate(
            Time(xp.array([15.0]), backend=xp),
            Time(xp.array([20.0]), backend=xp),
        )

    with pytest.raises(ValueError):
        path.truncate(
            Time(xp.array([8.0]), backend=xp),
            Time(xp.array([5.0]), backend=xp),
        )


def test_path_convert_backend(xp):
    ecef = xp.array([[1.0, 2.0, 3.0]])
    times = Time(xp.array([0.0]), backend=xp)
    path = Path(Point(ecef, time=times, backend=xp))

    path_np = path.convert_to(np)
    assert path_np.time.backend == "numpy"
    assert path_np.ecef.shape == (1, 3)
