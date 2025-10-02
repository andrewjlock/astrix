# pyright: basic

import pytest
import datetime as dt
from astrix import Point, Time
from .helpers import to_text
import numpy as np


@pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_point_no_time(xp):
    brisbane_geodet = xp.array([[-27.4705, 153.0260, 27]])
    brisbane_ecef = xp.array([[-5046981.62, 2568681.41, -2924582.78]])

    p_geo = Point.from_geodet(brisbane_geodet, backend=xp)
    to_text(p_geo)

    assert p_geo.ecef.shape == (1, 3)
    assert xp.allclose(p_geo.ecef, brisbane_ecef, atol=1e-2)

    p_ecef = Point(brisbane_ecef, backend=xp)
    assert p_ecef.ecef.shape == (1, 3)
    assert xp.allclose(p_ecef.geodet, brisbane_geodet, atol=1e-4)
    assert p_ecef.geodet.shape == (1, 3)

    with pytest.raises(ValueError):
        Point(xp.array([[1, 2], [3, 4]]), backend=xp)

    multi_ecef = xp.array(
        [
            [-5046981.62, 2568681.41, -2924582.78],
            [-5046981.62, 2568681.41, -2924582.78],
        ]
    )
    p_multi = Point(multi_ecef, backend=xp)
    assert p_multi.ecef.shape == (2, 3)
    assert len(p_multi) == 2
    assert p_multi[0].ecef.shape == (1, 3) 
    assert (p_multi[0].ecef == p_multi.ecef[0:1]).all()


@pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_point_with_time(xp):
    brisbane_geodet = xp.array([[-27.4705, 153.0260, 27]])
    brisbane_ecef = xp.array([[-5046981.62, 2568681.41, -2924582.78]])
    time1 = Time.from_datetime(
        dt.datetime(2023, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc), backend=xp
    )
    time2 = Time.from_datetime(
        dt.datetime(2023, 1, 1, 12, 10, 0, tzinfo=dt.timezone.utc), backend=xp
    )
    time3 = Time(xp.array([time1.unix[0], time2.unix[0]]), backend=xp)

    p_geo = Point.from_geodet(brisbane_geodet, time=time1, backend=xp)
    assert p_geo.ecef.shape == (1, 3)
    assert xp.allclose(p_geo.ecef, brisbane_ecef, atol=1e-2)
    assert p_geo.time == time1

    p_ecef = Point(brisbane_ecef, time=time1, backend=xp)
    assert p_ecef.ecef.shape == (1, 3)
    assert xp.allclose(p_ecef.geodet, brisbane_geodet, atol=1e-4)
    assert p_ecef.geodet.shape == (1, 3)
    assert p_ecef.time == time1

    with pytest.raises(ValueError):
        Point(xp.array([[1, 2], [3, 4]]), time=time1, backend=xp)

    multi_ecef = xp.array(
        [
            [-5046981.62, 2568681.41, -2924582.78],
            [-5046981.62, 2568681.41, -2924582.78],
        ]
    )
    p_multi = Point(multi_ecef, time=time3, backend=xp)
    assert p_multi.ecef.shape == (2, 3)
    assert p_multi.time == time3

def test_bad_time_length(xp):
    with pytest.raises(ValueError):
        Point(
            xp.array([[-5046981.62, 2568681.41, -2924582.78]]),
            time=Time(xp.array([1, 2]), backend=xp),
            backend=xp,
        )

def test_nonmatching_backends(xp):
    if xp is not np:
        p1 = Point(
            np.array([[-5046981.62, 2568681.41, -2924582.78]]),
            backend=np,
        )
        p2 = Point(
            xp.array([[-5046981.62, 2568681.41, -2924582.78]]),
            backend=xp,
        )
        with pytest.raises(ValueError):
            p1 + p2
        with pytest.raises(ValueError):
            Point.from_list([p1, p2])


def test_conversion(xp):
    p1 = Point(
        xp.array([[-5046981.62, 2568681.41, -2924582.78]]),
        time=Time.from_datetime(
            dt.datetime(2023, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc), backend=xp
        ),
        backend=xp,
    )
    p1.convert_to(np)

def test_adding_points(xp):
    p1 = Point(
        xp.array([[-5046981.62, 2568681.41, -2924582.78]]),
        time=Time.from_datetime(
            dt.datetime(2023, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc), backend=xp
        ),
        backend=xp,
    )
    p2 = Point(
        xp.array([[-5046981.62, 2568681.41, -2924582.78]]),
        time=Time.from_datetime(
            dt.datetime(2023, 1, 1, 12, 10, 0, tzinfo=dt.timezone.utc), backend=xp
        ),
        backend=xp,
    )

    p3 = p1 + p2
    assert p3.ecef.shape == (2, 3)
    assert p3.time.unix.shape == (2,)

    with pytest.raises(ValueError):
        p1 + Point(
            xp.array([[-5046981.62, 2568681.41, -2924582.78]]),
            backend=xp,
        )


def test_point_from_list(xp):
    point1 = Point(
        xp.array([[-5046981.62, 2568681.41, -2924582.78]]),
        time=Time.from_datetime(
            dt.datetime(2023, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc), backend=xp
        ),
        backend=xp,
    )
    point2 = Point(
        xp.array([[-5046981.62, 2568681.41, -2924582.78]]),
        time=Time.from_datetime(
            dt.datetime(2023, 1, 1, 12, 10, 0, tzinfo=dt.timezone.utc), backend=xp
        ),
        backend=xp,
    )
    points = Point.from_list([point1, point2])
    assert points.ecef.shape == (2, 3)
    assert points.time.unix.shape == (2,)
