# pyright: basic

import pytest
import datetime as dt
from astrix.primitives import Point, Time


@pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_point_no_time(xp):
    brisbane_geodet = xp.array([[-27.4705, 153.0260, 27]])
    brisbane_ecef = xp.array([[-5046981.62, 2568681.41, -2924582.78]])

    p_geo = Point.from_geodet(brisbane_geodet, backend=xp)
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
    time3 = Time(xp.array([time1.secs[0], time2.secs[0]]), backend=xp)

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
    assert p3.time.secs.shape == (2,)

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
    assert points.time.secs.shape == (2,)
