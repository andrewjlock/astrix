# pyright: basic

import pytest
import datetime as dt
from astrix.primatives import Time, Point


def test_time(xp):
    times = [
        dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2020, 1, 2, tzinfo=dt.timezone.utc),
        dt.datetime(2020, 1, 3, tzinfo=dt.timezone.utc),
    ]
    t = Time.from_datetime(times, backend=xp)
    assert t.secs.shape == (3,)
    assert all(isinstance(dt_i, dt.datetime) for dt_i in t.datetime)
    assert t.is_in_bounds(t)
    assert t[1].datetime[0] == times[1]

    t2 = Time(xp.asarray([t.secs[0] - 1000, t.secs[-1] + 1000]), backend=xp)
    assert not t.is_in_bounds(t2)
    assert t2.is_in_bounds(Time(xp.asarray([t.secs[0], t.secs[-1]]), backend=xp))

    with pytest.raises(ValueError):
        Time(xp.asarray([[1, 2], [3, 4]]), backend=xp)

    times_naive = [dt.datetime(2020, 1, 1)]
    with pytest.raises(ValueError):
        Time.from_datetime(times_naive, backend=xp)

@pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_point(xp):
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


#
