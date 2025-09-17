
import pytest
import datetime as dt
from astrix.primitives import Time

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
