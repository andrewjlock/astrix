
import pytest
import datetime as dt
from astrix import Time
from .helpers import to_text
import numpy as np

def test_time(xp):
    times = [
        dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2020, 1, 2, tzinfo=dt.timezone.utc),
        dt.datetime(2020, 1, 3, tzinfo=dt.timezone.utc),
    ]
    t = Time.from_datetime(times, backend=xp)
    to_text(t)
    assert len(times) == 3

    t2 = t.offset(1)
    t3 = t2.offset(-1)
    assert all(xp.isclose(t.unix, t3.unix))
    t = t.convert_to(np)

    assert t.unix.shape == (3,)
    assert all(isinstance(dt_i, dt.datetime) for dt_i in t.datetime)
    assert t.in_bounds(t)
    assert t[1].datetime[0] == times[1]

    t2 = Time(xp.asarray([t.unix[0] - 1000, t.unix[-1] + 1000]), backend=xp)
    assert not t.in_bounds(t2)
    assert t2.in_bounds(Time(xp.asarray([t.unix[0], t.unix[-1]]), backend=xp))

    with pytest.raises(ValueError):
        Time(xp.asarray([[1, 2], [3, 4]]), backend=xp)

    times_naive = [dt.datetime(2020, 1, 1)]
    with pytest.raises(ValueError):
        Time.from_datetime(times_naive, backend=xp)

