
import pytest
import datetime as dt
from astrix import Time, TimeGroup, time_linspace
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


def test_time_invariant_and_bounds(xp):
    t_inv = Time.invariant(3, backend=xp)
    assert t_inv.is_invariant
    assert t_inv.is_singular is False  # length 3 invariant
    assert float(np.asarray(t_inv.duration)) == 0.0
    t_conv = t_inv.convert_to(np)
    assert t_conv.is_invariant
    assert t_inv.in_bounds(Time(xp.asarray([0.0]), backend=xp))
    with pytest.raises(ValueError):
        _ = t_inv.start_sec
    with pytest.raises(ValueError):
        _ = t_inv.end_sec


def test_time_group_invariant_and_overlap(xp):
    t1 = Time(xp.asarray([0.0, 1.0, 2.0]), backend=xp)
    t2 = Time(xp.asarray([0.5, 1.5]), backend=xp)
    tg = TimeGroup([t1, t2], backend=xp)
    assert not tg.is_invariant
    assert tg.in_bounds(Time(xp.asarray([0.75]), backend=xp))
    assert not tg.in_bounds(Time(xp.asarray([-1.0]), backend=xp))

    tg_inv = TimeGroup([Time.invariant(1, backend=xp)], backend=xp)
    assert tg_inv.is_invariant
    ob = tg_inv.overlap_bounds
    assert ob[0].is_invariant and ob[1].is_invariant


def test_time_linspace_rejects_invariant(xp):
    t_inv = Time.invariant(1, backend=xp)
    t_real = Time(xp.asarray([0.0, 1.0]), backend=xp)
    with pytest.raises(ValueError):
        time_linspace(t_inv, t_real, num=3)
