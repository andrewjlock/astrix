import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from astrix import Frame, Time, Point, Path, RotationSequence, TIME_INVARIANT
from .helpers import to_text


def test_static_frame(xp):
    # Static frame without time or reference

    rot1 = R.from_euler("z", [[45]], degrees=True)
    loc1 = Point(xp.asarray([1, 0, 0]), backend=xp)
    frame1 = Frame(rot1, loc1, backend=xp)
    to_text(frame1)

    time_group = frame1.time_group
    time_bounds = frame1.time_bounds

    assert not frame1.has_ref
    assert xp.allclose(frame1.interp_rot().as_quat(), rot1.as_quat())
    assert xp.allclose(frame1.interp_loc().ecef, loc1.ecef)

    t1 = Time(0.0, backend=xp)
    loc2 = Point(xp.asarray([1, 0, 0]), t1, backend=xp)

    with pytest.warns():
        frame2 = Frame(rot1, loc2, backend=xp)

    time_group2 = frame2.time_group
    time_bounds2 = frame2.time_bounds

    loc3 = Point(xp.asarray([1, 1, 0]), backend=xp) + loc1
    with pytest.raises(ValueError):
        frame3 = Frame(rot1, loc3, backend=xp)


def test_dynamic_frame(xp):
    # Dynamic frame without reference

    rot0 = R.from_euler("z", [[0]], degrees=True)
    rot1 = R.from_euler("z", [[45]], degrees=True)
    rot2 = R.from_euler("z", [[90]], degrees=True)

    rot_time = Time([0.0, 1.0], backend=xp)
    rots = RotationSequence([rot0, rot2], rot_time, backend=xp)

    loc = Point(xp.asarray([1, 0, 0]), backend=xp)

    frame1 = Frame(rots, loc, backend=xp)
    to_text(frame1)

    assert not frame1.has_ref
    assert xp.allclose(
        frame1.interp_rot(Time(0.5, backend=xp)).as_quat(), rot1.as_quat()
    )

    loc0 = Point(xp.asarray([1, 0, 0]), Time(0, xp), backend=xp)
    loc1 = Point(xp.asarray([0, 1, 0]), Time(1, xp), backend=xp)
    loc2 = Point(xp.asarray([-1, 0, 0]), Time(2, xp), backend=xp)

    path1 = Path([loc0, loc1, loc2], backend=xp)
    frame2 = Frame(rots, path1, backend=xp)

    assert xp.allclose(
        frame2.interp_rot(Time(0.5, backend=xp)).as_quat(), rot1.as_quat()
    )
    assert xp.allclose(
        frame2.interp_loc(Time(0.5, backend=xp)).ecef, xp.asarray([0.5, 0.5, 0])
    )

    assert xp.isclose(frame2.time_bounds[0].unix, 0.0)
    assert xp.isclose(frame2.time_bounds[1].unix, 1.0)

    path2 = Path([loc1, loc2], backend=xp)
    with pytest.raises(ValueError):
        # Non-overlapping time elements
        frame3 = Frame(rots, path2, backend=xp)


def test_frame_with_ref(xp):
    rot0 = R.from_euler("z", [[30]], degrees=True)
    rot1 = R.from_euler("z", [[45]], degrees=True)
    rot2 = R.from_euler("z", [[75]], degrees=True)
    loc0 = Point(xp.asarray([1, 0, 0]), backend=xp)
    loc1 = Point(xp.asarray([0, 1, 0]), backend=xp)
    loc2 = Point(xp.asarray([-1, 0, 0]), backend=xp)

    # Static frame with static reference
    frame_ref = Frame(rot0, loc1, backend=xp)
    frame1 = Frame(rot1, loc2, ref_frame=frame_ref, backend=xp)
    to_text(frame1)
    assert frame1.has_ref
    assert xp.allclose(frame1.interp_rot().as_quat(), rot2.as_quat())
    assert xp.allclose(frame1.interp_loc().ecef, xp.asarray([-1, 0, 0]))
    assert frame1.time_bounds[0] is TIME_INVARIANT
    assert frame1.time_bounds[1] is TIME_INVARIANT

    # Dynamic frame with static reference
    time = Time([0.0, 1.0], backend=xp)
    points = Point((loc1 + loc2).ecef, time, backend=xp)
    rots = RotationSequence([rot1, rot2], time, backend=xp)
    path = Path(points, backend=xp)
    frame2 = Frame(rots, path, ref_frame=frame_ref, backend=xp)

    assert frame2.has_ref
    rot_expected = R.from_euler("z", [[30 + 60]], degrees=True)
    assert xp.allclose(
        frame2.interp_rot(Time(0.5, backend=xp)).as_quat(), rot_expected.as_quat()
    )
    assert xp.allclose(
        frame2.interp_loc(Time(0.5, backend=xp)).ecef, xp.asarray([-0.5, 0.5, 0])
    )
    assert xp.isclose(frame2.time_bounds[0].unix, 0.0)
    assert xp.isclose(frame2.time_bounds[1].unix, 1.0)

    # Dynamic frame with dynamic reference

    rot1_ref = R.from_euler("z", [[0]], degrees=True)
    rot2_ref = R.from_euler("z", [[60]], degrees=True)
    rot1 = R.from_euler("y", [[30]], degrees=True)
    rot2 = R.from_euler("y", [[70]], degrees=True)
    time_rot = Time([1.0, 2.0], backend=xp)

    ecef = xp.asarray([[0, 1, 0], [-1, 0, 0]])
    time_point = Time([0.0, 3.0], backend=xp)

    rots_ref = RotationSequence([rot1_ref, rot2_ref], time_rot, backend=xp)
    rots = RotationSequence([rot1, rot2], time_rot, backend=xp)
    points = Point(ecef, time_point, backend=xp)
    path = Path(points, backend=xp)
    frame_ref2 = Frame(rots_ref, loc1, backend=xp)
    frame3 = Frame(rots, path, ref_frame=frame_ref2, backend=xp)
    to_text(frame3)
    assert frame3.rel_rot == rots

    assert frame3.has_ref
    rot_expected = R.from_euler("z", [[30]], degrees=True) * R.from_euler(
        "y", 50, degrees=True
    )
    assert xp.allclose(
        frame3.interp_rot(Time(1.5, backend=xp)).as_quat(), rot_expected.as_quat()
    )
    assert xp.allclose(
        frame3.interp_loc(Time(1.5, backend=xp)).ecef, xp.asarray([-0.5, 0.5, 0])
    )
    assert xp.isclose(frame3.time_bounds[0].unix, 1.0)
    assert xp.isclose(frame3.time_bounds[1].unix, 2.0)
    frame3.convert_to(np)
    frame3.convert_to(xp)

    # Non-overlapping time elements
    time2 = Time([3.0, 4.0], backend=xp)
    points2 = Point((loc1 + loc2).ecef, time2, backend=xp)
    path2 = Path(points2, backend=xp)
    with pytest.raises(ValueError):
        frame4 = Frame(rots, path2, ref_frame=frame_ref2, backend=xp)
