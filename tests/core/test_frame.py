import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from astrix import Frame, Time
from .helpers import to_text


def test_frame(xp):
    rot0 = R.from_euler("z", 0, degrees=True)
    rot1 = R.from_euler("z", 45, degrees=True)
    rot2 = R.from_euler("z", 90, degrees=True)

    # Test static, singular frame
    frame1 = Frame(rot1, backend=xp)
    to_text(frame1)

    assert frame1.is_static
    assert not frame1.has_ref
    frame1.convert_to(np)

    # Test static, relative frame
    frame2 = Frame(rot1, ref_frame=frame1, backend=xp)
    assert frame2.is_static
    assert frame2.has_ref
    assert xp.allclose(frame2.static_rot.as_quat(), rot2.as_quat())
    assert xp.allclose(frame2.rel_rot.as_quat(), rot1.as_quat())

    # Test dynamic, static frame
    time = Time([0.0, 1.0], backend=xp)
    frame3 = Frame(R.concatenate([rot0, rot2]), time=time, backend=xp)
    time_interp = Time([0.5], backend=xp)
    rot_interp = frame3.interp(time_interp)
    assert xp.allclose(rot_interp.as_quat(), rot1.as_quat())

    # Test dynamic, relative frame
    frame4 = Frame(R.concatenate([rot0, rot2]), time=time, ref_frame=frame1, backend=xp)
    rot_interp = frame4.interp(time_interp)
    assert xp.allclose(rot_interp.as_quat(), rot2.as_quat())

    # Test dynamic, relative frame with dynamic reference
    frame5 = Frame(R.concatenate([rot0, rot2]), time=time, ref_frame=frame3, backend=xp)
    rot_interp = frame5.interp(time_interp)
    assert xp.allclose(rot_interp.as_quat(), rot2.as_quat())
    frame5.convert_to(np)

    # Test errors
    with pytest.raises(ValueError):
        Frame(R.concatenate([rot0, rot2]), ref_frame=frame3, backend=xp)
    with pytest.raises(ValueError):
        Frame(rot1, time=time, backend=xp)
    with pytest.raises(ValueError):
        Frame(R.concatenate([rot0, rot2, rot1]), time=time, backend=xp)
    with pytest.raises(ValueError):
        frame5.static_rot

    # Test non-increasing time error
    time_bad = Time([0.0, 1.0, 0.5], backend=xp)
    with pytest.raises(ValueError):
        Frame(R.concatenate([rot0, rot2, rot1]), time=time_bad, backend=xp)

    # Test out of bounds time interpolation
    if frame2.backend == "jax.numpy":
        with pytest.warns():
            frame3.interp(Time([-1.0], backend=xp))

    if frame2.backend == "numpy": # SciPy numpy Rotation backend has it's own error
        with pytest.raises(ValueError):
            with pytest.warns():
                frame3.interp(Time([-1.0], backend=xp))

    to_text(frame1)
