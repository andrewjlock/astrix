

import jax
from astrix import Path, Point, Time, Frame, RotationSequence
from scipy.spatial.transform import Rotation as R

def test_frame_interpolate(xp):
    
    ecef = xp.array([
        [3877000.0, 350000.0, 5027000.0],
        [3875000.0, 348000.0, 5029000.0],
        [3876000.0, 349000.0, 5028000.0],
    ])
    posix = xp.array([1577836920.0, 1577836800.0, 1577836860.0])  # 2020-01-01 00:02:00, 2020-01-01 00:00:00, 2020-01-01 00:01:00
    time = Time(posix, backend=xp)
    point = Point(ecef, time=time, backend=xp)
    path = Path(point, backend=xp)

    rot0 = R.from_euler("z", 0, degrees=True)
    rot1 = R.from_euler("z", 45, degrees=True)
    rot2 = R.from_euler("z", 90, degrees=True)

    rot_time = Time([1577836800.0, 1577836860.0], backend=xp)  # times for rot0 and rot2
    rots = RotationSequence([rot0, rot2], rot_time, backend=xp)

    frame = Frame(rots, path, backend=xp)

    t_interp = Time(xp.array([1577836830.0]), backend=xp)  # 2020-01-01 00:00:30

    point_interp = frame.interp_loc(t_interp)
    rot_interp = frame.interp_rot(t_interp)
    
    def myfunction(t):
        time = Time(t, backend=xp)
        return frame.interp_loc(time).ecef

    jacobian_loc = jax.jacobian(myfunction)(t_interp.secs)
    
    assert jacobian_loc.shape == (1, 3, 1)

    assert point_interp.ecef.shape == (1,3)
    correct = ecef[1] + (ecef[2] - ecef[1]) * 0.5
    assert xp.allclose(point_interp.ecef[0], correct, atol=1e-1)

    def myfunction_rot(t):
        time = Time(t, backend=xp)
        return frame.interp_rot(time, check_bounds=False).as_quat()

    jacobian_rot = jax.jacobian(myfunction_rot)(t_interp.secs)
    rot_jit = jax.jit(myfunction_rot)(t_interp.secs)

