# pyright: basic

import pytest
import numpy as np
from .helpers import to_text
from astrix import Ray, Time, Point, Path, Frame, RotationSequence
from scipy.spatial.transform import Rotation as R

@pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_single_ray_init(xp):
    origin = xp.array([1.0, 2.0, 3.0])
    direction = xp.array([0.0, 1.0, 0.0])
    ray1 = Ray(direction, origin, backend=xp)
    to_text(ray1)
    assert np.allclose(ray1.origin_points.ecef, origin)
    assert np.allclose(ray1.unit_rel, xp.array([0.0, 1.0, 0.0]))
    ray1.to_ecef()
    ray1.to_ned()
    ray1.az_el


    time = Time(0.0, backend=xp)
    ray2 = Ray(origin, direction, time=time, backend=xp)
    ray2.convert_to(np)


def test_bad_init(xp):
    origin = Point(xp.array([1.0, 2.0, 3.0]))
    direction = xp.array([0.0, 1.0])  # Incorrect shape

    with pytest.raises(ValueError):
        Ray(direction, origin, backend=xp)

    origin = xp.array([1.0, 2.0, 3.0])
    direction = xp.array([0.0, 0.0, 0.0])  # Zero vector
    with pytest.raises(ValueError):
        Ray(direction, origin, backend=xp)
#
    origin = xp.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0]])  # Multiple origins
    with pytest.raises(ValueError):
        Ray(direction, origin, backend=xp)


@pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_multiple_ray_init(xp):
    origins = xp.array([1.0, 2.0, 3.0])
    directions = xp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    ray1 = Ray(directions, origins, backend=xp)

    origins = xp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ray2 = Ray(directions, origins, backend=xp)

    to_text(ray1)
    to_text(ray2)
    assert np.allclose(ray2.origin_points.ecef, origins)
    ray2.to_ecef()
    ray2.to_ned()
    ray2.az_el
    assert len(ray1) == 2

    time_interp = Time(0.5, backend=xp)
    with pytest.raises(ValueError):
        ray2.interp(time_interp)

    time = Time([0.0, 1.0], backend=xp)
    ray3 = Ray(directions, origins, time=time, backend=xp)
    ray4 = ray3.interp(time_interp)
    assert len(ray4) == 1
    assert len(ray4.origin_points) == 1
    assert ray4.unit_rel.shape == (1, 3)
    assert np.allclose(ray4.origin_points.ecef, xp.array([[2.5, 3.5, 4.5]]))
    assert np.allclose(ray4.unit_rel, xp.array([[0.70710678, 0.70710678, 0.0]]))


@pytest.mark.filterwarnings("ignore:.*deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_ray_moving_frame(xp):
    time_path = Time(xp.array([0.0, 1.0, 2.0]), backend=xp)
    pt = Point(
        xp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        time_path,
        backend=xp,
    )
    path = Path(pt, backend=xp)
    rots = RotationSequence(
        R.from_euler("Z", [[0.0], [90.0], [180.0]], degrees=True), time_path, backend=xp
    )
    frame = Frame(rots, path, backend=xp)

    time_ray = Time(xp.array([0.0, 1.0, 2.0]), backend=xp)
    directions = xp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    ray = Ray(directions, time=time_ray, frame=frame, backend=xp)
    assert len(ray) == 3
    to_text(ray)
    assert np.allclose(ray.origin_points.ecef, pt.ecef)

    time_interp = Time([0.5, 2], backend=xp)
    ray_interp = ray.interp(time_interp)
    assert len(ray_interp) == 2
    assert np.allclose(
        ray_interp.origin_points.ecef, xp.array([[0.5, 0.0, 0.0], [0., 1., 0.0]])
    )
    assert np.allclose(ray_interp.unit_rel, xp.array([[0.70710678, 0.70710678, 0.0],
                                                     [0, 0.70710678, 0.70710678]]))

    ray_ecef = ray_interp.to_ecef()
    assert np.allclose(
        ray_ecef.origin_points.ecef, xp.array([[0.5, 0.0, 0.0], [0., 1., 0.0]])
    )
    assert np.allclose(ray_ecef.unit_rel, xp.array([[0.0, 1.0, 0.0],
                                                    [0, -0.70710678, 0.70710678]]))
