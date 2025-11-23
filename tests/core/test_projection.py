

import pytest
import numpy as np
from astrix import Frame, Time, Point, Path, RotationSequence, TIME_INVARIANT, Pixel, FixedZoomCamera, Ray
from scipy.spatial.transform import Rotation as R
from .helpers import to_text


def test_pixel_to_vector(xp):

    camera = FixedZoomCamera.from_foc_len((1920, 1080), (36, 20), 50, backend=xp)

    point = Point(xp.asarray([10, 0, 0]), backend=xp)
    rot = R.from_euler("Y", 90, degrees=True)
    frame1 = Frame(rot, point, backend=xp)

    pixel_centre = Pixel((960, 540), backend=xp)
    ray = Ray.from_camera(pixel_centre, camera, frame1, backend=xp)

    assert xp.allclose(ray.unit_rel, xp.asarray([1, 0, 0]))
    ray_ecef = ray.to_ecef()
    assert xp.allclose(ray_ecef.unit_rel, xp.asarray([0, 0, -1]), atol=1e-6)

def test_vector_to_pixel(xp):

    camera = FixedZoomCamera.from_foc_len((1920, 1080), (36, 20), 50, backend=xp)

    origin = Point(xp.asarray([10, 0, 0]), backend=xp)
    target= Point(xp.asarray([20, 10, 0]), backend=xp)
    rot = R.from_euler("Z", [[0],[90]], degrees=True)
    times = Time([0,1], backend=xp)
    rots = RotationSequence(rot, times, backend=xp)
    frame_cam = Frame(rots, origin, backend=xp)

    time_interp = Time(0.5, backend=xp)

    ray = Ray.from_points(target, origin, time=time_interp, backend=xp)
    ray_cam = ray.to_frame(frame_cam)
    pixel = ray_cam.project_to_cam(camera)

    assert xp.allclose(pixel.uv, xp.asarray([960, 540]), atol=1e-6)


def test_fixed_zoom_camera_radial_and_convert(xp):
    camera = FixedZoomCamera.from_foc_len(
        (640, 480), (12, 9), 24, rad_coef=[0.01, -0.001], backend=xp
    )

    assert camera.has_dist
    assert xp.allclose(camera.rad_coef(), xp.asarray([0.01, -0.001]))

    camera_np = camera.convert_to(np)
    assert camera_np.backend == "numpy"
    assert np.allclose(camera_np.rad_coef(), np.asarray([0.01, -0.001]))


def test_fixed_zoom_camera_requires_distortion(xp):
    camera = FixedZoomCamera.from_foc_len((320, 240), (8, 6), 12, backend=xp)
    assert not camera.has_dist
    with pytest.raises(ValueError):
        camera.rad_coef()


