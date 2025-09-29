

import pytest
import numpy as np
from astrix import Frame, Time, Point, Path, RotationSequence, TIME_INVARIANT, Pixel, FixedZoomCamera, Ray
from scipy.spatial.transform import Rotation as R
from .helpers import to_text


def test_pixel_to_vector(xp):

    camera = FixedZoomCamera((1920, 1080), (36, 20), 50, backend=xp)

    point = Point(xp.asarray([10, 0, 0]), backend=xp)
    rot = R.from_euler("Y", 90, degrees=True)
    frame1 = Frame(rot, point, backend=xp)

    pixel_centre = Pixel((960, 540), backend=xp)
    ray = Ray.from_camera(pixel_centre, camera, frame1, backend=xp)

    assert xp.allclose(ray.unit_rel, xp.asarray([1, 0, 0]))
    ray_ecef = ray.to_ecef()
    assert xp.allclose(ray_ecef.unit_rel, xp.asarray([0, 0, -1]), atol=1e-6)

def test_vector_to_pixel(xp):

    camera = FixedZoomCamera((1920, 1080), (36, 20), 50, backend=xp)

    origin = Point(xp.asarray([10, 0, 0]), backend=xp)
    target= Point(xp.asarray([20, 10, 0]), backend=xp)
    rot = R.from_euler("Z", [[0],[90]], degrees=True)
    times = Time([0,1], backend=xp)
    rots = RotationSequence(rot, times, backend=xp)
    frame_cam = Frame(rots, origin, backend=xp)

    time_interp = Time(0.5, backend=xp)

    ray = Ray.from_points(origin, target, time=time_interp, backend=xp)
    ray_cam = ray.to_frame(frame_cam)
    pixel = ray_cam.project_to_cam(camera)

    assert xp.allclose(pixel.uv, xp.asarray([960, 540]), atol=1e-6)





