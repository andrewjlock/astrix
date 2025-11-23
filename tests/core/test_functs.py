import pytest
import warnings
import numpy as np
from scipy.spatial.transform import Rotation as R

from astrix.functs import (
    great_circle_distance,
    interp_haversine,
    ned_rotation,
    geodet2ecef,
    ecef2geodet,
    ensure_1d,
    ensure_2d,
    is_increasing,
    sort_by_time,
    interp_nd,
    interp_unit_vec,
    central_diff,
    finite_diff_2pt,
    az_el_from_vec,
    vec_from_az_el,
    total_angle_from_vec,
    apply_rot,
    refraction_correction_bennett,
    project_velocity_to_az_el,
)

def test_enforce1d(xp):
    x = 1.0
    y = ensure_1d(x, xp)
    assert y.shape == (1,)
    assert y[0] == x

    x = xp.asarray([1.0, 2.0, 3.0])
    y = ensure_1d(x, xp)
    assert y.shape == (3,)

def test_enforce2d(xp):
    x = 1.0
    y = ensure_2d(x,  backend=xp)
    assert y.shape == (1, 1)
    assert y[0, 0] == x

    x = [1.0, 2.0, 3.0]
    y = ensure_2d(x, 3, xp)
    assert y.shape == (1, 3)
    assert (y[0, :] == xp.asarray(x)).all()

    x = [[1.0, 2.0], [3.0, 4.0]]
    y = ensure_2d(x, 2, xp)
    assert y.shape == (2, 2)
    assert (y[:, :] == xp.asarray(x)).all()

    x = xp.asarray([1.0, 2.0, 3.0])
    y = ensure_2d(x, 3, xp)
    assert y.shape == (1, 3)
    assert (y[0, :] == x).all()

@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_great_circle_distance(xp):

    geodet1 = xp.asarray(
        [
            [0.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [0, 45.0, 0.0],
        ]
    )
    geodet2 = xp.asarray(
        [
            [0.0, 90.0, 0.0],
            [90.0, 180.0, 0.0],
            [0, -45.0, 0.0],
        ]
    )


    radius = 6371e3  # meters

    d = great_circle_distance(geodet1, geodet2, backend=xp)
    expected = xp.asarray([xp.pi / 2, 0.0, xp.pi / 2]) * radius
    assert xp.allclose(d, expected, atol=1e5)

@pytest.mark.filterwarnings("ignore:.*Force converting.*:UserWarning")
def test_interp_haversine(xp):
    geodet_data = xp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 90.0, 0.0],
            [0.0, 180.0, 0.0],
        ]
    )
    ecef_data = geodet2ecef(geodet_data)
    secs_data = xp.asarray([0.0, 1.0, 2.0])
    secs = xp.asarray([0.5, 1.5])

    ecef_interp = interp_haversine(secs, secs_data, ecef_data, backend=xp)
    geodet_interp = ecef2geodet(ecef_interp)
    expected = xp.asarray(
        [
            [0.0, 45.0, 0.0],
            [0.0, 135.0, 0.0],
        ]
    )
    assert xp.allclose(geodet_interp, expected, atol=1e-5)

def test_ned_rotation(xp):
    geodet = xp.asarray(
        [
            [0.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [0.0, 90.0, 0.0],
            [90.0, 90.0, 0.0],
            [45.0, 45.0, 0.0],
        ]
    ) # lat, lon, alt
    rot_ned = ned_rotation(geodet, xp=xp)
    # Check that the rotation matrices are orthogonal
    for r in rot_ned:
        rt = r.as_matrix().T
        identity = rt @ r.as_matrix()
        assert xp.allclose(identity, xp.eye(3), atol=1e-5)
        
    # Check a specific known rotations
    expected_2 = xp.asarray([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]).T
    assert xp.allclose(rot_ned[2].as_matrix(), expected_2, atol=1e-5)

    expected_3 = xp.asarray([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).T
    assert xp.allclose(rot_ned[3].as_matrix(), expected_3, atol=1e-5)

    r0 = rot_ned[0].as_matrix()
    expected0 = xp.asarray([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    assert xp.allclose(r0, expected0, atol=1e-5)


def test_is_increasing_and_sort(xp):
    arr = xp.asarray([1.0, 2.0, 3.0])
    assert is_increasing(arr, backend=xp)
    assert not is_increasing(xp.asarray([1.0, 1.0, 2.0]), backend=xp)
    with pytest.raises(ValueError):
        is_increasing(xp.asarray([[1.0, 2.0]]), backend=xp)

    times, data = sort_by_time(arr[::-1], xp.asarray([[3], [2], [1]]), backend=xp)
    assert xp.allclose(times, arr)
    assert xp.allclose(data[:, 0], xp.asarray([1, 2, 3]))


def test_interp_nd_and_unit_vec(xp):
    xd = xp.asarray([0.0, 1.0, 2.0])
    fd = xp.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    xq = xp.asarray([0.5, 1.5])
    out = interp_nd(xq, xd, fd, backend=xp)
    assert xp.allclose(out[0], xp.asarray([0.5, 0.5]))
    assert xp.allclose(out[1], xp.asarray([1.5, 0.5]))

    td = xp.asarray([0.0, 1.0])
    vecs = xp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mid = interp_unit_vec(xp.asarray([0.5]), td, vecs, backend=xp)
    expected = xp.asarray([[np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0]])
    assert xp.allclose(mid, expected, atol=1e-6)


def test_diff_operators(xp):
    t = xp.asarray([0.0, 1.0, 3.0, 6.0])
    f = (t**2)[:, xp.newaxis]

    central = central_diff(t, f, backend=xp)
    expected_central = xp.asarray([[0.0], [2.0], [6.0], [12.0]])
    assert xp.allclose(central, expected_central, atol=1e-6)

    finite = finite_diff_2pt(t, f, backend=xp)
    expected_finite = xp.asarray([[1.0], [3.0], [7.0], [9.0]])
    assert xp.allclose(finite, expected_finite, atol=1e-6)


def test_angles_and_vectors_roundtrip(xp):
    vecs = xp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    az_el = az_el_from_vec(vecs, backend=xp)
    back = vec_from_az_el(az_el, backend=xp)
    assert xp.allclose(back, vecs, atol=1e-6)

    totals = total_angle_from_vec(vecs, backend=xp)
    assert xp.allclose(totals, xp.asarray([0.0, 90.0, 90.0]), atol=1e-6)


def test_apply_rot_and_refraction(xp):
    if xp.__name__.startswith("jax"):
        pytest.skip("Rotation.as_matrix returns NumPy; skip inverse check on JAX backend")

    rot = R.from_euler("z", 90, degrees=True)
    vec = np.asarray([[1.0, 0.0, 0.0]])
    out = apply_rot(rot, vec, xp=np)
    assert np.allclose(out, np.asarray([[0.0, 1.0, 0.0]]))
    inv = apply_rot(rot, out, inverse=True, xp=np)
    assert np.allclose(inv, vec)

    correction = refraction_correction_bennett(np.asarray([10.0]), alt=0.0, backend=np)
    assert np.allclose(correction, 0.0)
    high_alt = refraction_correction_bennett(np.asarray([10.0]), alt=20000.0, backend=np)
    assert (high_alt > correction).all()


def test_project_velocity_to_az_el(xp):
    pos = xp.asarray([[1.0, 0.0, 0.0]])
    vel = xp.asarray([[0.0, 1.0, 0.0]])  # yawing right

    rates = project_velocity_to_az_el(pos, vel, backend=xp)
    assert rates.shape == (2, 1)
    assert xp.allclose(rates[1], 0.0, atol=1e-8)  # no elevation rate
    assert xp.allclose(rates[0], xp.asarray([57.2957795]), atol=1e-6)
