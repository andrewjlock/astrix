
import pytest
from astrix.functs import great_circle_distance, interp_haversine, ned_rotation
from astrix.utils import geodet2ecef, ecef2geodet

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


