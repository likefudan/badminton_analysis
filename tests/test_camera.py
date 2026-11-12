import numpy as np

from badminton.camera import camera_from_calibration
from badminton.court import MARKERS, CourtCalibration
from conftest import H, W, marker_pixels


def test_camera_recovered_from_floor_homography(camera):
    names = list(MARKERS)[:10]
    calib = CourtCalibration(marker_pixels(camera, names), (W, H))
    cam = camera_from_calibration(calib)
    assert abs(cam.focal_px - 700) < 2
    np.testing.assert_allclose(cam.center, camera.c, atol=0.02)
    for p in ([3.05, 6.7, 1.55], [0.0, 13.4, 5.0], [6.1, 2.0, 2.5]):
        np.testing.assert_allclose(cam.project(p)[0], camera.project(p)[0], atol=0.5)
