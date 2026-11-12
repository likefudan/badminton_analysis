import numpy as np
import pytest

from badminton.court import LENGTH, MARKERS, WIDTH, CourtCalibration, detection_roi, in_play_area
from conftest import H, W, fake_person, marker_pixels

CORNERS = ["near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner"]
SIX = CORNERS + ["near_short_service_left", "near_short_service_right",
                 "far_short_service_right", "far_short_service_left"]


def test_four_exact_corners_recover_court(camera):
    calib = CourtCalibration(marker_pixels(camera, CORNERS), (W, H))
    assert calib.reprojection_error_px < 1e-3
    pts = np.array([[1.0, 2.0], [5.0, 12.0], [3.05, 6.7]])
    np.testing.assert_allclose(calib.to_court(camera.project(pts)), pts, atol=1e-6)


def test_more_points_reduce_far_court_error(camera):
    rng_points = np.array([[x, y] for x in (0.5, 3.0, 5.6) for y in (9.0, 11.0, 13.0)])  # far half
    errs = {}
    for label, names in (("4", CORNERS), ("all", list(MARKERS))):
        trials = []
        for seed in range(20):
            calib = CourtCalibration(marker_pixels(camera, names, noise_px=1.5, seed=seed), (W, H))
            est = calib.to_court(camera.project(rng_points))
            trials.append(np.hypot(*(est - rng_points).T).mean())
        errs[label] = np.mean(trials)
    assert errs["all"] < errs["4"]
    assert errs["all"] < 0.10  # metres, with 1.5 px click noise


def test_mis_click_shows_up_in_error(camera):
    pts = marker_pixels(camera, SIX)
    pts["far_left_corner"] = (pts["far_left_corner"][0] + 25, pts["far_left_corner"][1])
    calib = CourtCalibration(pts, (W, H))
    assert calib.reprojection_error_px > 2
    name, err_without = calib.suspect_point()
    assert name == "far_left_corner"
    assert err_without < 0.01
    assert CourtCalibration(marker_pixels(camera, SIX[:5]), (W, H)).suspect_point() is None


def test_rejects_bad_input(camera):
    with pytest.raises(ValueError):
        CourtCalibration(marker_pixels(camera, CORNERS[:3]), (W, H))
    with pytest.raises(ValueError):
        CourtCalibration({**marker_pixels(camera, CORNERS), "nope": (1, 1)}, (W, H))


def test_save_load_roundtrip(camera, tmp_path):
    calib = CourtCalibration(marker_pixels(camera, SIX), (W, H))
    calib.save(tmp_path / "c.json")
    again = CourtCalibration.load(tmp_path / "c.json")
    np.testing.assert_allclose(again.H, calib.H)
    assert again.frame_size == (W, H)


def test_play_area():
    pts = np.array([[3, 6], [-0.5, 0], [WIDTH + 2, 5], [3, LENGTH + 1.5], [3, -3]])
    assert in_play_area(pts).tolist() == [True, True, False, True, False]


def test_roi_contains_players_anywhere_on_court(camera):
    calib = CourtCalibration(marker_pixels(camera, SIX), (W, H))
    x0, y0, x1, y1 = detection_roi(calib)
    assert (x1 - x0) * (y1 - y0) < W * H  # actually crops something
    for x in (-0.5, 3.0, WIDTH + 0.5):
        for y in (-1.0, 6.0, 7.4, LENGTH + 1.0):
            box, kp, _ = fake_person(camera, x, y, height=1.9)
            vis = (kp[:, 0] >= 0) & (kp[:, 0] < W) & (kp[:, 1] >= 0) & (kp[:, 1] < H)
            k = kp[vis]
            assert (k[:, 0] >= x0).all() and (k[:, 0] <= x1).all(), (x, y)
            assert (k[:, 1] >= y0).all() and (k[:, 1] <= y1).all(), (x, y)
