"""Geometry of scripts/recording_planner.py (no video needed)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recording_planner as rp  # noqa: E402

from badminton.camera import camera_from_calibration  # noqa: E402
from badminton.court import LENGTH, MARKERS, NET_Y, WIDTH, CourtCalibration  # noqa: E402
from badminton.tracking import measurement_cov  # noqa: E402

CX = WIDTH / 2
BEHIND = rp.Setup("behind", CX, -3.0, 3.0, "0.5x", "1080p", 60)


def test_camera_position_aim_and_field_of_view():
    cam = rp.make_camera(BEHIND)
    np.testing.assert_allclose(cam.center, [CX, -3.0, 3.0], atol=1e-9)
    # the aim point lands on the image centre; a level camera keeps verticals vertical
    np.testing.assert_allclose(cam.project([CX, NET_Y, 0.0])[0], [960, 540], atol=1e-6)
    top, bottom = cam.project([[CX, 10.0, 2.0], [CX, 10.0, 0.0]])
    assert abs(top[0] - bottom[0]) < 1e-6 and top[1] < bottom[1]
    assert cam.focal_px == pytest.approx(960 / math.tan(math.radians(54)))
    assert rp.pitch_deg(cam) == pytest.approx(math.degrees(math.atan2(3.0, 9.7)))
    # left of the court is left in the image
    assert cam.project([0.0, NET_Y, 0.0])[0, 0] < 960 < cam.project([WIDTH, NET_Y, 0.0])[0, 0]
    # 4K is the same view with twice the pixels
    cam4k = rp.make_camera(rp.replace(BEHIND, res="4k"))
    np.testing.assert_allclose(cam4k.project([1.0, 9.0, 2.0]), 2 * cam.project([1.0, 9.0, 2.0]), atol=1e-6)


def test_setup_round_trips_through_a_court_calibration():
    """Synthesize a camera, click the court markers in its image, recover the camera the
    way the pipeline does and describe it as a Setup again."""
    s = rp.Setup("tilted", 5.0, -2.5, 2.2, hfov_deg=90.0, aim=(2.0, 7.0, 0.5), roll_deg=2.0)
    cam = rp.make_camera(s)
    pts = {n: tuple(cam.project([(*xy, 0.0)])[0]) for n, xy in MARKERS.items()}
    back = rp.setup_from_camera(camera_from_calibration(CourtCalibration(pts, (1920, 1080))), "back")
    np.testing.assert_allclose([back.x, back.y, back.z], [5.0, -2.5, 2.2], atol=1e-3)
    assert back.hfov == pytest.approx(90.0, abs=0.02)
    assert back.roll_deg == pytest.approx(2.0, abs=0.02)
    cam2 = rp.make_camera(back)
    for p in ([0.0, 13.4, 0.0], [6.1, 3.0, 5.0], [3.0, 8.0, 1.5]):
        np.testing.assert_allclose(cam2.project(p), cam.project(p), atol=0.05)


def test_position_error_matches_the_pipelines_homography_jacobian():
    cam = rp.make_camera(BEHIND)
    pts = {n: tuple(cam.project([(*xy, 0.0)])[0]) for n, xy in MARKERS.items()}
    calib = CourtCalibration(pts, (1920, 1080))
    for xy in ((CX, 0.5), (1.0, NET_Y), (CX, LENGTH), (5.5, 12.0)):
        cov = measurement_cov(np.array([xy]), np.array([True]), calib)[0]
        p = rp.position_precision(cam, xy, sigma_px=4.0)
        assert p["err_m"] == pytest.approx(math.sqrt(np.trace(cov)), rel=1e-3)
        assert p["err_along_m"] == pytest.approx(math.sqrt(cov[1, 1]), rel=1e-3)
        assert p["err_across_m"] == pytest.approx(math.sqrt(cov[0, 0]), rel=1e-3)


def test_metres_per_pixel_closed_form():
    """Level camera, floor point straight ahead at distance d: one pixel across is d / f
    metres, one pixel along is d^2 / (f h) (h = camera height)."""
    h, d, f = 2.0, 12.0, 1000.0
    hfov = math.degrees(2 * math.atan(960 / f))
    cam = rp.make_camera(rp.Setup("level", CX, 0.0, h, hfov_deg=hfov, aim=(CX, 50.0, h)))
    p = rp.position_precision(cam, (CX, d), sigma_px=1.0)
    assert p["m_per_px_across"] == pytest.approx(d / f, rel=1e-3)
    assert p["m_per_px_along"] == pytest.approx(d * d / (f * h), rel=1e-2)
    # twice as high: half the along-court error
    hi = rp.make_camera(rp.Setup("level", CX, 0.0, 2 * h, hfov_deg=hfov, aim=(CX, 50.0, 2 * h)))
    assert rp.position_precision(hi, (CX, d), 1.0)["m_per_px_along"] == pytest.approx(d * d / (f * 2 * h), rel=1e-2)
    assert rp.person_px(cam, CX, d, 1.7) == pytest.approx(1.7 * f / d, rel=1e-6)
    assert rp.shuttle_px(cam, (CX, d, h)) == pytest.approx(rp.SHUTTLE_DIAMETER * f / d)
    assert rp.blur_px(cam, (CX, d, h), 60.0, 1 / 500) == pytest.approx(f * 60 / 500 / d)


def test_keypoint_sigma_model():
    hd, uhd = rp.make_camera(BEHIND), rp.make_camera(rp.replace(BEHIND, res="4k"))
    assert rp.kpt_sigma_px(hd, 50) == rp.kpt_sigma_px(hd, 500) == 4.0  # the pipeline's constant
    assert rp.kpt_sigma_px(uhd, 100) == 4.0 and rp.kpt_sigma_px(uhd, 1000) == 8.0
    # 4K never hurts and helps most where the player is small
    for xy in ((CX, 1.0), (CX, LENGTH)):
        assert rp.position_precision(uhd, xy)["err_m"] <= rp.position_precision(hd, xy)["err_m"] + 1e-9
    assert rp.position_precision(uhd, (CX, LENGTH))["err_m"] < 0.6 * rp.position_precision(hd, (CX, LENGTH))["err_m"]


def test_framing():
    wide = rp.make_camera(rp.best_aim(rp.Setup("w", CX, -3.0, 3.0, "0.5x")))
    assert rp.court_fits(wide, 0.3) and rp.player_coverage(wide) > 0.97
    assert rp.visible_height(wide, 0.0) > 2.5 and rp.visible_height(wide, NET_Y) > 6.0
    # the main lens two metres behind the baseline cannot hold the near corners
    tele = rp.make_camera(rp.best_aim(rp.Setup("t", CX, -2.0, 3.0, "1x")))
    assert not rp.court_fits(tele, 0.0)
    assert rp.visible_height(tele, 0.0) == 0.0
    # points behind the camera are never "in frame"
    assert not rp.in_frame(wide, [[CX, -10.0, 1.0]])[0]
    assert 0.0 < rp.airspace_fraction(tele) < rp.airspace_fraction(wide) <= 1.0


def test_partner_occlusion_drops_with_height():
    hidden = [rp.partner_hidden_m(rp.make_camera(rp.replace(BEHIND, z=z))) for z in (1.5, 2.5, 3.0, 4.0)]
    assert hidden[0] == 1.75  # camera below head height: fully hidden
    assert hidden == sorted(hidden, reverse=True) and hidden[-1] < 0.3
    # rear partner 4.5 m from the camera, which is 1.25 m above his head
    assert hidden[2] == pytest.approx(1.75 - 3.2 * 1.25 / 4.5)


def test_shuttle_flights_are_plausible():
    flights = {(n, side, bool(tr[0, 0] == CX)): tr for n, side, tr in rp.standard_flights()}
    clear = flights[("clear", "near", True)]
    assert 5.5 < clear[:, 2].max() < rp.AIR_HEIGHT
    assert clear[-1, 1] == pytest.approx(12.7, abs=0.05) and clear[-1, 2] <= 0
    v = np.linalg.norm(np.diff(clear, axis=0), axis=1) * rp.SIM_HZ
    assert v[-1] == pytest.approx(rp.TERMINAL_SPEED, abs=0.5)  # it falls at terminal speed
    smash = flights[("smash", "near", True)]
    vs = np.linalg.norm(np.diff(smash, axis=0), axis=1) * rp.SIM_HZ
    assert vs[0] == pytest.approx(70, abs=1) and vs[-1] < 15  # decelerates hard
    at_net = smash[np.argmin(abs(smash[:, 1] - NET_Y))]
    assert at_net[2] > 1.55  # clears the net
    far = flights[("clear", "far", True)]
    np.testing.assert_allclose(far[:, 1], LENGTH - clear[:, 1])


def test_frame_rate_and_shutter_scale_the_shuttle_numbers():
    cam = rp.make_camera(rp.best_aim(BEHIND))
    st = rp.flight_stats(cam)
    assert st["disp_px_30fps_median"] == pytest.approx(2 * st["disp_px_60fps_median"], rel=0.1)
    assert st["blur_px_1/120_p90"] == pytest.approx(st["blur_px_1/500_p90"] * 500 / 120)
    assert st["shuttle_in_frame"] > 0.9
    low = rp.flight_stats(rp.make_camera(rp.Setup("floor", CX, -2.0, 2.0, "1x")))  # aimed at the floor
    assert low["in_frame_by_shot"]["clear"] < 0.7 < low["in_frame_by_shot"]["smash"]
    assert rp.iso_needed(1 / 500, 2.2, 500) == pytest.approx(1210)
    assert rp.iso_needed(1 / 1000, 2.2, 500) == pytest.approx(2 * rp.iso_needed(1 / 500, 2.2, 500))


def test_airspace_mask_removes_no_neighbour_from_close_behind_and_some_from_far_away():
    near = rp.make_camera(rp.best_aim(BEHIND))
    seen, removed = rp.neighbour_removed(near, (WIDTH + 2.0, 0.0))
    assert seen > 0.3 and removed < 0.02
    tv = rp.make_camera(rp.Setup("tv", CX, -16.0, 5.7, hfov_deg=34.0, aim=(CX, 7.5, 0.0)))
    seen, removed = rp.neighbour_removed(tv, (WIDTH + 2.0, 0.0))
    assert seen > 0.05 and removed > 0.15


def test_grades_of_the_named_setups():
    rec, mini = (rp.evaluate_setup(rp.best_aim(s)) for s in rp.named_setups())
    assert rec["grade"] == "recommended" and mini["grade"] == "ok"
    assert rec["far_err_m"] < 0.30 and rec["far_player_px"] > 100 and rec["lines_fit"]
    # the first phone video's kind of setup: low, at the corner
    low = rp.evaluate_setup(rp.Setup("low", 6.75, -1.6, 1.4, hfov_deg=94.0, aim=(1.6, 7.0, 0.5)))
    assert low["grade"] == "poor" and low["far_err_m"] > 0.6
    assert rp.evaluate_setup(rp.replace(rp.best_aim(rp.named_setups()[0]), fps=30))["grade"] == "ok"
