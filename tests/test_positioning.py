import json

import numpy as np

from badminton.cli import main
from badminton.court import CourtCalibration
from badminton.positioning import _debounced_changes, analyse_pairs, pair_report
from badminton.tracking import Track, Tracking
from conftest import H, W, marker_pixels
from test_tracking import SIX, make_data

FPS = 15.0


def path(keypoints, t):
    """Piecewise-linear path through (time, x, y) keypoints."""
    kt, kx, ky = np.array(keypoints).T
    return np.stack([np.interp(t, kt, kx), np.interp(t, kt, ky)], 1)


def doubles_script(t):
    """Near pair: front-back with one front/back swap at ~3 s, then side-by-side with
    one left/right swap at ~9 s. Far pair: side-by-side all the time."""
    a = path([(0, 3.05, 5.4), (2.7, 3.05, 5.4), (3.3, 3.2, 1.6), (5.6, 3.2, 1.6), (6.2, 1.5, 3.0),
              (8.7, 1.5, 3.0), (9.3, 4.6, 3.2), (12, 4.6, 3.2)], t)
    b = path([(0, 3.1, 1.6), (2.7, 3.1, 1.6), (3.3, 2.9, 5.3), (5.6, 2.9, 5.3), (6.2, 4.6, 3.0),
              (8.7, 4.6, 3.0), (9.3, 1.5, 2.8), (12, 1.5, 2.8)], t)
    # a little footwork so speeds are realistic and the rally counts as in play
    wob = 0.35 * np.stack([np.sin(7 * t), np.cos(5 * t)], 1)
    c = np.stack([np.full_like(t, 1.5), np.full_like(t, 10.0)], 1) + wob
    d = np.stack([np.full_like(t, 4.6), np.full_like(t, 10.2)], 1) - wob
    return a + wob, b - wob, c, d


def test_formation_rotation_spacing_on_known_script():
    t = np.arange(0, 12, 1 / FPS)
    a, b, _, _ = doubles_script(t)
    rep = pair_report(a, b, 0, t, np.ones(len(t), bool), (1, 2), ("A", "B"))
    f = rep["formation_pct"]
    assert 35 < f["front_back"] < 50 and 35 < f["side_by_side"] < 50
    assert abs(sum(f.values()) - 100) < 0.5
    assert rep["rotations"]["front_back_swaps"] == 1 and rep["rotations"]["left_right_swaps"] == 1
    assert 2.9 < rep["rotations"]["front_back_swap_times_s"][0] < 3.4
    assert 3.0 < rep["spacing_m"]["median"] < 4.2
    assert abs(sum(rep["coverage_pct"].values()) - 100) < 0.5
    assert abs(sum(rep["open_court"]["biggest_gap_zone_pct"].values()) - 100) < 0.5
    assert rep["open_court"]["mean_gap_m"] > 1.0
    # A is in front for the first ~2.7 s and behind for the next ~2.3 s of front-back time
    assert 45 < rep["front_time_pct"]["A"] < 65


def test_gap_zone_follows_the_hole():
    """Both players stuck in the back right corner -> the biggest hole is front left."""
    t = np.arange(0, 3, 1 / FPS)
    a = np.tile([[5.0, 1.0]], (len(t), 1))
    b = np.tile([[4.0, 1.5]], (len(t), 1))
    rep = pair_report(a, b, 0, t, np.ones(len(t), bool), (1, 2), ("A", "B"))
    z = rep["open_court"]["biggest_gap_zone_pct"]
    assert z["front_left"] == 100.0
    assert rep["spacing_m"]["too_close_pct"] == 100.0
    # mirrored for the far pair: their front-left is the camera's front-right of their half
    rep = pair_report(np.tile([[1.0, 12.4]], (len(t), 1)), np.tile([[2.0, 11.9]], (len(t), 1)), 1, t,
                      np.ones(len(t), bool), (3, 4), ("C", "D"))
    assert rep["open_court"]["biggest_gap_zone_pct"]["front_left"] == 100.0


def test_short_flickers_are_not_rotations():
    t = np.arange(0, 6, 1 / FPS)
    role = (t > 3).astype(int)
    role[(t > 1) & (t < 1.2)] = 1  # 0.2 s flicker
    assert len(_debounced_changes(role, np.ones(len(t), bool), t, 0.5)) == 1


def test_no_pair_and_no_overlap():
    t = np.arange(0, 2, 1 / FPS)
    F = len(t)
    solo = Track(1, 0, np.zeros((F, 2)), np.ones((F, 2)), np.ones(F, bool), np.zeros(F, bool), np.zeros(F, int))
    assert analyse_pairs(Tracking(t, np.arange(F), [solo]), np.ones(F, bool)) == []
    rep = pair_report(np.full((F, 2), np.nan), np.ones((F, 2)), 0, t, np.ones(F, bool), (1, 2), ("A", "B"))
    assert "formation_pct" not in rep and rep["analysed_s"] == 0


def test_cli_movement_and_positioning_on_synthetic_doubles(camera, tmp_path):
    calib = CourtCalibration(marker_pixels(camera, SIX), (W, H))
    t = np.arange(0, 12, 1 / FPS)
    a, b, c, d = doubles_script(t)
    data = make_data(camera, calib, [[(*p[i], 0.0) for p in (a, b, c, d)] for i in range(len(t))])
    np.savez(tmp_path / "doubles.skeleton.npz", **data)
    calib.save(tmp_path / "court.json")
    common = [str(tmp_path / "doubles.mp4"), "--court", str(tmp_path / "court.json"),
              "--data", str(tmp_path / "doubles.skeleton.npz"), "--no-appearance"]
    assert main(["movement", *common, "-o", str(tmp_path)]) == 0
    assert main(["positioning", *common, "-o", str(tmp_path / "pos")]) == 0
    reps = json.loads((tmp_path / "doubles.positioning.json").read_text())
    assert [r["side"] for r in reps] == ["near", "far"]
    near = reps[0]
    assert near["rotations"]["front_back_swaps"] == 1 and near["rotations"]["left_right_swaps"] == 1
    assert reps[1]["formation_pct"]["side_by_side"] > 90
    assert (tmp_path / "doubles.positioning.near.png").exists()
    mv = json.loads((tmp_path / "doubles.movement.json").read_text())
    assert len(mv["players"]) == 4
