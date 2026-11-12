"""Automatic court calibration on rendered courts (no video needed)."""

import cv2
import numpy as np
import pytest

from badminton import autocal
from badminton.autocal import AutoCalibrationError, auto_calibrate, painted_lines
from badminton.court import LENGTH, MARKERS, WIDTH, CourtCalibration
from conftest import Camera

GAP = 1.6  # metres between neighbouring courts


def render(camera: Camera, courts=(0.0,), clutter: int = 0, seed: int = 0, line_px: int = 3) -> np.ndarray:
    """A green floor with white courts at the given x offsets, plus white clutter."""
    rng = np.random.default_rng(seed)
    w, h = camera.size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (70, 140, 60)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    for dx in courts:
        for a, b in painted_lines():
            seg = np.linspace(a, b, 40) + (dx, 0)
            uv = camera.project(seg).round().astype(np.int32)
            cv2.polylines(img, [uv], False, (235, 235, 235), line_px, cv2.LINE_AA)
    for _ in range(clutter):  # shoes, shuttles, bits of text
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if rng.random() < 0.5:
            cv2.circle(img, c, int(rng.integers(3, 14)), (240, 240, 240), -1)
        else:
            d = rng.integers(-40, 40, 2)
            cv2.line(img, c, (int(c[0] + d[0]), int(c[1] + d[1])), (240, 240, 240), 3)
    return cv2.GaussianBlur(img, (3, 3), 0)


def truth(camera: Camera, dx: float = 0.0) -> CourtCalibration:
    names = [n for n in MARKERS if not n.startswith("net")]
    uv = camera.project(np.array([MARKERS[n] for n in names]) + (dx, 0))
    return CourtCalibration({n: (float(u), float(v)) for n, (u, v) in zip(names, uv)}, camera.size)


def test_single_court_from_behind(camera):
    res = auto_calibrate(render(camera))
    mean, worst = autocal.intersection_distance_px(res.calib, truth(camera))
    assert mean < 1.0 and worst < 2.0
    assert res.confidence > 0.9 and res.lines_found == res.lines_visible >= 10


def test_axes_follow_the_convention(camera):
    """y = 0 is the base line next to the camera, x = 0 is on the camera's left."""
    calib = auto_calibrate(render(camera)).calib
    near_left, near_right, far_left = calib.to_image(np.array([(0, 0), (WIDTH, 0), (0, LENGTH)]))
    assert near_left[0] < near_right[0]
    assert near_left[1] > far_left[1]  # nearer = lower in the picture


def test_low_diagonal_camera_with_neighbours_and_clutter():
    """Like the phone on a tripod beside the base line: three courts side by side."""
    cam = Camera(pos=(-1.2, -1.5, 1.5), target=(4.0, 6.0, 0.0), f=620.0)
    step = WIDTH + GAP
    img = render(cam, courts=(-step, 0.0, step), clutter=40, seed=3)
    res = auto_calibrate(img)
    mean, worst = autocal.intersection_distance_px(res.calib, truth(cam), in_frame=True)
    assert mean < 2.0 and worst < 4.0


def test_hint_selects_the_neighbouring_court(camera):
    cam = Camera(pos=(6.9, -4.0, 4.0), target=(6.9, 6.5, 0.0), f=520.0)
    step = WIDTH + GAP
    img = render(cam, courts=(0.0, step), clutter=20, seed=1)
    for dx in (0.0, step):
        hint = cam.project([(dx + WIDTH / 2, 4.0)])[0]
        res = auto_calibrate(img, hint=(float(hint[0]), float(hint[1])))
        mean, worst = autocal.intersection_distance_px(res.calib, truth(cam, dx), in_frame=True)
        assert mean < 2.0 and worst < 4.0, dx


def test_refuses_without_a_court(camera):
    img = np.full((camera.size[1], camera.size[0], 3), 90, np.uint8)
    with pytest.raises(AutoCalibrationError):
        auto_calibrate(img)
    for seed in range(4):  # random long white strokes are not a court
        rng = np.random.default_rng(seed)
        strokes = img.copy()
        for _ in range(25):
            p = rng.integers(0, 700, 4)
            cv2.line(strokes, (int(p[0]) * 2, int(p[1])), (int(p[2]) * 2, int(p[3])), (240, 240, 240), 3)
        with pytest.raises(AutoCalibrationError):
            auto_calibrate(strokes)


def test_refuses_a_mostly_hidden_court(camera):
    img = render(camera)
    img[:, : camera.size[0] * 2 // 3] = (70, 140, 60)  # only the right third of the court is left
    with pytest.raises(AutoCalibrationError):
        auto_calibrate(img)


def test_saved_file_loads_as_a_normal_calibration(camera, tmp_path):
    res = auto_calibrate(render(camera))
    path = tmp_path / "court.json"
    res.calib.save(path)
    loaded = CourtCalibration.load(path)
    assert set(loaded.image_points) <= set(MARKERS) and len(loaded.image_points) >= 4
    assert loaded.reprojection_error_px < 0.05
    assert autocal.intersection_distance_px(loaded, res.calib)[1] < 0.05


def test_large_frames_are_handled_at_reduced_size():
    cam = Camera(f=2100.0, size=(3840, 2160))
    res = auto_calibrate(render(cam, line_px=8))
    assert res.calib.frame_size == (3840, 2160)
    assert autocal.intersection_distance_px(res.calib, truth(cam))[0] < 3.0


def test_track_follows_a_small_camera_shift(camera):
    first = auto_calibrate(render(camera)).calib
    moved = Camera(target=(3.2, 7.4, 0.0))  # a nudge of several pixels
    res = autocal.track(render(moved), first)
    assert autocal.intersection_distance_px(first, truth(moved))[0] > 3.0
    assert autocal.intersection_distance_px(res.calib, truth(moved))[0] < 1.0


def test_median_frame_removes_a_moving_player_but_not_a_moving_camera(camera):
    base = render(camera)
    frames = []
    for i in range(5):
        f = base.copy()
        cv2.rectangle(f, (300 + 120 * i, 300), (360 + 120 * i, 480), (250, 250, 250), -1)
        frames.append(f)
    med = autocal.median_frame(frames)
    assert np.abs(med.astype(int) - base).max() < 30
    shifted = [np.roll(base, 6 * i, axis=1) for i in range(5)]
    assert autocal.median_frame(shifted) is shifted[0]


def test_line_mask_ignores_wide_white_areas():
    img = np.full((360, 640, 3), (70, 140, 60), np.uint8)
    cv2.line(img, (20, 300), (600, 250), (235, 235, 235), 2)
    cv2.rectangle(img, (200, 50), (330, 180), (245, 245, 245), -1)  # a white shirt
    mask = autocal.line_mask(img)
    assert mask[275, 310] or mask[274, 310] or mask[276, 310]
    assert not mask[115, 265]
