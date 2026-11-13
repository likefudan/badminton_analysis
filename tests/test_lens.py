"""Lens distortion: the model itself, and calibrating a court through it."""

import cv2
import numpy as np
import pytest
from conftest import Camera, marker_pixels
from test_autocal import render, truth

from badminton import autocal
from badminton.autocal import auto_calibrate
from badminton.camera import camera_from_calibration
from badminton.court import LENGTH, MARKERS, WIDTH, CourtCalibration, detection_roi
from badminton.lens import Lens, parse_lens, rectify

SIZE = (1280, 720)
BARREL = Lens(0.08)  # corners belong 8 % further out: a mild ultra-wide


def bend(img: np.ndarray, lens: Lens) -> np.ndarray:
    """What a camera with this lens records of a scene that `img` shows without distortion."""
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    src = lens.undistort(np.stack([gx.ravel(), gy.ravel()], 1), (w, h)).astype(np.float32)
    return cv2.remap(img, src[:, 0].reshape(h, w), src[:, 1].reshape(h, w), cv2.INTER_LINEAR)


def bent_truth(camera: Camera, lens: Lens) -> CourtCalibration:
    flat = truth(camera)
    pts = {n: tuple(lens.distort([p], camera.size)[0]) for n, p in flat.image_points.items()}
    return CourtCalibration(pts, camera.size, lens)


def floor_grid() -> np.ndarray:
    return np.array([(x, y) for x in np.linspace(0, WIDTH, 7) for y in np.linspace(0, LENGTH, 12)])


@pytest.mark.parametrize("lens", [Lens(0.08), Lens(-0.05), Lens(0.25, 0.1), Lens(0.5, 0.15)])
def test_distort_undoes_undistort(lens):
    rng = np.random.default_rng(0)
    px = rng.uniform((-200, -200), (SIZE[0] + 200, SIZE[1] + 200), (500, 2))  # also outside the picture
    np.testing.assert_allclose(lens.distort(lens.undistort(px, SIZE), SIZE), px, atol=1e-6)
    far = np.array([(-5000.0, 300.0), (9000.0, 4000.0)])  # court points far outside the frame
    np.testing.assert_allclose(lens.undistort(lens.distort(far, SIZE), SIZE), far, atol=1e-6)


def test_model_is_relative_to_the_picture_size():
    lens = Lens(0.1, 0.02)
    px = np.array([(100.0, 80.0), (1000.0, 650.0)])
    np.testing.assert_allclose(lens.undistort(px * 3, (3840, 2160)), lens.undistort(px, SIZE) * 3, atol=1e-9)
    np.testing.assert_allclose(lens.undistort([(0.0, 0.0)], SIZE), [(-0.12 * 640, -0.12 * 360)], atol=1e-9)
    centre = [(640.0, 360.0)]
    np.testing.assert_allclose(lens.undistort(centre, SIZE), centre)


def test_rejects_a_lens_that_folds_the_picture():
    with pytest.raises(ValueError):
        Lens(-0.6)
    with pytest.raises(ValueError):
        Lens(float("nan"))


def test_parse_lens():
    assert parse_lens("none") is None
    assert parse_lens("0") is None
    assert parse_lens("0.05") == Lens(0.05)
    assert parse_lens("0.05,-0.01") == Lens(0.05, -0.01)
    with pytest.raises(ValueError):
        parse_lens("wide")
    with pytest.raises(ValueError):
        parse_lens("1,2,3")


def test_rectify_makes_bent_lines_straight(camera):
    img = bend(render(camera), BARREL)
    flat, A, valid = rectify(img, BARREL)
    assert flat.shape[0] > img.shape[0] and flat.shape[1] > img.shape[1]  # grown to keep the corners
    assert valid[flat.shape[0] // 2, flat.shape[1] // 2] and not valid[0, 0]
    # the near long service line, low in the picture: bent in the recording, straight after rectifying
    ends = truth(camera).to_image(np.linspace((0, 0.76), (WIDTH, 0.76), 9))
    recorded = BARREL.distort(ends, camera.size)
    sag = recorded[4, 1] - (recorded[0, 1] + recorded[-1, 1]) / 2
    assert abs(sag) > 2
    q = np.linalg.solve(A, np.c_[ends, np.ones(len(ends))].T).T[:, :2]
    on_line = [flat[int(round(y)), int(round(x))].min() for x, y in q[1:-1]]
    assert min(on_line) > 150
    assert rectify(img, None)[0] is img


def test_calibration_through_a_lens(camera, tmp_path):
    names = list(MARKERS)
    recorded = {n: tuple(BARREL.distort([p], camera.size)[0]) for n, p in marker_pixels(camera, names).items()}
    grid = floor_grid()
    px = BARREL.distort(camera.project(grid), camera.size)
    with_lens = CourtCalibration(recorded, camera.size, BARREL)
    assert with_lens.reprojection_error_px < 1e-3
    np.testing.assert_allclose(with_lens.to_court(px), grid, atol=1e-5)
    np.testing.assert_allclose(with_lens.to_image(grid), px, atol=1e-4)
    # a plain homography on the same clicks is a compromise: wrong by pixels, and by centimetres on the floor
    plain = CourtCalibration(recorded, camera.size)
    assert plain.reprojection_error_px > 1.0
    assert np.linalg.norm(plain.to_court(px) - grid, axis=1).max() > 0.05

    with_lens.save(tmp_path / "c.json")
    again = CourtCalibration.load(tmp_path / "c.json")
    assert again.lens == BARREL
    np.testing.assert_allclose(again.to_court(px), grid, atol=1e-5)
    plain.save(tmp_path / "p.json")
    assert CourtCalibration.load(tmp_path / "p.json").lens is None
    assert CourtCalibration(recorded, camera.size, Lens()).lens is None  # k = 0 is no lens


def test_camera_projects_through_the_lens(camera):
    recorded = {n: tuple(BARREL.distort([p], camera.size)[0]) for n, p in marker_pixels(camera, list(MARKERS)).items()}
    cam = camera_from_calibration(CourtCalibration(recorded, camera.size, BARREL))
    np.testing.assert_allclose(cam.center, camera.c, atol=1e-3)
    pts = np.array([(1.0, 2.0, 0.0), (3.05, 6.7, 1.55), (5.0, 12.0, 4.0)])
    np.testing.assert_allclose(cam.project(pts), BARREL.distort(camera.project(pts), camera.size), atol=0.05)


def test_detection_roi_follows_the_bent_court(camera):
    strong = Lens(0.3)
    recorded = {n: tuple(strong.distort([p], camera.size)[0]) for n, p in marker_pixels(camera, list(MARKERS)).items()}
    calib = CourtCalibration(recorded, camera.size, strong)
    x0, y0, x1, y1 = detection_roi(calib)
    edge = calib.to_image(np.linspace((-1.0, 3.0), (WIDTH + 1.0, 3.0), 2))  # a row across the play area
    assert x0 <= edge[:, 0].min() and edge[:, 0].max() <= x1
    assert y0 <= edge[:, 1].min() and edge[:, 1].max() <= y1


def test_auto_leaves_an_undistorted_picture_alone(camera):
    img = render(camera)
    auto, plain = auto_calibrate(img), auto_calibrate(img, lens=None)
    assert auto.calib.lens is None
    assert auto.calib.image_points == plain.calib.image_points


@pytest.mark.parametrize("lens", [Lens(0.08), Lens(-0.05), Lens(0.3)])
def test_auto_finds_the_lens_from_the_court_lines(camera, lens):
    img = bend(render(camera), lens)
    res = auto_calibrate(img)
    assert res.calib.lens is not None
    want = bent_truth(camera, lens)
    grid = floor_grid()
    px = want.to_image(grid)
    err = np.linalg.norm(res.calib.to_image(grid) - px, axis=1)
    # the plain homography is off by several times as much
    try:
        plain_err = np.linalg.norm(auto_calibrate(img, lens=None).calib.to_image(grid) - px, axis=1).max()
    except autocal.AutoCalibrationError:
        plain_err = np.inf
    assert err.max() < 1.5 and err.max() < plain_err / 2
    assert np.linalg.norm(res.calib.to_court(px) - grid, axis=1).mean() < 0.03
    # a corner pixel is straightened to within 2 px of where the true lens puts it
    corner = [(0.0, 0.0)]
    assert np.linalg.norm(res.calib.undistort(corner) - want.undistort(corner)) < 2 + 0.1 * abs(lens.k1) * 734


def test_auto_with_a_known_lens_and_track(camera):
    img = bend(render(camera), BARREL)
    res = auto_calibrate(img, lens=BARREL)
    assert res.calib.lens == BARREL
    assert autocal.intersection_distance_px(res.calib, bent_truth(camera, BARREL))[0] < 1.5
    moved = Camera(target=(3.2, 7.4, 0.0))
    cur = autocal.track(bend(render(moved), BARREL), res.calib)
    assert cur.calib.lens == BARREL
    assert autocal.intersection_distance_px(cur.calib, bent_truth(moved, BARREL))[0] < 1.5
    conf, dist = autocal.mask_agreement(res.calib, img)
    assert conf > 0.9


def test_lens_from_a_clicked_court(camera):
    img = bend(render(camera), BARREL)
    clicks = marker_pixels(camera, list(MARKERS), noise_px=1.0)
    clicks = {n: tuple(BARREL.distort([p], camera.size)[0]) for n, p in clicks.items()}
    lens = autocal.estimate_lens(img, CourtCalibration(clicks, camera.size))
    assert lens is not None and abs(lens.k1 - BARREL.k1) < 0.02
    flat = render(camera)
    assert autocal.estimate_lens(flat, CourtCalibration(marker_pixels(camera, list(MARKERS), noise_px=1.0), camera.size)) is None
    assert autocal.estimate_lens(np.zeros_like(flat), truth(camera)) is None  # no lines to judge by


def test_cli_lens_option(camera, tmp_path, capsys):
    from badminton.cli import main

    img = bend(render(camera), BARREL)
    video = tmp_path / "court.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, camera.size)
    for _ in range(4):
        vw.write(img)
    vw.release()
    out = tmp_path / "a.court.json"
    assert main(["calibrate", str(video), "--auto", "--time", "0", "--frames", "1", "-o", str(out)]) == 0
    found = CourtCalibration.load(out).lens
    assert found is not None and abs(found.k1 - BARREL.k1) < 0.03
    assert "镜头畸变（从球场线估计）" in capsys.readouterr().out
    # reuse that lens for another video of the same phone, by file or by numbers
    for spec in (str(out), f"{BARREL.k1}"):
        other = tmp_path / "b.court.json"
        assert main(["calibrate", str(video), "--auto", "--time", "0", "--frames", "1", "--lens", spec, "-o", str(other)]) == 0
        assert CourtCalibration.load(other).lens is not None
    assert main(["calibrate", str(video), "--auto", "--time", "0", "--frames", "1", "--lens", "none", "-o", str(other)]) == 0
    assert CourtCalibration.load(other).lens is None
    assert main(["calibrate", str(video), "--auto", "--lens", "wide"]) == 2
    # hand-clicked points: the lens still comes from the lines
    clicks = bent_truth(camera, BARREL).image_points
    spec = ";".join(f"{n}={x:.2f},{y:.2f}" for n, (x, y) in clicks.items())
    assert main(["calibrate", str(video), "--time", "0", "--points", spec, "-o", str(other)]) == 0
    assert abs(CourtCalibration.load(other).lens.k1 - BARREL.k1) < 0.03


def test_4k_frame_with_a_lens():
    cam = Camera(f=2100.0, size=(3840, 2160))
    img = bend(render(cam, line_px=8), BARREL)
    res = auto_calibrate(img)
    assert res.calib.frame_size == (3840, 2160)
    assert abs(res.calib.lens.k1 - BARREL.k1) < 0.01  # the same numbers as at 1280 x 720
    assert autocal.intersection_distance_px(res.calib, bent_truth(cam, BARREL))[0] < 4.0
