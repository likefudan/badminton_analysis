"""End-to-end run of step 1 with a stand-in for the neural networks."""

import json
import subprocess

import cv2
import numpy as np
import pytest

from badminton.cli import main
from badminton.court import CourtCalibration
from badminton.pipeline import SkeletonOptions, run_skeleton
from badminton.pose import PoseResult, provider_config, resolve_device
from badminton.video import probe
from conftest import H, W, fake_person, marker_pixels

SIX = ["near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner",
       "near_short_service_left", "near_short_service_right"]
N_FRAMES = 24
FPS = 30


@pytest.fixture
def clip(tmp_path, camera):
    """A 24-frame clip with the court lines drawn, so a human could calibrate it."""
    from badminton.render import draw_court

    calib = CourtCalibration(marker_pixels(camera, SIX), (W, H))
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N_FRAMES):
        f = np.full((H, W, 3), (40, 110, 40), np.uint8)
        draw_court(f, calib, (255, 255, 255), 2)
        w.write(f)
    w.release()
    return path, calib


class FakeEstimator:
    """Returns four players moving along the court plus one bystander."""

    def __init__(self, camera):
        self.camera = camera
        self.calls = 0
        self.rois = []

    def __call__(self, frame, roi=None):
        t = self.calls * 0.05
        self.calls += 1
        self.rois.append(roi)
        people = [(1.5 + t, 2.0), (4.5, 4.0 + t), (1.5, 9.0), (4.5 - t, 11.0), (-3.0, 5.0)]
        parts = [fake_person(self.camera, x, y) for x, y in people]
        return PoseResult(*(np.array(v) for v in zip(*parts)))


def test_run_skeleton_writes_video_and_data(clip, camera, tmp_path):
    video, calib = clip
    est = FakeEstimator(camera)
    out = tmp_path / "out.mp4"
    res = run_skeleton(video, est, out, tmp_path / "out.npz", calib, SkeletonOptions(stride=2))

    assert res["frames_read"] == N_FRAMES
    assert est.calls == N_FRAMES // 2
    assert est.rois[0] is not None  # detection limited to the court area

    info = probe(out)
    assert (info.width, info.height) == (W, H)
    assert info.codec == "h264"
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True, check=True).stdout.strip())
    assert n == N_FRAMES

    d = np.load(tmp_path / "out.npz")
    assert d["keypoints"].shape == (N_FRAMES // 2, 4, 17, 2)
    assert d["valid"].sum(1).tolist() == [4] * (N_FRAMES // 2)  # bystander dropped
    np.testing.assert_allclose(d["court_xy"][0][np.argsort(d["court_xy"][0][:, 1])],
                               [[1.5, 2.0], [4.5, 4.0], [1.5, 9.0], [4.5, 11.0]], atol=1e-6)
    np.testing.assert_allclose(d["time_s"][:3], [0, 2 / FPS, 4 / FPS])

    # overlay was actually drawn: output differs from input somewhere around a player
    cap = cv2.VideoCapture(str(out))
    ok, first = cap.read()
    assert ok
    box = fake_person(camera, 1.5, 2.0)[0].astype(int)
    patch = first[box[1]:box[3], box[0]:box[2]].astype(int)
    assert (np.abs(patch - (40, 110, 40)).sum(2) > 150).mean() > 0.02


def test_segment_and_singles(clip, camera, tmp_path):
    video, calib = clip
    est = FakeEstimator(camera)
    opts = SkeletonOptions(stride=1, start_s=10 / FPS, duration_s=6 / FPS, players_per_side=1)
    res = run_skeleton(video, est, None, tmp_path / "s.npz", calib, opts)
    d = np.load(tmp_path / "s.npz")
    assert res["frames_read"] == 6
    assert d["frame"].tolist() == list(range(10, 16))
    assert d["valid"].sum(1).tolist() == [2] * 6


def test_calibration_size_mismatch(clip, camera, tmp_path):
    video, _ = clip
    calib = CourtCalibration(marker_pixels(camera, SIX), (1920, 1080))
    with pytest.raises(ValueError):
        run_skeleton(video, FakeEstimator(camera), None, None, calib)


def test_cli_calibrate_with_points(clip, camera, tmp_path):
    video, _ = clip
    pts = marker_pixels(camera, SIX)
    spec = ";".join(f"{k}={x:.2f},{y:.2f}" for k, (x, y) in pts.items())
    out = tmp_path / "court.json"
    assert main(["calibrate", str(video), "--time", "0", "--points", spec, "-o", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["reprojection_error_px"] < 0.1
    assert (tmp_path / "court.png").exists()


def test_device_selection():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda", "coreml"}
    coreml = provider_config("coreml")
    assert coreml[0][0] == "CoreMLExecutionProvider" and coreml[-1] == "CPUExecutionProvider"
