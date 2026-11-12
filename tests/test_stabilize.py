import json

import cv2
import numpy as np

from badminton.stabilize import stabilize


def _write(path, frames, fps=30):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def test_bumped_camera_is_held_still(tmp_path):
    """A still camera bumped by 25 px at frame 30 (over 4 frames): the output must
    not move within the still stretches, and both stretches must line up."""
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur((rng.random((520, 720)) * 255).astype(np.uint8), (0, 0), 1.5)
    scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    frames = []
    for i in range(60):
        dx = 0 if i < 30 else (25 if i >= 34 else 25 * (i - 29) / 5)
        M = np.float32([[1, 0, 40 + dx], [0, 1, 40]])
        frames.append(cv2.warpAffine(scene, M, (640, 480)))
    src = tmp_path / "bump.mp4"
    _write(src, frames)
    out = tmp_path / "stab.mp4"
    log = stabilize(src, out, ref_time=0.0)
    assert len(log["still_segments"]) == 2
    cap = cv2.VideoCapture(str(out))
    got = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        got.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float))
    assert len(got) == 60
    centre = (slice(120, 360), slice(160, 480))
    first, last = got[5][centre], got[55][centre]
    assert np.abs(first - last).mean() < 6  # both still stretches aligned to the reference
    assert max(np.abs(got[k][centre] - got[k + 1][centre]).mean() for k in range(36, 58)) < 3
    saved = json.loads(out.with_suffix(".stabilize.json").read_text())
    assert len(saved["H"]) == 60
