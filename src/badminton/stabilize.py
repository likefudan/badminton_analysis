"""Undo camera motion: warp every frame onto one reference frame.

All later steps assume a fixed camera (one court calibration for the whole
video). A phone held in the hand or on a wobbly stand turns and shifts a few
degrees, so the court lines move by tens of pixels. For a camera that mostly
rotates (or moves little compared with the distance to the court) frames are
related by a homography, so each frame is mapped onto the reference with
one: ORB features of the frame are matched to the reference frame's,
RANSAC rejects the matches on moving people, and the frame is warped.

If a frame matches the reference poorly (e.g. a big swing), its homography is
chained through the previous frame instead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

FEAT_WIDTH = 960  # features are found on a downscaled copy (speed), homographies scaled back
MIN_INLIERS = 60


class _Matcher:
    def __init__(self, n_features: int = 4000, floor_from: float | None = None):
        """floor_from: only use features below this fraction of the image height.
        The court calibration is a floor homography, so when the camera also
        moves (not only turns) aligning the floor matters, and walls/ceiling at
        another depth would pull the fit away from it."""
        self.orb = cv2.ORB_create(n_features, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.floor_from = floor_from

    def features(self, frame):
        s = FEAT_WIDTH / frame.shape[1]
        g = cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        mask = None
        if self.floor_from is not None:
            mask = np.zeros(g.shape, np.uint8)
            mask[int(self.floor_from * g.shape[0]):] = 255
        k, d = self.orb.detectAndCompute(g, mask)
        pts = np.float32([p.pt for p in k]) / s if k else np.zeros((0, 2), np.float32)
        return pts, d

    def homography(self, src, dst):
        """H mapping points of `src` features onto `dst` features, and the inlier count."""
        (ps, ds), (pd, dd) = src, dst
        if ds is None or dd is None or len(ps) < 8 or len(pd) < 8:
            return None, 0
        pairs = self.bf.knnMatch(ds, dd, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < 0.8 * rest[0].distance]
        if len(good) < 12:
            return None, 0
        a = ps[[m.queryIdx for m in good]]
        b = pd[[m.trainIdx for m in good]]
        H, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0, maxIters=3000, confidence=0.999)
        return H, int(inl.sum()) if inl is not None else 0


def stabilize(video, out_path, ref_time: float, start_s: float = 0.0, duration_s: float | None = None,
              encoder: str = "libx264", progress=None, floor_from: float | None = None) -> dict:
    """Write a stabilized copy of [start, start+duration) aligned to the frame at `ref_time`.
    Returns per-frame homographies (frame -> reference) and inlier counts, also saved
    next to the output as <out>.stabilize.json."""
    from .video import VideoWriter, probe, read_frame, read_frames

    info = probe(video)
    fps = info.fps or 30.0
    m = _Matcher(floor_from=floor_from)
    ref = m.features(read_frame(video, int(round(ref_time * fps))))
    start = int(round(start_s * fps))
    n_max = int(round(duration_s * fps)) if duration_s else None
    writer = VideoWriter(out_path, info.width, info.height, fps,
                         audio_from=video if info.has_audio else None, audio_start_s=start / fps,
                         encoder=encoder)
    Hs, inliers, chained = [], [], []
    prev_feat, prev_H = None, np.eye(3)
    t0 = time.perf_counter()
    try:
        for n, (i, frame) in enumerate(read_frames(video, start, n_max), 1):
            f = m.features(frame)
            H, k = m.homography(f, ref)
            via_prev = False
            if H is None or k < MIN_INLIERS:
                H2, k2 = m.homography(f, prev_feat) if prev_feat is not None else (None, 0)
                if H2 is not None and k2 >= MIN_INLIERS:
                    H, k, via_prev = prev_H @ H2, k2, True
                else:
                    H, via_prev = prev_H, True  # keep the last good alignment
            warped = cv2.warpPerspective(frame, H, (info.width, info.height), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
            writer.write(warped)
            Hs.append(H.tolist())
            inliers.append(k)
            chained.append(via_prev)
            prev_feat, prev_H = f, H
            if progress:
                progress(n, n_max or max(0, info.frames - start), time.perf_counter() - t0)
    finally:
        writer.close()
    log = {"source": str(video), "ref_time_s": ref_time, "start_s": start_s, "fps": fps,
           "inliers": inliers, "chained": chained, "H": Hs}
    Path(out_path).with_suffix(".stabilize.json").write_text(json.dumps(log))
    return log
