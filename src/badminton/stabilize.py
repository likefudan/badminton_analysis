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
MAX_STEP_PX = 150  # a frame-to-frame alignment moving the corners more than this is a bad match
SNAP_PX = 40  # chain vs direct match disagreeing by more than this: trust the direct match
SNAP_MIN_INLIERS = 150


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


def _corners(H, w, h):
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None]
    return cv2.perspectiveTransform(c, H)[0]


def _from_corners(dst, w, h):
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.getPerspectiveTransform(c, np.float32(dst))


STILL_PX = 2.0  # frame-to-frame camera motion below this: the camera is still
MIN_STILL_S = 0.5
SEGMENT_SAMPLES = 9  # direct matches per still stretch, combined by the median


def stabilize(video, out_path, ref_time: float, start_s: float = 0.0, duration_s: float | None = None,
              encoder: str = "libx264", progress=None, floor_from: float | None = None) -> dict:
    """Write a stabilized copy of [start, start+duration) aligned to the frame at `ref_time`.

    A phone on a stand or a bench is still most of the time and gets bumped
    now and then. Matching every frame to the reference independently turns
    the matching noise into jitter (10 px median on the first user video,
    whose camera itself moved 0.2 px per frame). So:

      1. a first pass measures the camera motion between consecutive frames
         and splits the video into still stretches and bumps;
      2. each still stretch gets one alignment to the reference: the
         median (corner by corner) of up to SEGMENT_SAMPLES direct matches
         spread over the stretch, so matching noise averages out and the
         picture does not move at all within the stretch;
      3. frames during a bump follow the camera frame to frame and are blended
         linearly into the next stretch's alignment.

    Returns per-frame homographies (frame -> reference), also saved next to the
    output as <out>.stabilize.json."""
    from .video import VideoWriter, probe, read_frame, read_frames

    info = probe(video)
    fps = info.fps or 30.0
    W, Hh = info.width, info.height
    m = _Matcher(floor_from=floor_from)
    ref = m.features(read_frame(video, int(round(ref_time * fps))))
    start = int(round(start_s * fps))
    n_max = int(round(duration_s * fps)) if duration_s else None
    total = n_max or max(0, info.frames - start)
    t0 = time.perf_counter()

    # pass 1: camera motion between consecutive frames
    steps, feats_keep = [np.eye(3)], {}
    motion = [0.0]
    prev = None
    n = 0
    for n, (i, frame) in enumerate(read_frames(video, start, n_max), 1):
        f = m.features(frame)
        if prev is not None:
            st, k = m.homography(f, prev)
            mv = np.abs(_corners(st, W, Hh) - _corners(np.eye(3), W, Hh)).max() if st is not None else np.inf
            if st is None or k < MIN_INLIERS or not np.isfinite(mv) or mv > MAX_STEP_PX:
                st, mv = np.eye(3), np.inf  # unknown: treat as a bump
            steps.append(st)
            motion.append(float(mv))
        feats_keep[n - 1] = f
        prev = f
        if progress:
            progress(n, 2 * total, time.perf_counter() - t0)
    N = n
    motion = np.array(motion)
    still = motion < STILL_PX
    still[0] = still[1] if N > 1 else True
    # still stretches: runs of still frames long enough to average over
    segs = []
    k = 0
    while k < N:
        if still[k]:
            e = k
            while e + 1 < N and still[e + 1]:
                e += 1
            if (e - k + 1) / fps >= MIN_STILL_S:
                segs.append((k, e))
            k = e + 1
        else:
            k += 1
    if not segs:
        segs = [(0, N - 1)]
    seg_H = []
    for a, b in segs:
        idx = np.unique(np.linspace(a, b, min(SEGMENT_SAMPLES, b - a + 1)).astype(int))
        corners = []
        for j in idx:
            D, kk = m.homography(feats_keep[j], ref)
            if D is not None and kk >= MIN_INLIERS:
                corners.append(_corners(D, W, Hh))
        seg_H.append(_from_corners(np.median(corners, axis=0), W, Hh) if corners else None)
    # fill missing segment alignments from neighbours
    for q in range(len(seg_H)):
        if seg_H[q] is None:
            known = [r for r in range(len(seg_H)) if seg_H[r] is not None]
            seg_H[q] = seg_H[min(known, key=lambda r: abs(r - q))] if known else np.eye(3)
    feats_keep.clear()

    # per-frame alignment
    Hs = [None] * N
    for (a, b), Hq in zip(segs, seg_H):
        for j in range(a, b + 1):
            Hs[j] = Hq
    # before the first stretch: chain backwards; after the last: chain forwards
    for j in range(segs[0][0] - 1, -1, -1):
        Hs[j] = Hs[j + 1] @ np.linalg.inv(steps[j + 1])
    for j in range(segs[-1][1] + 1, N):
        Hs[j] = Hs[j - 1] @ steps[j]
    # bumps between stretches: chain forward from the left stretch, then blend into the right one
    for (a0, b0), (a1, b1) in zip(segs[:-1], segs[1:]):
        if a1 - b0 <= 1:
            continue
        chain = {b0: Hs[b0]}
        for j in range(b0 + 1, a1 + 1):
            chain[j] = chain[j - 1] @ steps[j]
        miss = _corners(Hs[a1], W, Hh) - _corners(chain[a1], W, Hh)
        for j in range(b0 + 1, a1):
            w = (j - b0) / (a1 - b0)
            Hs[j] = _from_corners(_corners(chain[j], W, Hh) + w * miss, W, Hh)
    Hs = [h / h[2, 2] for h in Hs]

    # pass 2: warp and write
    writer = VideoWriter(out_path, W, Hh, fps,
                         audio_from=video if info.has_audio else None, audio_start_s=start / fps,
                         encoder=encoder)
    try:
        for n2, (i, frame) in enumerate(read_frames(video, start, N), 1):
            writer.write(cv2.warpPerspective(frame, Hs[n2 - 1], (W, Hh), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT))
            if progress:
                progress(N + n2, 2 * total, time.perf_counter() - t0)
    finally:
        writer.close()
    log = {"source": str(video), "ref_time_s": ref_time, "start_s": start_s, "fps": fps,
           "still_segments": [[int(a), int(b)] for a, b in segs], "camera_motion_px": [round(x, 2) for x in motion.tolist()],
           "H": [h.tolist() for h in Hs]}
    Path(out_path).with_suffix(".stabilize.json").write_text(json.dumps(log))
    return log
