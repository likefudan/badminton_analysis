"""Step 2a: give every player a stable id and a clean floor position over time.

Input is step 1's .npz (per processed frame: up to P people with keypoints,
foot point on the court, side of the net). Output is one track per player:
ground position in metres for every processed frame, with

  * ids kept through crossings: each player stays on their own half, and
    detections are matched to tracks by predicted position (constant
    velocity) plus, when the video is available, the colour of the torso;
  * jumps removed: while a player is in the air the ankles are above the
    floor, so the floor homography puts the "foot point" further from the
    camera than the player really is. Those frames are detected and the
    ground position is interpolated across them;
  * short gaps (missed detections, airborne frames) up to MAX_GAP_S bridged
    by interpolation; longer gaps stay NaN and break the track.

Airborne test. For an upright player the image height from shoulders to
ankles divided by the floor's pixels-per-metre at the foot point is nearly
constant (both scale with distance from the camera). In the air the foot
point is projected too far away, where the floor's scale is smaller, so the
ratio jumps up; crouching and lunging make it smaller, never larger. A
frame whose ratio is clearly above the player's recent upright level is
treated as airborne. With a high broadcast camera this catches only part of
each jump (a jump moves the projected point little there). A speed gate (no one runs faster than MAX_SPEED) catches what
the ratio misses, e.g. with a high broadcast camera where a jump moves the
projected point only a little.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

import numpy as np

from .court import CourtCalibration

MAX_SPEED = 8.0  # m/s: faster than any badminton player moves on court
MAX_GAP_S = 0.8  # bridge missing / airborne stretches up to this long (a jump smash with take-off and landing ~0.7 s)
AIRBORNE_RATIO = 1.06  # body-height/floor-scale ratio above the player's upright level by this factor
_SHOULDERS, _HIPS, _ANKLES = (5, 6), (11, 12), (15, 16)
_KPT_THR = 0.3


@dataclass
class Track:
    id: int
    side: int  # 0 near, 1 far
    raw_xy: np.ndarray  # (F, 2) foot point as detected, NaN when not seen
    xy: np.ndarray  # (F, 2) cleaned ground position, NaN in long gaps
    seen: np.ndarray  # (F,) detected in this frame
    airborne: np.ndarray  # (F,) detected but judged off the ground (or implausible jump)
    det_index: np.ndarray  # (F,) row in the step-1 arrays, -1 when not seen
    name: str = ""


@dataclass
class Tracking:
    time_s: np.ndarray  # (F,)
    frame: np.ndarray  # (F,)
    tracks: list[Track] = field(default_factory=list)
    cuts: np.ndarray | None = None  # (F,) edit cut before this frame

    def by_id(self, tid: int) -> Track:
        return next(t for t in self.tracks if t.id == tid)


def load_npz(path: str | Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# --- appearance ---------------------------------------------------------------


def video_features(video: str | Path, data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read the processed frames once and return
      * (F, P, D) normalised HSV histogram of each person's torso, NaN if not visible;
      * (F,) True where the picture changes completely from the previous processed
        frame (an edit cut: tracks must not be joined across it).
    """
    import cv2

    from .video import read_frames

    F, P = data["valid"].shape
    out = np.full((F, P, 16 * 8), np.nan, np.float32)
    thumbs = np.zeros((F, 36, 64), np.float32)
    want = {int(f): i for i, f in enumerate(data["frame"])}
    start, stop = int(data["frame"][0]), int(data["frame"][-1]) + 1
    for fidx, img in read_frames(video, start, stop - start):
        i = want.get(fidx)
        if i is None:
            continue
        thumbs[i] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
        hsv = None
        for p in np.flatnonzero(data["valid"][i]):
            box = _torso_box(data["keypoints"][i, p], data["scores"][i, p], img.shape)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            if hsv is None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv[y0:y1, x0:x1]], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
            out[i, p] = h / max(h.sum(), 1e-6)
    return out, scene_cuts(thumbs)


def scene_cuts(thumbs: np.ndarray, factor: float = 4.0, min_diff: float = 20.0) -> np.ndarray:
    """(F,) True at frames that differ from the previous one far more than usual."""
    cuts = np.zeros(len(thumbs), bool)
    if len(thumbs) < 3:
        return cuts
    d = np.abs(np.diff(thumbs, axis=0)).mean(axis=(1, 2))
    cuts[1:] = (d > factor * np.median(d)) & (d > min_diff)
    return cuts

def _torso_box(kp, sc, shape):
    idx = [j for j in _SHOULDERS + _HIPS if sc[j] >= _KPT_THR]
    if len(idx) < 3:
        return None
    pts = kp[idx]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    pad = 0.1 * (x1 - x0)
    x0, x1 = int(max(0, x0 + pad)), int(min(shape[1], x1 - pad))
    y0, y1 = int(max(0, y0)), int(min(shape[0], y1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return x0, y0, x1, y1


def _hist_distance(a, b) -> float:
    """Bhattacharyya-style distance in [0, 1]; 0.5 when either is unknown."""
    if a is None or b is None or np.isnan(a).any() or np.isnan(b).any():
        return 0.5
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)))))


# --- association ------------------------------------------------------------------


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost matching for tiny matrices (<= 4 x 4): try every permutation."""
    n_t, n_d = cost.shape
    best, best_pairs = np.inf, []
    if n_t >= n_d:
        for perm in permutations(range(n_t), n_d):
            c = sum(cost[t, d] for d, t in enumerate(perm))
            if c < best:
                best, best_pairs = c, [(t, d) for d, t in enumerate(perm)]
    else:
        for perm in permutations(range(n_d), n_t):
            c = sum(cost[t, d] for t, d in enumerate(perm))
            if c < best:
                best, best_pairs = c, list(enumerate(perm))
    return best_pairs


class _Live:
    def __init__(self, side: int, F: int):
        self.side = side
        self.det = np.full(F, -1)
        self.last_xy = None
        self.last_t = None
        self.vel = np.zeros(2)
        self.hist = None

    def predict(self, t: float) -> np.ndarray:
        dt = min(t - self.last_t, MAX_GAP_S)
        return self.last_xy + self.vel * dt

    def update(self, xy, t, hist):
        if self.last_xy is not None and self.last_t is not None and t > self.last_t:
            v = (xy - self.last_xy) / (t - self.last_t)
            speed = np.linalg.norm(v)
            if speed > MAX_SPEED:  # a jump or a cut: don't let it drive the prediction
                v = np.zeros(2)
            self.vel = 0.5 * self.vel + 0.5 * v
        self.last_xy, self.last_t = np.asarray(xy, float), t
        if hist is not None and not np.isnan(hist).any():
            self.hist = hist if self.hist is None else 0.8 * self.hist + 0.2 * hist


def track_players(
    data: dict,
    calib: CourtCalibration | None = None,
    players_per_side: int = 2,
    features: np.ndarray | None = None,
    appearance_weight: float = 1.5,
    cuts: np.ndarray | None = None,
) -> Tracking:
    """Stable ids for the players in step 1's output.

    Each half keeps at most `players_per_side` tracks; every detection on a
    half is matched to one of them by predicted position (metres, scaled by
    how far a player could have moved) plus appearance distance.
    `features` (F, P, D) are optional appearance vectors and `cuts` (F,) marks
    edit cuts (both from video_features). Across a cut, positions are not
    joined: no interpolation, no distance, no speed.
    """
    time_s = np.asarray(data["time_s"], float)
    F = len(time_s)
    valid, side, court_xy = data["valid"], data["side"], data["court_xy"]
    cuts = np.zeros(F, bool) if cuts is None else np.asarray(cuts, bool)
    live: list[_Live] = []
    for i in range(F):
        t = time_s[i]
        if cuts[i]:
            for tr in live:  # positions before the cut say nothing about after it
                tr.last_xy, tr.vel = None, np.zeros(2)
        for s in (0, 1):
            dets = [p for p in np.flatnonzero(valid[i]) if side[i, p] == s and np.isfinite(court_xy[i, p]).all()]
            tracks = [k for k, tr in enumerate(live) if tr.side == s]
            while len(tracks) < min(players_per_side, len(dets)):
                live.append(_Live(s, F))
                tracks.append(len(live) - 1)
            if not dets or not tracks:
                continue
            cost = np.zeros((len(tracks), len(dets)))
            for a, k in enumerate(tracks):
                tr = live[k]
                for b, p in enumerate(dets):
                    if tr.last_xy is None or tr.last_t is None:
                        geo = 0.0
                    else:
                        reach = MAX_SPEED * min(t - tr.last_t, MAX_GAP_S) + 0.5
                        geo = np.linalg.norm(court_xy[i, p] - tr.predict(t)) / reach
                    app = _hist_distance(tr.hist, None if features is None else features[i, p])
                    cost[a, b] = geo + appearance_weight * app
            for a, b in _assign(cost):
                tr, p = live[tracks[a]], dets[b]
                tr.det[i] = p
                tr.update(court_xy[i, p], t, None if features is None else features[i, p])

    out = Tracking(time_s, np.asarray(data["frame"]), cuts=cuts)
    for tr in live:
        out.tracks.append(_finish(tr, data, calib, cuts))
    # ids: near players first, then far; left to right by mean position
    out.tracks.sort(key=lambda t: (t.side, np.nanmean(t.xy[:, 0]) if np.isfinite(t.xy).any() else 0))
    for n, t in enumerate(out.tracks, 1):
        t.id = n
    return out


# --- ground position ---------------------------------------------------------------


def _finish(tr: _Live, data: dict, calib: CourtCalibration | None, cuts: np.ndarray) -> Track:
    time_s = np.asarray(data["time_s"], float)
    F = len(time_s)
    seen = tr.det >= 0
    raw = np.full((F, 2), np.nan)
    idx = np.flatnonzero(seen)
    raw[idx] = data["court_xy"][idx, tr.det[idx]]
    airborne = np.zeros(F, bool)
    if calib is not None:
        airborne |= _airborne_by_scale(raw, tr.det, data, calib)
    ground = seen & ~airborne
    airborne |= seen & _speed_outliers(raw, ground, time_s, cuts)
    xy = _bridge(raw, seen & ~airborne, time_s, cuts)
    return Track(0, tr.side, raw, xy, seen, airborne & seen, tr.det)


def body_scale_ratio(kp: np.ndarray, sc: np.ndarray, foot_xy: np.ndarray, calib: CourtCalibration) -> float:
    """Shoulder-to-ankle image height divided by the floor's px-per-metre at the foot point."""
    sh = [j for j in _SHOULDERS if sc[j] >= _KPT_THR]
    an = [j for j in _ANKLES if sc[j] >= _KPT_THR]
    if not sh or not an or not np.isfinite(foot_xy).all():
        return np.nan
    height_px = kp[an, 1].mean() - kp[sh, 1].mean()
    a, b = calib.to_image(np.array([foot_xy - (0.5, 0.0), foot_xy + (0.5, 0.0)]))
    px_per_m = float(np.hypot(*(b - a)))
    return height_px / px_per_m if px_per_m > 0 else np.nan


def _airborne_by_scale(raw, det, data, calib, window_s: float = 6.0, upright_pct: float = 80) -> np.ndarray:
    F = len(raw)
    r = np.full(F, np.nan)
    for i in np.flatnonzero(det >= 0):
        p = det[i]
        r[i] = body_scale_ratio(data["keypoints"][i, p], data["scores"][i, p], raw[i], calib)
    t = np.asarray(data["time_s"], float)
    out = np.zeros(F, bool)
    ok = np.isfinite(r)
    if ok.sum() < 5:
        return out
    # Crouching and lunging lower the ratio, so the reference is the player's
    # upright level (a high percentile over a few seconds), not the median.
    # Broadcast singles clip (camera high above the court): upright ~1.3,
    # crouched ~1.0, jumps 1.36-1.43 near / +30-45 % far. With a phone 2.5-4 m
    # high a jump moves the projected foot much further, so the cue is stronger.
    lo = np.searchsorted(t, t - window_s / 2, side="left")  # t is sorted: window = slice
    hi = np.searchsorted(t, t + window_s / 2, side="right")
    for i in np.flatnonzero(ok):
        out[i] = r[i] > AIRBORNE_RATIO * np.nanpercentile(r[lo[i]:hi[i]], upright_pct)
    return out


def _speed_outliers(raw, ground, t, cuts) -> np.ndarray:
    """Frames whose position implies running faster than MAX_SPEED from the last
    accepted ground frame. After MAX_GAP_S the new position is accepted anyway
    (a cut in the video, or a long miss), so a track never gets stuck."""
    out = np.zeros(len(raw), bool)
    last = None
    seg = np.cumsum(cuts)  # segment number between cuts
    for i in np.flatnonzero(ground):
        if last is not None and seg[i] == seg[last]:
            dt = t[i] - t[last]
            if dt <= MAX_GAP_S and np.linalg.norm(raw[i] - raw[last]) > MAX_SPEED * dt + 0.3:
                out[i] = True
                continue
        last = i
    return out


def _bridge(raw, good, t, cuts) -> np.ndarray:
    """Keep good positions, interpolate gaps up to MAX_GAP_S (not across a cut),
    NaN elsewhere. The frame of a cut is left NaN so later smoothing and
    distance never join the two sides."""
    xy = np.full_like(raw, np.nan)
    idx = np.flatnonzero(good)
    xy[idx] = raw[idx]
    seg = np.cumsum(cuts)
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and t[b] - t[a] <= MAX_GAP_S and seg[a] == seg[b]:
            w = ((t[a + 1:b] - t[a]) / (t[b] - t[a]))[:, None]
            xy[a + 1:b] = (1 - w) * raw[a] + w * raw[b]
    xy[np.asarray(cuts, bool)] = np.nan
    return xy
