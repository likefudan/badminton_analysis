"""Pick the shuttle's path from all of TrackNetV3's candidates, not top-1 per frame.

TrackNetV3 marks every blob above 0.5 in its heatmap and keeps the one with
the largest bounding box, frame by frame. On amateur phone footage (first user
video, low camera, back wall with small lights, shuttles lying on the floor,
neighbouring courts) that choice fails in two ways:

  * static distractors: 37 % of top-1 points sat on spots that light up for
    much of the video (wall lights and fixtures, shuttles on the floor, a
    fixed "nothing found" point);
  * alternation: with two blobs the larger one wins, so the output jumps
    between the shuttle and another object from frame to frame.

So instead:
  1. static map: heatmap cells holding a blob in more than STATIC_FRAC of all
     frames are distractors - a flying shuttle never stays in one place;
  2. airspace mask (optional): the image of the volume above this court (up
     to AIR_HEIGHT_M), from the camera recovered from the court calibration;
  3. path: a best-scoring path through the remaining candidates over time
     (dynamic programming on a DAG). A candidate earns its heatmap score;
     linking two candidates g frames apart costs a speed penalty (fast is
     fine, teleporting is not) plus a per-missing-frame cost. Frames the path
     skips have no shuttle.
"""

from __future__ import annotations

import numpy as np

STATIC_FRAC = 0.05
STATIC_CELL = 8  # heatmap pixels
MIN_SCORE = 0.3
MAX_GAP = 5  # frames the path may skip between two candidates
FREE_SPEED = 40.0  # px/frame (video pixels) moved without penalty
MAX_SPEED = 260.0  # px/frame: anything faster is another object
SPEED_SCALE = 60.0
GAP_COST = 0.15
REWARD_OFFSET = 0.25  # a candidate is worth (score - offset)
AIR_HEIGHT_M = 9.0


def static_map(candidates: list, hm_shape=(288, 512), cell: int = STATIC_CELL, frac: float = STATIC_FRAC) -> np.ndarray:
    """Boolean grid (cells of `cell` heatmap pixels): True where blobs sit too often."""
    gh, gw = hm_shape[0] // cell + 1, hm_shape[1] // cell + 1
    freq = np.zeros((gh, gw))
    for c in candidates:
        seen = set()
        for x, y, *_ in c:
            seen.add((int(y // cell), int(x // cell)))
        for k in seen:
            freq[k] += 1
    freq /= max(1, len(candidates))
    hot = freq > frac
    # a blob near a cell boundary lands in either neighbour: grow by one cell
    grown = hot.copy()
    grown[1:] |= hot[:-1]
    grown[:-1] |= hot[1:]
    grown[:, 1:] |= hot[:, :-1]
    grown[:, :-1] |= hot[:, 1:]
    return grown


def airspace_mask(calib, margin_m: float = 1.0, height_m: float = AIR_HEIGHT_M) -> np.ndarray:
    """Image mask of the volume above the court (plus margins) up to `height_m`."""
    from .camera import camera_from_calibration

    return airspace_mask_for_camera(camera_from_calibration(calib), margin_m, height_m)


def airspace_mask_for_camera(cam, margin_m: float = 1.0, height_m: float = AIR_HEIGHT_M) -> np.ndarray:
    """The same mask for a known camera (also used by scripts/recording_planner.py)."""
    import cv2

    from .court import LENGTH, WIDTH

    w, h = cam.size
    # sample the volume densely and keep what is in front of the camera (a low camera
    # may stand inside the margins); the image region is the hull of those points
    g = np.stack(np.meshgrid(np.linspace(-margin_m, WIDTH + margin_m, 9),
                             np.linspace(-margin_m - 1, LENGTH + margin_m + 1, 25),
                             np.linspace(0.0, height_m, 10)), -1).reshape(-1, 3)
    g = g[cam.depth(g) > 0.5]
    mask = np.zeros((h, w), np.uint8)
    if len(g) < 4:
        return np.ones((h, w), bool)
    uv = cam.project(g)
    uv = np.clip(uv, -4 * w, 5 * w).astype(np.float32)
    hull = cv2.convexHull(uv).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def choose_path(candidates: list, scale: tuple[float, float], static: np.ndarray | None = None,
                mask: np.ndarray | None = None, cell: int = STATIC_CELL) -> np.ndarray:
    """candidates: per frame an array of (x, y, peak, ...) in heatmap pixels.
    Returns (F, 3): visibility, x, y in video pixels."""
    sx, sy = scale
    F = len(candidates)
    nodes = []  # (frame, x, y, reward)
    for f, c in enumerate(candidates):
        for x, y, peak, *_ in c:
            if peak < MIN_SCORE:
                continue
            if static is not None and static[min(static.shape[0] - 1, int(y // cell)), min(static.shape[1] - 1, int(x // cell))]:
                continue
            X, Y = x * sx, y * sy
            if mask is not None and not mask[min(mask.shape[0] - 1, int(Y)), min(mask.shape[1] - 1, int(X))]:
                continue
            nodes.append((f, X, Y, peak - REWARD_OFFSET))
    out = np.zeros((F, 3))
    if not nodes:
        return out
    nodes.sort(key=lambda n: n[0])
    fr = np.array([n[0] for n in nodes])
    xy = np.array([[n[1], n[2]] for n in nodes])
    rew = np.array([n[3] for n in nodes])
    # best[i]: best total of a path ending at node i; paths may start anywhere
    best = rew.copy()
    prev = np.full(len(nodes), -1)
    start_of = {}
    for i in range(len(nodes)):
        start_of.setdefault(fr[i], i)
    order = np.argsort(fr, kind="stable")
    lo = 0
    for i in order:
        f = fr[i]
        while fr[lo] < f - MAX_GAP:
            lo += 1
        j = np.arange(lo, len(nodes))
        j = j[(fr[j] < f)]
        if len(j) == 0:
            continue
        g = (f - fr[j]).astype(float)
        speed = np.hypot(*(xy[i] - xy[j]).T) / g
        ok = speed <= MAX_SPEED
        if not ok.any():
            continue
        cost = (np.maximum(0, speed - FREE_SPEED) / SPEED_SCALE) ** 2 + GAP_COST * (g - 1)
        val = np.where(ok, best[j] - cost + rew[i], -np.inf)
        k = int(np.argmax(val))
        if val[k] > best[i]:
            best[i], prev[i] = val[k], j[k]
    # the best path may be several disjoint stretches: take the best chain, then
    # repeat on the frames it does not cover (rallies are separated by dead time)
    used = np.zeros(F, bool)
    taken = np.zeros(len(nodes), bool)
    while True:
        cand = np.where(~taken & ~used[fr], best, -np.inf)
        i = int(np.argmax(cand))
        if not np.isfinite(cand[i]) or cand[i] <= 0.5:
            break
        chain = []
        while i >= 0 and not used[fr[i]]:
            chain.append(i)
            i = prev[i]
        if len(chain) < 3:
            taken[chain] = True
            continue
        for n in chain:
            out[fr[n]] = (1, xy[n, 0], xy[n, 1])
            used[fr[n]] = True
            taken[n] = True
        taken |= used[fr]
    return out


def fill_short_gaps(path: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Linear interpolation over gaps of up to `max_gap` frames inside the chosen path."""
    out = path.copy()
    seen = np.flatnonzero(path[:, 0] > 0)
    for a, b in zip(seen[:-1], seen[1:]):
        if 1 < b - a <= max_gap + 1:
            for f in range(a + 1, b):
                w = (f - a) / (b - a)
                out[f] = (1, (1 - w) * path[a, 1] + w * path[b, 1], (1 - w) * path[a, 2] + w * path[b, 2])
    return out


def rows_from_candidates(candidates: list, frame_size: tuple[int, int], calib=None, start_frame: int = 0,
                         measured: list | None = None) -> list:
    """Full pipeline on TrackNetV3 candidates: static map, airspace mask, path, short gaps.

    If `measured` is a list it is filled with one bool per row: True where the
    position is a blob TrackNet found in that frame, False where the row is a
    short gap filled by interpolation (or has no shuttle)."""
    w, h = frame_size
    static = static_map(candidates)
    mask = airspace_mask(calib) if calib is not None else None
    chosen = choose_path(candidates, (w / 512, h / 288), static, mask)
    path = fill_short_gaps(chosen)
    if measured is not None:
        measured[:] = [bool(v) for v in chosen[:, 0] > 0]
    return [(f + start_frame, int(v), int(round(x)), int(round(y))) for f, (v, x, y) in enumerate(path)]
