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
     skips have no shuttle;
  4. still runs (with a calibration): stretches of that path that stay within a few pixels for
     8 frames or more, at a pixel that can be on the floor, are dropped (a shuttle lying on a
     neighbouring court, a shoe). Still stretches above the horizon are kept: the top of a lob;
  5. other courts (with a calibration): at the top of a flight the shuttle falls with g and nothing
     else (its drag is horizontal there), so how fast it curves down in the picture tells how far
     from the camera it is. A flight whose top is clearly nearer or further than this court's
     airspace along that pixel's ray belongs to another court: it is dropped and the path is
     chosen again without it.
"""

from __future__ import annotations

import numpy as np

STATIC_FRAC = 0.05
STATIC_CELL = 8  # heatmap pixels
MIN_SCORE = 0.3
MAX_GAP = 5  # frames the path may skip between two candidates
# speeds in px/frame as tuned on 1080p at 30 fps; choose_path scales them with `speed_unit`
FREE_SPEED = 40.0  # moved without penalty
MAX_SPEED = 260.0  # anything faster is another object
SPEED_SCALE = 60.0
GAP_COST = 0.15
REWARD_OFFSET = 0.25  # a candidate is worth (score - offset)
AIR_HEIGHT_M = 9.0
STILL_EXTENT = 0.005  # of the frame width: a stretch of the path that stays within this is not a flying shuttle
STILL_SECONDS = 0.13  # ... for at least this long (and at least STILL_FRAMES processed frames)
STILL_FRAMES = 8
GROUND_MAX_M = 50.0   # a still spot is "on the floor" if its pixel, read as a floor point, is this close to the camera
GRAVITY = 9.81
APEX_SECONDS = 0.2    # half of the stretch of path around a top of flight that a parabola is fitted to
APEX_FIT = 0.0015     # of the frame width: rms distance the path may keep from that parabola
AWAY_RATIO = 1.3      # distance from gravity vs the ray's stretch through this court's airspace (with margins)
AWAY_MARGIN_M = 1.0
AWAY_ROUNDS = 3
AWAY_RADIUS = 0.004   # of the frame width: candidates this close to a dropped point go with it
FLIGHT_TOL = 0.006    # of the frame width: a flight goes on while each point is this close to where it was heading


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


def speed_unit(frame_width: int, fps: float | None, frame_step: int = 1) -> float:
    """How many times further (in video pixels) the same shot moves between two
    processed frames than in 1080p at 30 fps: 4K has twice the pixels, 60 fps half
    the time, taking every second frame doubles it again."""
    rate = round(fps) / frame_step if fps else 30.0
    return (frame_width / 1920) * (30.0 / rate)


def choose_path(candidates: list, scale: tuple[float, float], static: np.ndarray | None = None,
                mask: np.ndarray | None = None, cell: int = STATIC_CELL, speed_unit: float = 1.0) -> np.ndarray:
    """candidates: per frame an array of (x, y, peak, ...) in heatmap pixels.
    Returns (F, 3): visibility, x, y in video pixels."""
    sx, sy = scale
    free_speed, max_speed, speed_scale = FREE_SPEED * speed_unit, MAX_SPEED * speed_unit, SPEED_SCALE * speed_unit
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
        ok = speed <= max_speed
        if not ok.any():
            continue
        cost = (np.maximum(0, speed - free_speed) / speed_scale) ** 2 + GAP_COST * (g - 1)
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


def drop_still_runs(path: np.ndarray, extent_px: float, min_frames: int = STILL_FRAMES, max_gap: int = MAX_GAP,
                    overlap: bool = True, on_floor=None) -> np.ndarray:
    """Remove stretches of the chosen path that do not move: `min_frames` or more points in a row
    (gaps of up to `max_gap` frames allowed) that all fit in a box of `extent_px`.

    The static map only catches spots that light up in a large share of the whole video. A shuttle
    lying on the floor of the next court, a white shoe or a shuttle held before a serve is picked
    up for a second or so and scores as high as a real one (first phone doubles clip: 42 frames on
    a shuttle lying 4.5 m outside the court, cell seen in 1.4 % of frames). What tells them apart
    is that a shuttle in flight does not stay within a few pixels for 8 frames.

    overlap: also join stretches that drift slowly (every window of `min_frames` points is tested,
    not consecutive blocks), so a spot that creeps is removed whole instead of leaving its ends.

    on_floor(xy) -> bool: if given, a still stretch is only dropped when its pixel can be a point on
    the floor. A lob or clear flying along the camera axis barely moves in the image near its top
    (same clip: frames 176-194 and 219-233), but there it is above the horizon, where nothing lying
    on a floor can be. A shuttle below the camera flying straight at it can still be lost."""
    out = path.copy()
    seen = np.flatnonzero(path[:, 0] > 0)
    drop = np.zeros(len(seen), bool)
    i = 0
    while i < len(seen):
        lo = hi = path[seen[i], 1:3]
        j = i
        while j + 1 < len(seen) and seen[j + 1] - seen[j] <= max_gap + 1:
            p = path[seen[j + 1], 1:3]
            nlo, nhi = np.minimum(lo, p), np.maximum(hi, p)
            if (nhi - nlo).max() >= extent_px:
                break
            lo, hi, j = nlo, nhi, j + 1
        if j - i + 1 >= min_frames and (on_floor is None or on_floor((lo + hi) / 2)):
            drop[i:j + 1] = True
            i = i + 1 if overlap else j + 1
        else:
            i += 1
    out[seen[drop]] = 0
    return out


def floor_test(calib, max_m: float = GROUND_MAX_M):
    """on_floor(xy) for drop_still_runs: True where the pixel, taken as a point on the floor, lies in front of
    the camera and within `max_m` of it. Pixels above the horizon map behind the camera or very far away."""
    from .camera import camera_from_calibration

    cam = camera_from_calibration(calib)
    eye = np.asarray(cam.center, float)

    def on_floor(xy) -> bool:
        g = np.asarray(calib.to_court(np.asarray([xy], float)), float)[0]
        p = np.array([[g[0], g[1], 0.0]])
        return bool(np.isfinite(g).all() and cam.depth(p)[0] > 0 and np.hypot(*(g - eye[:2])) <= max_m)

    return on_floor


def _ray_box(c, d, lo, hi):
    """Distances along the ray c + s d where it is inside the box lo..hi, or None."""
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    a, b = (lo - c) / d, (hi - c) / d
    a, b = np.minimum(a, b), np.maximum(a, b)
    s0, s1 = max(0.0, float(a.max())), float(b.min())
    return (s0, s1) if s1 > s0 else None


class CourtDistance:
    """How far along a pixel's ray the shuttle is, judged from gravity, and where that ray crosses this
    court's airspace."""

    def __init__(self, calib, margin_m: float = AWAY_MARGIN_M, height_m: float = AIR_HEIGHT_M):
        from .camera import camera_from_calibration
        from .court import LENGTH, WIDTH

        self.calib, self.cam = calib, camera_from_calibration(calib)
        self.Kinv = np.linalg.inv(self.cam.K)
        self.lo = np.array([-margin_m, -margin_m - 1, 0.0])
        self.hi = np.array([WIDTH + margin_m, LENGTH + margin_m + 1, height_m])

    def ray(self, xy) -> np.ndarray:
        u, v = self.calib.undistort(np.asarray([xy], float))[0]
        d = self.cam.R.T @ (self.Kinv @ np.array([u, v, 1.0]))
        return d / np.linalg.norm(d)

    def airspace(self, xy):
        """(nearest, furthest) distance from the camera where the ray of pixel xy is above this court."""
        return _ray_box(np.asarray(self.cam.center, float), self.ray(xy), self.lo, self.hi)

    def from_gravity(self, xy, acc) -> float | None:
        """Distance from the camera along the ray of pixel xy at which falling with g curves the path by
        `acc` (px/s^2, picture axes). None if the path does not curve downwards."""
        c, d = np.asarray(self.cam.center, float), self.ray(xy)

        def pull(s):  # picture acceleration of a point at distance s falling with g
            p = c + s * d
            return (self.cam.project(p - [0, 0, 0.005])[0] - self.cam.project(p)[0]) / 0.005 * GRAVITY

        g = pull(1.0)
        down = g / np.linalg.norm(g)
        a = float(np.dot(acc, down))
        if a <= 0:
            return None
        lo, hi = 0.05, 1000.0  # the pull falls off roughly as 1 / s
        if np.dot(pull(lo), down) < a or np.dot(pull(hi), down) > a:
            return None
        for _ in range(50):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if np.dot(pull(mid), down) > a else (lo, mid)
        return float(np.sqrt(lo * hi))


def _chains(path, max_gap: int = MAX_GAP) -> list:
    seen = np.flatnonzero(path[:, 0] > 0)
    if len(seen) == 0:
        return []
    return np.split(seen, np.flatnonzero(np.diff(seen) > max_gap + 1) + 1)


def flight_tops(path, rate: float, width: int):
    """Tops of flight in the chosen path: (frame, xy, acceleration px/s^2) where the path is highest in the
    picture over +-APEX_SECONDS and a parabola fits it within APEX_FIT of the frame width."""
    W = max(3, int(round(APEX_SECONDS * rate)))
    out = []
    for ch in _chains(path):
        inside = set(ch.tolist())
        best = None
        for k in ch:
            if k - W < ch[0] or k + W > ch[-1]:
                continue
            win = [f for f in range(k - W, k + W + 1) if f in inside]
            if len(win) < 1.5 * W or path[k, 2] > path[win, 2].min():
                continue
            t = (np.array(win) - k) / rate
            cx, cy = np.polyfit(t, path[win, 1], 2), np.polyfit(t, path[win, 2], 2)
            res = np.sqrt(np.mean((path[win, 1] - np.polyval(cx, t)) ** 2 + (path[win, 2] - np.polyval(cy, t)) ** 2))
            if res > APEX_FIT * width:
                continue
            top = (k, path[k, 1:3].copy(), np.array([2 * cx[0], 2 * cy[0]]), res)
            if best is not None and k - best[0] <= W:  # the same top seen from a neighbouring frame
                if res < best[3]:
                    best = top
                continue
            if best is not None:
                out.append(best[:3])
            best = top
        if best is not None:
            out.append(best[:3])
    return out


def flight_around(path, k: int, width: int, max_gap: int = MAX_GAP) -> list:
    """Frames of the chosen path that belong to the same flight as frame k: going out from k, each next
    point must be within FLIGHT_TOL of the frame width (plus a share of the step) of where the last few
    points were heading. A hit or a switch to another object ends the flight."""
    seen = np.flatnonzero(path[:, 0] > 0)
    i0 = int(np.searchsorted(seen, k))
    frames = [k]
    for step in (1, -1):
        got = [k]
        i = i0 + step
        while 0 <= i < len(seen) and abs(seen[i] - got[-1]) <= max_gap + 1:
            f = seen[i]
            recent = got[-6:]
            t = np.array(recent, float)
            deg = min(2, len(recent) - 1)
            if deg >= 1:
                pred = np.array([np.polyval(np.polyfit(t, path[recent, j], deg), f) for j in (1, 2)])
                move = np.hypot(*(path[recent[-1], 1:3] - pred))
                if np.hypot(*(path[f, 1:3] - pred)) > FLIGHT_TOL * width + 0.5 * move:
                    break
            got.append(f)
            i += step
        frames += got[1:]
    return sorted(frames)


def other_court_flights(path, calib, rate: float, width: int, dist: CourtDistance | None = None) -> list:
    """Flights of the chosen path whose top is clearly not above this court: list of
    (frames, top frame, distance from gravity, (nearest, furthest) of this court along that ray)."""
    dist = dist or CourtDistance(calib)
    out = []
    for k, xy, acc in flight_tops(path, rate, width):
        span = dist.airspace(xy)
        s = dist.from_gravity(xy, acc)
        if span is None or s is None:
            continue
        if s > span[1] * AWAY_RATIO or s < span[0] / AWAY_RATIO:
            out.append((flight_around(path, k, width), k, s, span))
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
                         measured: list | None = None, fps: float | None = None, frame_step: int = 1,
                         n_frames: int | None = None) -> list:
    """Full pipeline on TrackNetV3 candidates: static map, airspace mask, path, still runs and flights over
    other courts (with a calibration), short gaps.

    candidates: one array per processed frame (every `frame_step`-th video frame);
    the rows cover every video frame (skipped ones interpolated, at most `n_frames`).
    fps: the video's frame rate; with the frame width and frame_step it scales the
    speed limits, which were tuned on 1080p at 30 fps (None: taken as 30).

    If `measured` is a list it is filled with one bool per row: True where the
    position is a blob TrackNet found in that frame, False where the row is
    filled by interpolation (a short gap, a skipped frame) or has no shuttle."""
    w, h = frame_size
    static = static_map(candidates)
    mask = airspace_mask(calib) if calib is not None else None
    scale = (w / 512, h / 288)
    chosen = choose_path(candidates, scale, static, mask, speed_unit=speed_unit(w, fps, frame_step))
    if calib is not None:  # without the camera a still shuttle in the air cannot be told from one on the floor
        rate = (round(fps) if fps else 30.0) / frame_step
        on_floor = floor_test(calib)

        def still(p):
            return drop_still_runs(p, STILL_EXTENT * w, max(STILL_FRAMES, round(STILL_SECONDS * rate)), on_floor=on_floor)

        chosen = still(chosen)
        dist = CourtDistance(calib)
        candidates = list(candidates)
        for _ in range(AWAY_ROUNDS):  # take out flights over other courts and choose again without them
            away = other_court_flights(chosen, calib, rate, w, dist)
            if not away:
                break
            for frames, *_ in away:
                for f in frames:
                    c = candidates[f]
                    near = np.hypot(c[:, 0] * scale[0] - chosen[f, 1], c[:, 1] * scale[1] - chosen[f, 2]) < AWAY_RADIUS * w
                    candidates[f] = c[~near]
            chosen = still(choose_path(candidates, scale, static, mask, speed_unit=speed_unit(w, fps, frame_step)))
    path = fill_short_gaps(chosen)
    if frame_step > 1:
        from .shuttle import expand_rows

        rows = expand_rows(np.rint(path[:, 1]), np.rint(path[:, 2]), path[:, 0], frame_step,
                           n_frames if n_frames is not None else len(path) * frame_step)
    else:
        rows = [(f, int(v), int(round(x)), int(round(y))) for f, (v, x, y) in enumerate(path)]
    if measured is not None:
        seen = chosen[:, 0] > 0
        measured[:] = [f % frame_step == 0 and bool(seen[f // frame_step]) for f, *_ in rows]
    return [(f + start_frame, v, x, y) for f, v, x, y in rows]


SHUTTLE_SIZE_M = 0.065  # across the feathers
MIN_SHUTTLE_PX = 2.0  # model pixels: a shuttle smaller than this in the full frame needs a magnified tile


def far_roi(calib, frame_size: tuple[int, int], margin_m: float = 1.0, height_m: float = AIR_HEIGHT_M,
            min_px: float = MIN_SHUTTLE_PX) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1) of the part of the picture where this court's
    shuttle is smaller than `min_px` model pixels once the full frame is shrunk to
    512 wide: the image of the airspace further from the camera than
    focal length x shuttle size / (min_px x frame width / 512). Only that part needs
    tiles (shuttle.tile_views). None if the shuttle is large enough everywhere."""
    from .camera import camera_from_calibration
    from .court import LENGTH, WIDTH

    w, h = frame_size
    cam = camera_from_calibration(calib)
    far = cam.focal_px * SHUTTLE_SIZE_M / (min_px * w / 512)  # metres along the optical axis
    g = np.stack(np.meshgrid(np.linspace(-margin_m, WIDTH + margin_m, 17),
                             np.linspace(-margin_m - 1, LENGTH + margin_m + 1, 49),
                             np.linspace(0.0, height_m, 19)), -1).reshape(-1, 3)
    g = g[cam.depth(g) > max(far, 0.5)]
    if len(g) == 0:
        return None
    uv = cam.project(g)
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
    if len(uv) == 0:
        return None
    x0, y0 = np.floor(uv.min(0)).astype(int)
    x1, y1 = np.ceil(uv.max(0)).astype(int) + 1
    return int(x0), int(y0), int(min(w, x1)), int(min(h, y1))
