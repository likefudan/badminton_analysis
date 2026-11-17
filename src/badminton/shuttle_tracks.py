"""Choose this court's shuttle among several heatmap peaks per frame, by whole flights.

In a hall with many courts TrackNetV3 also lights up on the shuttles of other courts. On the user's
phone videos (IMG_0995/0999/0971, flying frames) a quarter of the model's top-1 points were a shuttle
of another court, while this court's shuttle was among the frame's three strongest heatmap peaks in
nine of ten of those frames. Which court a shuttle belongs to cannot be read off the picture position
(courts further back lie inside this court's airspace in the picture), but it can be read off how it
moves: a shuttle falls with g and is slowed by drag towards a fixed terminal speed, so the same
curve in pixels is a near flight seen up close or a big one far away. So:

  1. candidates: the strongest few local maxima of each frame's heatmap (also below the model's own
     threshold 0.5), minus spots that light up for long (static map);
  2. tracklets: candidates linked frame to frame along a smooth predicted path; a hit or a switch to
     another object ends a tracklet;
  3. flights: every long enough tracklet is fitted with a 3D flight (gravity + quadratic drag,
     camera from the court calibration); where the best fit puts the shuttle tells whether the
     flight is above this court;
  4. path: the dynamic-programming path of shuttle_path.choose_path over all candidates, where a
     candidate earns as before (peak - 0.25 from 0.3 up; a weaker peak inside a tracklet of 6 or more
     a little), times AWAY_WEIGHT if its tracklet's flight is above another court;
  5. a stretch of that path with fewer than CHAIN_OWN_POINTS points of flights above this court but
     some above another is dropped: this court is not playing, the path follows another one.

The rest is as in shuttle_path.rows_from_candidates (still stretches, gravity check at the tops of
flights, short gaps). Only for wide lenses (judges_courts): through a TV lens the flight's distance
cannot be told and the per-candidate path stays.

User's phone videos, frames left out of tuning (IMG_0999 from frame 1400, IMG_0971's held-out rally,
IMG_0995), frames where the shuttle flies: right 84.4 % -> 91.3 %, another object 6.1 % -> 3.2 %;
frames with no shuttle of this court that report one 34 % -> 22 %.
"""

from __future__ import annotations

import numpy as np

from . import shuttle_path as sp

TOP_K = 5              # peaks per frame
STATIC_PEAK = 0.3      # peaks that count for the static map (weaker ones flicker on every light)
PEAK_MIN = 0.05        # weakest peak used (the model's own threshold: 0.5)
LINK_GAP = 4           # frames a tracklet may miss
FIRST_STEP = 0.04      # of the frame width per 1/60 s: how far the second point of a tracklet may be
LINK_TOL = 0.004       # of the frame width: distance from the predicted point that is always fine
LINK_SPEED_TOL = 0.35  # ... plus this share of the last step
MIN_TRACK = 4          # points: shorter tracklets are noise (unless a single blob is strong, see STRONG)
STRONG = 0.5           # a peak this high counts even alone
FIT_MIN = 8            # points needed to fit a 3D flight
FIT_MAX_POINTS = 40    # longer tracklets are fitted on their first FIT_MAX_POINTS points
FIT_BATCH = 200        # tracklets fitted together
FIT_RMS = 0.003        # of the frame width: a fit this far off (rms) says nothing (a hit inside, two objects)
CHAIN_OWN_POINTS = 10  # a stretch of the path with fewer points of flights above this court than this ...
                       # and some of flights above another court is dropped
G = 9.81
TERMINAL = 6.8         # m/s, a feather shuttle's terminal speed
DEPTHS = (4.0, 9.0, 18.0, 35.0)  # m: starting guesses along the first point's ray
OWN_MARGIN = 1.0       # m around the court (and up to AIR_HEIGHT_M) that counts as this court
OWN_SHARE = 0.5        # share of a fitted flight that must be above this court
AWAY_WEIGHT = 0.15     # reward factor of a tracklet whose flight is above another court
MAX_ZOOM = 1.0         # focal length / frame width: through a longer lens (TV, 2.4) the distance of a flight
                       # cannot be told from its curve, and no other court is in view anyway
# rewards (as in shuttle_path.choose_path: a candidate with heatmap peak p earns p - 0.25 if p >= 0.3)
WEAK_REWARD = 0.08     # a peak below 0.3 inside a tracklet of at least WEAK_TRACK points
WEAK_TRACK = 6


def frame_candidates(weak: list, k: int = TOP_K, thr: float = PEAK_MIN) -> list:
    """weak: per frame rows (x, y, peak), strongest first (shuttle.weak_peaks). -> the top k above thr."""
    out = []
    for w in weak:
        w = np.asarray(w, float).reshape(-1, 3)
        out.append(w[w[:, 2] >= thr][:k])
    return out


def link(cands: list, width: int, fps: float | None) -> list:
    """Greedy frame-to-frame linking with a polynomial prediction. cands: per frame rows (x, y, peak),
    any pixel unit. Returns tracklets as arrays (frame, candidate index, x, y, peak)."""
    per = (60.0 / fps) if fps else 2.0
    first = FIRST_STEP * width * per
    active: list = []
    done: list = []
    for f, c in enumerate(cands):
        pairs = []
        for ti, t in enumerate(active):
            tail = t[-6:]
            fr = np.array([q[0] for q in tail], float)
            xy = np.array([q[2:4] for q in tail], float)
            gap = f - fr[-1]
            if len(t) == 1:
                pred, gate = xy[-1], first * gap
            else:
                deg = 1 if len(t) < 4 else 2
                tt = fr - fr[-1]
                pred = np.array([np.polyval(np.polyfit(tt, xy[:, j], deg), gap) for j in (0, 1)])
                step = np.hypot(*(xy[-1] - xy[-2])) / max(fr[-1] - fr[-2], 1.0)
                gate = LINK_TOL * width + LINK_SPEED_TOL * step * gap
            if len(c):
                d = np.hypot(c[:, 0] - pred[0], c[:, 1] - pred[1])
                for ci in np.flatnonzero(d <= gate):
                    pairs.append((d[ci] / gate - 0.3 * c[ci, 2], ti, int(ci)))
        pairs.sort()
        used_t, used_c = set(), set()
        for _, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            active[ti].append((f, ci, *c[ci][:3]))
        for ci in range(len(c)):
            if ci not in used_c:
                active.append([(f, ci, *c[ci][:3])])
        keep = []
        for t in active:
            (keep if f - t[-1][0] <= LINK_GAP else done).append(t)
        active = keep
    done += active
    return [np.array(t, float) for t in done]


class FlightFit:
    """Fits shuttle flights seen by a known camera (court calibration), many at once."""

    def __init__(self, calib, fps: float | None, substeps: int = 2):
        from .camera import camera_from_calibration
        from .court import LENGTH, WIDTH

        self.calib, self.cam = calib, camera_from_calibration(calib)
        self.fps = fps or 30.0
        self.sub = substeps
        self.C = np.asarray(self.cam.center, float)
        self.Kinv = np.linalg.inv(self.cam.K)
        self.lo = np.array([-OWN_MARGIN, -OWN_MARGIN - 1, -0.3])
        self.hi = np.array([WIDTH + OWN_MARGIN, LENGTH + OWN_MARGIN + 1, sp.AIR_HEIGHT_M + 3])

    def rays(self, xy):
        uv = self.calib.undistort(np.asarray(xy, float).reshape(-1, 2))
        d = (self.cam.R.T @ (self.Kinv @ np.column_stack([uv, np.ones(len(uv))]).T)).T
        return d / np.linalg.norm(d, axis=1, keepdims=True), uv

    def _simulate(self, th, span):
        """th (B, 6) = start position, start velocity. -> positions at frames 0..span (B, span+1, 3)."""
        dt = 1.0 / (self.fps * self.sub)
        k = G / TERMINAL**2
        P, V = th[:, :3].copy(), th[:, 3:].copy()
        out = np.empty((len(th), span + 1, 3))
        out[:, 0] = P
        for f in range(1, span + 1):
            for _ in range(self.sub):
                A = -k * np.linalg.norm(V, axis=1, keepdims=True) * V
                A[:, 2] -= G
                V = V + A * dt
                P = P + V * dt
            out[:, f] = P
        return out

    def _project(self, X):
        c = X @ self.cam.R.T + self.cam.t
        z = np.maximum(c[..., 2:3], 1e-3)
        return (c @ self.cam.K.T)[..., :2] / z

    def fit(self, tracks: list, iters: int = 15) -> list:
        """tracks: arrays with columns (frame, -, x, y, ...) in video pixels. -> per track the best fit's
        (rms px, share of the flight above this court, mean distance from the camera). Starts from a few
        distances along the first point's ray and runs Levenberg-Marquardt on all of them at once."""
        if not tracks:
            return []
        n_d = len(DEPTHS)
        obs, offs, firsts = [], [], []
        for t in tracks:
            t = t[:FIT_MAX_POINTS]
            d, uv = self.rays(t[:, 2:4])
            obs.append(uv)
            offs.append((t[:, 0] - t[0, 0]).astype(int))
            firsts.append((d[0], d[-1], max(t[-1, 0] - t[0, 0], 1) / self.fps))
        span = max(int(o[-1]) for o in offs)
        M = max(len(o) for o in offs)
        T = len(tracks)
        B = T * n_d
        idx = np.zeros((B, M), int)
        msk = np.zeros((B, M), bool)
        U = np.zeros((B, M, 2))
        th = np.zeros((B, 6))
        for i in range(T):
            d0, d1, dur = firsts[i]
            for j, s in enumerate(DEPTHS):
                b = i * n_d + j
                n = len(offs[i])
                idx[b, :n] = offs[i]
                msk[b, :n] = True
                U[b, :n] = obs[i]
                p0, p1 = self.C + s * d0, self.C + s * d1
                th[b] = np.concatenate([p0, (p1 - p0) / dur])

        def residual(thb, reps):
            X = self._simulate(thb, span)
            Xs = X[np.arange(len(thb))[:, None], np.tile(idx, (reps, 1))]
            m = np.tile(msk, (reps, 1))[..., None]
            r = ((self._project(Xs) - np.tile(U, (reps, 1, 1))) * m).reshape(len(thb), -1)
            return r, Xs

        eps = np.array([1e-3] * 3 + [1e-2] * 3)
        lam = np.full(B, 1e-2)
        r0, _ = residual(th, 1)
        cost = (r0**2).sum(1)
        for _ in range(iters):
            thb = np.concatenate([th] + [th + np.eye(6)[k] * eps[k] for k in range(6)])
            R, _ = residual(thb, 7)
            R = R.reshape(7, B, -1)
            J = ((R[1:] - R[0]) / eps[:, None, None]).transpose(1, 2, 0)  # (B, m, 6)
            Hm = J.transpose(0, 2, 1) @ J
            g = (J.transpose(0, 2, 1) @ R[0][..., None])[..., 0]
            improved = np.zeros(B, bool)
            for _try in range(4):
                todo = ~improved
                if not todo.any():
                    break
                D = np.eye(6)[None] * (np.diagonal(Hm, axis1=1, axis2=2) + 1e-9)[:, None, :]
                try:
                    step = -np.linalg.solve(Hm + lam[:, None, None] * D, g[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    break
                th2 = th + step
                r2, _ = residual(th2, 1)
                c2 = (r2**2).sum(1)
                ok = todo & np.isfinite(c2) & (c2 < cost) & (self.cam.depth(th2[:, :3]) > 0.3)
                th[ok], cost[ok] = th2[ok], c2[ok]
                lam[ok] = np.maximum(lam[ok] / 3, 1e-7)
                lam[todo & ~ok] *= 10
                improved |= ok
        _, Xs = residual(th, 1)
        out = []
        for i in range(T):
            n = len(offs[i])
            bs = [i * n_d + j for j in range(n_d)]
            b = min(bs, key=lambda q: cost[q])
            X = Xs[b, :n]
            inside = np.all((X >= self.lo) & (X <= self.hi), axis=1).mean()
            out.append((float(np.sqrt(cost[b] / n)), float(inside), float(np.linalg.norm(X - self.C, axis=1).mean())))
        return out


def judges_courts(calib) -> bool:
    """Whether to choose by flights at all (rows_from_peaks; else shuttle_path.rows_from_candidates): only
    with a calibration from a wide lens (phone). Through a long lens (TV) the distance of a flight cannot
    be told from its curve and no other court is in view; there the per-candidate path, tuned on
    broadcast matches, stays (choosing by tracklets alone lost 1.7 points of hit F1 on a BFMD final)."""
    if calib is None:
        return False
    from .camera import camera_from_calibration

    try:
        cam = camera_from_calibration(calib)
    except ValueError:
        return False
    return cam.focal_px / calib.frame_size[0] < MAX_ZOOM


def track_weights(tracks: list, calib, fps: float | None) -> np.ndarray:
    """Reward factor per tracklet: AWAY_WEIGHT if the flight that best explains it is mostly not above
    this court, else 1 (also when it is too short to fit, or without a calibration)."""
    w = np.ones(len(tracks))
    if calib is None:
        return w
    which = [i for i, t in enumerate(tracks) if len(t) >= FIT_MIN]
    if not which:
        return w
    ff = FlightFit(calib, fps)
    fits = []
    with np.errstate(all="ignore"):  # diverging trial steps are rejected by the fit
        for k in range(0, len(which), FIT_BATCH):  # bounded memory on long videos
            fits += ff.fit([tracks[i] for i in which[k:k + FIT_BATCH]])
    for i, (rms, inside, _) in zip(which, fits):
        if inside < OWN_SHARE and rms <= FIT_RMS * calib.frame_size[0]:
            w[i] = AWAY_WEIGHT
    return w


def rows_from_peaks(weak: list, frame_size: tuple[int, int], calib=None, start_frame: int = 0,
                    measured: list | None = None, fps: float | None = None, frame_step: int = 1,
                    n_frames: int | None = None, debug: dict | None = None) -> list:
    """Like shuttle_path.rows_from_candidates, but from the heatmap peaks of each processed frame
    (shuttle.weak_peaks rows (x, y, peak) in heatmap pixels) and choosing by tracklets."""
    w, h = frame_size
    scale = (w / 512, h / 288)
    rate_fps = (fps / frame_step) if fps else None
    cands = frame_candidates(weak)
    static = sp.static_map([c[c[:, 2] >= STATIC_PEAK] for c in cands], min_count=sp.static_min_count(rate_fps))
    keep = []
    for c in cands:
        if len(c):
            gy = np.minimum(static.shape[0] - 1, (c[:, 1] // sp.STATIC_CELL).astype(int))
            gx = np.minimum(static.shape[1] - 1, (c[:, 0] // sp.STATIC_CELL).astype(int))
            c = c[~static[gy, gx]]
        keep.append(c)
    cands = keep
    px = [np.column_stack([c[:, 0] * scale[0], c[:, 1] * scale[1], c[:, 2]]) if len(c) else np.zeros((0, 3))
          for c in cands]
    tracks = link(px, w, rate_fps)
    tracks = [t for t in tracks if len(t) >= MIN_TRACK or t[:, 4].max() >= STRONG]
    weights = track_weights(tracks, calib, rate_fps)
    reward = [np.where(c[:, 2] >= sp.MIN_SCORE, c[:, 2] - sp.REWARD_OFFSET, -np.inf) if len(c) else np.zeros(0)
              for c in cands]
    for t, wt in zip(tracks, weights):
        for f, ci, _, _, pk in t:
            r = reward[int(f)][int(ci)]
            if not np.isfinite(r) and len(t) >= WEAK_TRACK:
                r = WEAK_REWARD
            reward[int(f)][int(ci)] = r * wt if wt > 0 else -np.inf
    nodes = [np.column_stack([c[:, :2], r]) if len(c) else np.zeros((0, 3)) for c, r in zip(cands, reward)]
    nodes = [n[np.isfinite(n[:, 2])] for n in nodes]
    if debug is not None:
        debug.update(tracks=tracks, weights=weights)
    rows = sp.rows_from_candidates(nodes, frame_size, calib, 0, measured, fps, frame_step, n_frames, scored=True)
    if CHAIN_OWN_POINTS > 0 and calib is not None:
        rows = drop_away_chains(rows, tracks, weights, frame_step, measured)
    return [(f + start_frame, v, x, y) for f, v, x, y in rows]


def drop_away_chains(rows: list, tracks: list, weights: np.ndarray, frame_step: int = 1,
                     measured: list | None = None) -> list:
    """Remove stretches of the chosen path (no gap over shuttle_path.MAX_GAP processed frames) that hold
    fewer than CHAIN_OWN_POINTS points of flights fitted above this court and some of flights fitted
    above another: when this court is not playing, the path otherwise follows whatever flies elsewhere."""
    kind = {}
    for t, wt in zip(tracks, weights):
        if len(t) >= FIT_MIN:
            for f, _, x, y, _ in t:
                kind.setdefault(int(f), []).append((x, y, wt >= 1.0))
    rows = list(rows)
    seen = [i for i, r in enumerate(rows) if r[1]]
    if not seen:
        return rows
    gap = (sp.MAX_GAP + 1) * frame_step
    runs, cur = [], [seen[0]]
    for i in seen[1:]:
        if rows[i][0] - rows[cur[-1]][0] <= gap:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    for run in runs:
        n_own = n_away = 0
        for i in run:
            f, _, x, y = rows[i]
            if f % frame_step:
                continue
            for a, b, own in kind.get(f // frame_step, ()):
                if abs(x - a) <= 2 and abs(y - b) <= 2:
                    n_own += own
                    n_away += not own
                    break
        if n_own < CHAIN_OWN_POINTS and n_away > 0:
            for i in run:
                rows[i] = (rows[i][0], 0, 0, 0)
                if measured is not None:
                    measured[i] = False
    return rows
