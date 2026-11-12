"""Step 2b: movement statistics per player (distance, speed, sprints, zones, heatmap).

Everything is computed from the cleaned ground positions of tracking.py,
over in-play time only. In-play time comes from the players' own motion:
during a rally at least one player keeps moving fast, between rallies both
walk slowly. This is rough (a long, slow defensive exchange can look like a
pause) but needs no shuttle tracking; step 5 will refine it with the shuttle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .court import LENGTH, NET_Y, WIDTH
from .tracking import Tracking

SPRINT_SPEED = 3.0  # m/s
SPRINT_MIN_S = 0.3
ACTIVE_SPEED = 1.5  # m/s: someone moving this fast -> rally is on
PAUSE_MIN_S = 2.5  # slower than ACTIVE_SPEED for this long -> dead time
RALLY_MIN_S = 1.5
SMOOTH_S = 0.5  # Savitzky-Golay window
HEAT_CELL_M = 0.25


# --- smoothing ------------------------------------------------------------------


def savgol_coeffs(window: int, order: int, deriv: int = 0) -> np.ndarray:
    """Savitzky-Golay convolution coefficients (centre point), numpy only."""
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, order + 1, increasing=True)
    c = np.linalg.pinv(A)[deriv] * np.prod(range(1, deriv + 1))
    return c[::-1]  # for np.convolve


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of each run of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


ACCEL_PSD = 10.0  # m^2/s^3, white-acceleration noise of the constant-velocity model; tuned on synthetic rallies (see tests)


def kalman_smooth(xy: np.ndarray, t: np.ndarray, cov: np.ndarray, q: float = ACCEL_PSD) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity Kalman filter + Rauch-Tung-Striebel smoother, per continuous stretch.

    Each position is weighted by its own covariance `cov` (F, 2, 2), so precise
    near-court positions are followed closely and imprecise far-court ones are
    smoothed along the direction in which they are uncertain. Frames with a
    position but no covariance (interpolated across a jump) get a loose one.
    """
    pos = np.full_like(xy, np.nan)
    vel = np.full_like(xy, np.nan)
    I2 = np.eye(2)
    for a, b in _runs(np.isfinite(xy).all(1)):
        n = b - a
        if n < 2:
            pos[a:b], vel[a:b] = xy[a:b], 0.0
            continue
        tt = t[a:b]
        R = cov[a:b].copy()
        fallback = np.nanmedian(R, axis=0) if np.isfinite(R).any() else 0.05 * I2
        loose = ~np.isfinite(R).all(axis=(1, 2))
        R[loose] = 4 * fallback + 0.01 * I2
        xs = np.zeros((n, 4)); Ps = np.zeros((n, 4, 4))
        xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4))
        x = np.r_[xy[a], 0.0, 0.0]
        P = np.diag([R[0, 0, 0], R[0, 1, 1], 9.0, 9.0])
        Fs = []
        for k in range(n):
            if k:
                dt = tt[k] - tt[k - 1]
                F_ = np.block([[I2, dt * I2], [np.zeros((2, 2)), I2]])
                Q = q * np.block([[dt**3 / 3 * I2, dt**2 / 2 * I2], [dt**2 / 2 * I2, dt * I2]])
                x = F_ @ x
                P = F_ @ P @ F_.T + Q
                Fs.append(F_)
            xp[k], Pp[k] = x, P
            S = P[:2, :2] + R[k]
            K = P[:, :2] @ np.linalg.inv(S)
            x = x + K @ (xy[a + k] - x[:2])
            P = P - K @ P[:2, :]
            xs[k], Ps[k] = x, P
        for k in range(n - 2, -1, -1):  # RTS backward pass
            C = Ps[k] @ Fs[k].T @ np.linalg.inv(Pp[k + 1])
            xs[k] = xs[k] + C @ (xs[k + 1] - xp[k + 1])
            Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
        pos[a:b], vel[a:b] = xs[:, :2], xs[:, 2:]
    return pos, vel


def smooth_track(xy: np.ndarray, t: np.ndarray, window_s: float = SMOOTH_S,
                 cov: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed position and velocity (m/s), per continuous (non-NaN) stretch.
    With per-frame covariances (from tracking) a Kalman smoother is used,
    otherwise a Savitzky-Golay filter."""
    if cov is not None and np.isfinite(cov).any():
        return kalman_smooth(xy, t, cov)
    pos = np.full_like(xy, np.nan)
    vel = np.full_like(xy, np.nan)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    win = max(5, int(round(window_s / dt)) | 1)
    for a, b in _runs(np.isfinite(xy).all(1)):
        seg = xy[a:b]
        n = b - a
        if n < 3:
            pos[a:b] = seg
            vel[a:b] = 0.0
            continue
        w = min(win, n if n % 2 else n - 1)
        if w < 5:
            pos[a:b] = seg
            vel[a:b] = np.gradient(seg, t[a:b], axis=0)
            continue
        c0, c1 = savgol_coeffs(w, 2, 0), savgol_coeffs(w, 2, 1)
        h = w // 2
        # odd (point-symmetric) padding keeps a straight run straight at the ends;
        # plain mirroring would flatten it and lose distance there
        padded = np.pad(seg, ((h, h), (0, 0)), mode="reflect", reflect_type="odd") if n > h else \
            np.pad(seg, ((h, h), (0, 0)), mode="edge")
        for k in range(2):
            pos[a:b, k] = np.convolve(padded[:, k], c0, mode="valid")
            vel[a:b, k] = np.convolve(padded[:, k], c1, mode="valid") / dt
    return pos, vel


# --- rallies --------------------------------------------------------------------------


def segment_rallies(speeds: np.ndarray, t: np.ndarray, breaks: np.ndarray | None = None) -> np.ndarray:
    """(F,) bool in-play mask from players' speeds (F, N), NaN = unknown.
    `breaks` (F,) marks edit cuts: a rally never spans one."""
    with np.errstate(invalid="ignore"):
        active = np.nanmax(np.where(np.isfinite(speeds), speeds, -1), axis=1) > ACTIVE_SPEED
    play = active.copy()
    brk = np.zeros(len(play), bool) if breaks is None else np.asarray(breaks, bool)
    play &= ~brk
    # close short pauses (not across a cut)
    for a, b in _runs(~play):
        if a > 0 and b < len(play) and t[b - 1] - t[a] < PAUSE_MIN_S and not brk[a:b].any():
            play[a:b] = True
    # drop short bursts
    for a, b in _runs(play):
        if t[b - 1] - t[a] < RALLY_MIN_S:
            play[a:b] = False
    return play


# --- stats --------------------------------------------------------------------------


@dataclass
class PlayerStats:
    id: int
    name: str
    side: str  # "near" / "far" (camera side at the start)
    in_play_s: float
    distance_m: float
    mean_speed_mps: float  # distance / in-play time
    max_speed_mps: float
    sprints: int
    zones_pct: dict  # front / mid / back, left / right (from the player's own view)
    airborne_frames: int
    rally_distances_m: list = field(default_factory=list)
    heatmap: dict = field(default_factory=dict)  # cells over the player's half, for drawing

    def to_json(self) -> dict:
        return asdict(self)


def zone_of(xy: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Depth zone (0 front, 1 mid, 2 back) and lateral side (0 left, 1 right) from
    the player's own view facing the net."""
    depth = NET_Y - xy[:, 1] if side == 0 else xy[:, 1] - NET_Y  # metres from the net
    third = NET_Y / 3
    zone = np.clip((depth // third).astype(int), 0, 2)
    left = xy[:, 0] < WIDTH / 2 if side == 0 else xy[:, 0] > WIDTH / 2
    return zone, np.where(left, 0, 1)


def heatmap(xy: np.ndarray, side: int, cell: float = HEAT_CELL_M, margin: float = 1.0) -> dict:
    y0, y1 = (-margin, NET_Y) if side == 0 else (NET_Y, LENGTH + margin)
    xe = np.arange(-margin, WIDTH + margin + 1e-9, cell)
    ye = np.arange(y0, y1 + 1e-9, cell)
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
    return {"x_edges": xe.round(3).tolist(), "y_edges": ye.round(3).tolist(),
            "counts": h.T.astype(int).tolist()}  # rows = y


def player_stats(track, t: np.ndarray, in_play: np.ndarray, rallies) -> PlayerStats:
    pos, vel = smooth_track(track.xy, t, cov=getattr(track, "meas_cov", None))
    speed = np.hypot(vel[:, 0], vel[:, 1])
    ok = np.isfinite(pos).all(1)
    play = in_play & ok
    step_ok = play[1:] & play[:-1]
    steps = np.hypot(*np.diff(pos, axis=0).T)
    steps = np.where(step_ok, steps, 0.0)
    dt = np.diff(t)
    in_play_s = float(np.sum(np.where(step_ok, dt, 0.0)))
    distance = float(np.nansum(steps))
    sprints = sum(1 for a, b in _runs(play & (speed > SPRINT_SPEED)) if t[b - 1] - t[a] >= SPRINT_MIN_S)
    zone, lat = zone_of(pos[play], track.side)
    n = max(1, len(zone))
    zones = {"front": 100 * np.mean(zone == 0) if len(zone) else 0.0,
             "mid": 100 * np.mean(zone == 1) if len(zone) else 0.0,
             "back": 100 * np.mean(zone == 2) if len(zone) else 0.0,
             "left": 100 * np.sum(lat == 0) / n, "right": 100 * np.sum(lat == 1) / n}
    rally_d = []
    for a, b in rallies:
        rally_d.append(round(float(steps[a:max(a, b - 1)].sum()), 2))
    return PlayerStats(
        id=track.id, name=track.name or f"球员 {track.id}", side="near" if track.side == 0 else "far",
        in_play_s=round(in_play_s, 2), distance_m=round(distance, 2),
        mean_speed_mps=round(distance / in_play_s, 2) if in_play_s else 0.0,
        max_speed_mps=round(float(np.nanmax(np.where(play, speed, np.nan))) if play.any() else 0.0, 2),
        sprints=int(sprints), zones_pct={k: round(float(v), 1) for k, v in zones.items()},
        airborne_frames=int(track.airborne.sum()), rally_distances_m=rally_d,
        heatmap=heatmap(pos[play], track.side),
    )


def analyse(tracking: Tracking) -> dict:
    """Rally segmentation + per-player stats. Returns a JSON-ready dict."""
    t = tracking.time_s
    smoothed = [smooth_track(tr.xy, t, cov=getattr(tr, "meas_cov", None)) for tr in tracking.tracks]
    speeds = np.stack([np.hypot(v[:, 0], v[:, 1]) for _, v in smoothed], axis=1) if smoothed else np.zeros((len(t), 0))
    in_play = segment_rallies(speeds, t, tracking.cuts) if smoothed else np.zeros(len(t), bool)
    rallies = _runs(in_play)
    players = [player_stats(tr, t, in_play, rallies) for tr in tracking.tracks]
    return {
        "duration_s": round(float(t[-1] - t[0]), 2) if len(t) else 0.0,
        "in_play_s": round(float(sum(t[b - 1] - t[a] for a, b in rallies)), 2),
        "rallies": [{"start_s": round(float(t[a]), 2), "end_s": round(float(t[b - 1]), 2)} for a, b in rallies],
        "players": [p.to_json() for p in players],
        "_in_play": in_play,  # for drawing / checks; dropped when saving
        "_smoothed": smoothed,
    }


def json_ready(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def shuttle_agreement(csv_path, frames: np.ndarray, in_play: np.ndarray, min_px_per_frame: float = 4.0) -> dict:
    """Sanity check against TrackNetV3 output (not a dependency): how often the
    motion-based in-play mask agrees with "the shuttle is moving".

    TrackNetV3 marks a shuttle lying on the floor or held in a hand as visible,
    and puts a fixed phantom point when it sees nothing, so "moving" (more than
    a few pixels per frame, smoothed over ~0.5 s) is used instead of "visible".
    """
    import csv

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    n = max(int(r["Frame"]) for r in rows) + 1
    xy = np.full((n, 2), np.nan)
    for r in rows:
        if int(r["Visibility"]):
            xy[int(r["Frame"])] = (float(r["X"]), float(r["Y"]))
    step = np.full(n, 0.0)
    step[1:] = np.nan_to_num(np.hypot(*np.diff(xy, axis=0).T), nan=0.0)
    k = 15
    moving = np.convolve(step > min_px_per_frame, np.ones(k) / k, mode="same") > 0.5
    frames = np.clip(np.asarray(frames, int), 0, n - 1)
    m = moving[frames]
    return {
        "agreement_pct": round(100 * float(np.mean(m == in_play)), 1),
        "in_play_shuttle_moving_pct": round(100 * float(m[in_play].mean()), 1) if in_play.any() else 0.0,
        "dead_shuttle_moving_pct": round(100 * float(m[~in_play].mean()), 1) if (~in_play).any() else 0.0,
    }
