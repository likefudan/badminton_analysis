"""Recording planner: what a phone at a given spot will see of a badminton court.

Given a camera setup (position in court coordinates, where it is aimed, lens,
resolution, frame rate) this synthesizes a pinhole camera and computes the
numbers the analysis depends on:

  * framing   - does the court (+ margin) fit, how much of the air above it,
                how high the frame reaches over each baseline and the net;
  * players   - metres per pixel along / across the court, the position error
                a 4 px (at 1080p) ankle keypoint error turns into, the pixel
                height of a player, how much of a doubles partner is hidden;
  * shuttle   - its size in pixels, how far it moves per frame at 30 / 60 fps
                and how long its motion-blur streak is for a given shutter,
                from simulated flights (clear, smash, drop, lift, drive, serve);
  * clutter   - how much of the neighbouring courts the airspace mask removes.

`python scripts/recording_planner.py` validates the model against the two real
videos (camera recovered from their court calibration), evaluates a grid of
practical setups and writes tables and plots to outputs/recording_guide/ (key
plots also to docs/). See docs/recording_guide.md for the conclusions.

    PYTHONPATH=src python scripts/recording_planner.py
    PYTHONPATH=src python scripts/recording_planner.py --setup 3.05,-2,3,0.5x,4k,60
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from badminton.camera import Camera, camera_from_calibration
from badminton.court import LENGTH, NET_Y, WIDTH, CourtCalibration, court_lines
from badminton.shuttle_path import airspace_mask_for_camera

CX = WIDTH / 2
# Horizontal field of view of iPhone video. 26 mm / 13 mm equivalent lenses are
# about 69 / 108 degrees on the full sensor; the task brief gives 77 for 1x.
# Video stabilisation crops a few percent more, so check the real framing on site.
LENSES = {"1x": 77.0, "0.5x": 108.0}
F_NUMBER = {"1x": 1.78, "0.5x": 2.2}
RESOLUTIONS = {"1080p": (1920, 1080), "4k": (3840, 2160)}
KPT_SIGMA_PX_1080 = 4.0  # tracking.KPT_SIGMA_PX: ankle keypoint error at 1080p
PLAYER_HEIGHT = 1.70
SHUTTLE_DIAMETER = 0.065  # feather skirt, metres
AIR_HEIGHT = 8.0  # clears and lifts peak below this in almost every hall
MARGIN = 1.0
TRACKNET_WIDTH = 512  # TrackNetV3 input is 512 x 288
G = 9.81
TERMINAL_SPEED = 6.8  # m/s, feather shuttle
SHUTTERS = (1 / 60, 1 / 120, 1 / 250, 1 / 500, 1 / 1000)
SIM_HZ = 960


# --------------------------------------------------------------------------- camera

@dataclass(frozen=True)
class Setup:
    name: str
    x: float
    y: float
    z: float
    lens: str = "1x"
    res: str = "1080p"
    fps: int = 60
    aim: tuple[float, float, float] = (CX, NET_Y, 0.0)  # court centre, on the floor
    roll_deg: float = 0.0
    hfov_deg: float | None = None  # overrides `lens`
    size: tuple[int, int] | None = None  # overrides `res`

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.size or RESOLUTIONS[self.res]

    @property
    def hfov(self) -> float:
        return self.hfov_deg if self.hfov_deg is not None else LENSES[self.lens]


def make_camera(s: Setup) -> Camera:
    w, h = s.frame_size
    f = (w / 2) / math.tan(math.radians(s.hfov) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    c = np.array([s.x, s.y, s.z], float)
    fwd = np.asarray(s.aim, float) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    r = math.radians(s.roll_deg)
    R = np.stack([math.cos(r) * right + math.sin(r) * down, -math.sin(r) * right + math.cos(r) * down, fwd])
    return Camera(K, R, -R @ c, (w, h))


def setup_from_camera(cam: Camera, name: str, fps: int = 30) -> Setup:
    """The planner's description (position, aim, roll, field of view) of a real camera."""
    c, fwd = cam.center, cam.R[2]
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    roll = math.degrees(math.atan2(cam.R[0] @ down, cam.R[0] @ right))
    w, h = cam.size
    hfov = math.degrees(2 * math.atan(w / 2 / cam.focal_px))
    return Setup(name, *(float(v) for v in c), fps=fps, aim=tuple(float(v) for v in c + 10 * fwd),
                 roll_deg=roll, hfov_deg=hfov, size=(w, h))


def pitch_deg(cam: Camera) -> float:
    """Degrees the optical axis points below the horizon."""
    return -math.degrees(math.asin(cam.R[2][2]))


def in_frame(cam: Camera, xyz) -> np.ndarray:
    p = np.atleast_2d(np.asarray(xyz, float))
    w, h = cam.size
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = cam.project(p)
    return (cam.depth(p) > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)


# --------------------------------------------------------------------------- framing

def _floor_outline(margin: float, n: int = 13) -> np.ndarray:
    xs, ys = np.linspace(-margin, WIDTH + margin, n), np.linspace(-margin, LENGTH + margin, 2 * n)
    pts = [(x, y) for x in xs for y in (ys[0], ys[-1])] + [(x, y) for y in ys for x in (xs[0], xs[-1])]
    return np.array([(x, y, 0.0) for x, y in pts])


def court_fits(cam: Camera, margin: float = MARGIN) -> bool:
    return bool(in_frame(cam, _floor_outline(margin)).all())


def player_coverage(cam: Camera, height: float = 2.0) -> float:
    """Fraction of where players are (over the court and half a metre behind each
    baseline, floor to `height`) that is in frame."""
    g = np.stack(np.meshgrid(np.linspace(0, WIDTH, 9), np.linspace(-0.5, LENGTH + 0.5, 21), np.linspace(0, height, 3)),
                 -1).reshape(-1, 3)
    return float(in_frame(cam, g).mean())


def airspace_fraction(cam: Camera, margin: float = MARGIN, height: float = AIR_HEIGHT) -> float:
    """Fraction of the volume over the court (+ margin), floor to `height`, that is in frame."""
    g = np.stack(np.meshgrid(np.linspace(-margin, WIDTH + margin, 9), np.linspace(-margin, LENGTH + margin, 25),
                             np.linspace(0, height, 17)), -1).reshape(-1, 3)
    return float(in_frame(cam, g).mean())


def visible_height(cam: Camera, y: float, top: float = 15.0) -> float:
    """Height (m) up to which the frame holds the full court width at court line `y`,
    counted from the floor; 0 if the floor there is out of frame."""
    xs = np.linspace(0, WIDTH, 7)
    z = 0.0
    for z_try in np.arange(0.0, top + 1e-9, 0.05):
        if not in_frame(cam, np.column_stack([xs, np.full(7, y), np.full(7, z_try)])).all():
            break
        z = z_try
    return float(z)


# --------------------------------------------------------------------------- players

def floor_jacobian(cam: Camera, xy) -> np.ndarray:
    """(n, 2, 2) d(court x, y) / d(image u, v) at floor points: metres per pixel.
    Same finite differences as tracking.measurement_cov, on the camera's own floor
    homography K [r1 r2 t]."""
    xy = np.atleast_2d(np.asarray(xy, float))
    Hc = cam.K @ np.column_stack([cam.R[:, 0], cam.R[:, 1], cam.t])  # court (x, y, 1) -> image
    H = np.linalg.inv(Hc)

    def to_court(uv):
        q = np.column_stack([uv, np.ones(len(uv))]) @ H.T
        return q[:, :2] / q[:, 2:3]

    img = cam.project(np.column_stack([xy, np.zeros(len(xy))]))
    e = 0.5
    du = (to_court(img + (e, 0)) - to_court(img - (e, 0))) / (2 * e)
    dv = (to_court(img + (0, e)) - to_court(img - (0, e))) / (2 * e)
    return np.stack([du, dv], axis=2)


KPT_REL = KPT_SIGMA_PX_1080 / 160  # keypoint error as a fraction of the player's pixel height


def kpt_sigma_px(cam: Camera, player_px: float = float("nan")) -> float:
    """Ankle keypoint error in this camera's pixels. At 1080p it is the pipeline's 4 px
    (tracking.KPT_SIGMA_PX). The pose model sees a resized crop of the player, so on a
    sharper frame a large player's error grows with the frame (same error in metres),
    while a small player's error stays near 4 px: more pixels on a small player are real
    extra detail. Assumption (not measured): the error is 2.5 % of the player's pixel
    height, kept between 4 px and 4 px x (width / 1920)."""
    hi = KPT_SIGMA_PX_1080 * max(1.0, cam.size[0] / 1920)
    if not np.isfinite(player_px):
        return hi
    return float(np.clip(KPT_REL * player_px, KPT_SIGMA_PX_1080, hi))


def position_precision(cam: Camera, xy, sigma_px: float | None = None) -> dict:
    """Metres per pixel across (x) and along (y) the court at a floor point, and the
    position error (1 sigma, metres) an isotropic `sigma_px` keypoint error becomes."""
    J = floor_jacobian(cam, [xy])[0]
    s = kpt_sigma_px(cam, person_px(cam, xy[0], xy[1])) if sigma_px is None else sigma_px
    across, along = float(np.hypot(*J[0])), float(np.hypot(*J[1]))
    return {"m_per_px_across": across, "m_per_px_along": along,
            "err_across_m": s * across, "err_along_m": s * along, "err_m": s * math.hypot(across, along)}


def person_px(cam: Camera, x: float, y: float, height: float = PLAYER_HEIGHT) -> float:
    """Image height (px) of a person standing at floor point (x, y); nan if not fully in frame."""
    p = np.array([[x, y, 0.0], [x, y, height]])
    if not in_frame(cam, p).all():
        return float("nan")
    uv = cam.project(p)
    return float(np.hypot(*(uv[1] - uv[0])))


def partner_hidden_m(cam: Camera, rear=(CX, 1.5), gap: float = 3.2, height: float = 1.75) -> float:
    """Doubles, front-and-back formation lined up with the camera: how much of the front
    partner (standing `gap` metres beyond the rear one, as seen from the camera) is hidden
    behind the rear partner, in metres from the feet up."""
    c = cam.center
    d = math.hypot(rear[0] - c[0], rear[1] - c[1])
    return float(np.clip(height - gap * (c[2] - height) / max(d, 1e-6), 0.0, height))


# --------------------------------------------------------------------------- shuttle

def simulate_flight(p0, v0, dt: float = 1 / SIM_HZ, t_max: float = 4.0) -> np.ndarray:
    """Shuttle positions (n, 3) every `dt` from launch until it reaches the floor.
    Quadratic drag with terminal speed TERMINAL_SPEED: a 70 m/s smash is down to
    half its speed within about 3 m."""
    k = G / TERMINAL_SPEED**2
    p, v = np.array(p0, float), np.array(v0, float)
    out = [p.copy()]

    def acc(v):
        return np.array([0, 0, -G]) - k * np.linalg.norm(v) * v

    for _ in range(int(t_max / dt)):
        # midpoint (RK2) step
        vm = v + 0.5 * dt * acc(v)
        p = p + dt * vm
        v = v + dt * acc(vm)
        out.append(p.copy())
        if p[2] <= 0:
            break
    return np.array(out)


def _launch(p0, target_xy, elev_deg: float, speed: float) -> np.ndarray:
    d = np.array([target_xy[0] - p0[0], target_xy[1] - p0[1]])
    d /= np.linalg.norm(d)
    e = math.radians(elev_deg)
    return simulate_flight(p0, [speed * math.cos(e) * d[0], speed * math.cos(e) * d[1], speed * math.sin(e)])


def flight_to(p0, target_xy, elev_deg: float) -> np.ndarray:
    """Flight launched at `elev_deg` that lands on `target_xy` (launch speed by bisection)."""
    want = math.hypot(target_xy[0] - p0[0], target_xy[1] - p0[1])
    lo, hi = 3.0, 150.0
    for _ in range(40):
        mid = (lo + hi) / 2
        tr = _launch(p0, target_xy, elev_deg, mid)
        if math.hypot(*(tr[-1, :2] - np.asarray(p0)[:2])) < want:
            lo = mid
        else:
            hi = mid
    return _launch(p0, target_xy, elev_deg, (lo + hi) / 2)


# name, hit point (y from the hitter's baseline, z), landing y, elevation deg, fixed launch speed
SHOTS = (
    ("clear", (1.0, 2.6), 12.7, 36.0, None),
    ("smash", (2.5, 2.8), 10.5, -14.0, 70.0),
    ("drop", (1.5, 2.6), 8.3, 8.0, None),
    ("lift", (5.3, 0.4), 12.6, 55.0, None),
    ("drive", (4.0, 1.6), 10.5, 4.0, None),
    ("serve", (4.6, 1.0), 12.8, 50.0, None),
)


@functools.lru_cache(maxsize=1)
def standard_flights() -> tuple:
    """A fixed mix of flights: each shot type from both ends, straight and cross-court.
    Returns ((shot, hitter_side, positions), ...)."""
    out = []
    for name, (y0, z0), y1, elev, speed in SHOTS:
        for xa, xb in ((CX, CX), (1.2, WIDTH - 1.2)):
            p0, tgt = (xa, y0, z0), (xb, y1)
            tr = _launch(p0, tgt, elev, speed) if speed else flight_to(p0, tgt, elev)
            out.append((name, "near", tr))
            far = tr.copy()
            far[:, 1] = LENGTH - far[:, 1]
            out.append((name, "far", far))
    return tuple(out)


def shuttle_px(cam: Camera, xyz) -> float:
    """Diameter of the shuttle's skirt in pixels at a 3D point."""
    return float(cam.focal_px * SHUTTLE_DIAMETER / np.linalg.norm(np.asarray(xyz, float) - cam.center))


def blur_px(cam: Camera, xyz, speed: float, shutter: float) -> float:
    """Streak length (px) of a shuttle at `xyz` moving at `speed` m/s across the view."""
    return float(cam.focal_px * speed * shutter / np.linalg.norm(np.asarray(xyz, float) - cam.center))


def flight_stats(cam: Camera, fps_list=(30, 60), shutters=SHUTTERS, flights=None) -> dict:
    """Over the standard flights: fraction of flight time in frame, per-frame displacement
    (px, median / p90) at each frame rate, blur streak (px, median / p90) for each shutter,
    and the same for the first 1/30 s of a smash."""
    flights = standard_flights() if flights is None else flights
    seen, speed_px, smash_px = [], [], []
    disp = {f: [] for f in fps_list}
    for name, _, tr in flights:
        ok = in_frame(cam, tr)
        seen.append(ok)
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = cam.project(tr)
        step = np.hypot(*(uv[1:] - uv[:-1]).T) * SIM_HZ  # px per second
        good = ok[1:] & ok[:-1]
        speed_px.append(step[good])
        if name == "smash":
            n = SIM_HZ // 30
            smash_px.append(step[:n][good[:n]])
        for f in fps_list:
            k = SIM_HZ // f
            d = np.hypot(*(uv[k:] - uv[:-k]).T)
            disp[f].append(d[ok[k:] & ok[:-k]])
    speed_px = np.concatenate(speed_px)
    smash_px = np.concatenate(smash_px) if smash_px else np.array([])
    out = {"shuttle_in_frame": float(np.concatenate(seen).mean())}
    by_shot = {}
    for (name, _, _), ok in zip(flights, seen):
        by_shot.setdefault(name, []).append(ok)
    out["in_frame_by_shot"] = {k: float(np.concatenate(v).mean()) for k, v in by_shot.items()}

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    for f in fps_list:
        d = np.concatenate(disp[f])
        out[f"disp_px_{f}fps_median"], out[f"disp_px_{f}fps_p90"] = pct(d, 50), pct(d, 90)
    for s in shutters:
        tag = f"1/{round(1 / s)}"
        out[f"blur_px_{tag}_median"], out[f"blur_px_{tag}_p90"] = pct(speed_px, 50) * s, pct(speed_px, 90) * s
        out[f"smash_blur_px_{tag}"] = pct(smash_px, 50) * s
    return out


def iso_needed(shutter: float, f_number: float, lux: float = 500.0) -> float:
    """ISO for a normal exposure of a scene lit at `lux` (incident-light equation, C = 250)."""
    return 250.0 * f_number**2 / (lux * shutter)


# --------------------------------------------------------------------------- clutter

def neighbour_removed(cam: Camera, offset_xy, mask: np.ndarray | None = None) -> tuple[float, float]:
    """For a court shifted by `offset_xy`: (fraction of its airspace that is in frame,
    fraction of that in-frame part which the own-court airspace mask removes)."""
    mask = airspace_mask_for_camera(cam) if mask is None else mask
    g = np.stack(np.meshgrid(np.linspace(0, WIDTH, 9), np.linspace(0, LENGTH, 19), np.linspace(0, AIR_HEIGHT, 13)),
                 -1).reshape(-1, 3) + (offset_xy[0], offset_xy[1], 0.0)
    ok = in_frame(cam, g)
    if not ok.any():
        return 0.0, float("nan")
    uv = cam.project(g[ok]).astype(int)
    return float(ok.mean()), float(1.0 - mask[uv[:, 1], uv[:, 0]].mean())


# --------------------------------------------------------------------------- everything

SPOTS = {"near": (CX, 0.0), "net": (CX, NET_Y), "far": (CX, LENGTH)}


def evaluate(cam: Camera, fps: int = 60, name: str = "", lens: str | None = None) -> dict:
    w, h = cam.size
    c = cam.center
    r = {"name": name, "x": float(c[0]), "y": float(c[1]), "z": float(c[2]), "width": w, "height": h, "fps": fps,
         "focal_px": cam.focal_px, "hfov_deg": math.degrees(2 * math.atan(w / 2 / cam.focal_px)),
         "pitch_deg": pitch_deg(cam)}
    r["court_fits"] = court_fits(cam, MARGIN)
    r["lines_fit"] = court_fits(cam, 0.0)
    r["player_coverage"] = player_coverage(cam)
    r["fit_margin_m"] = next((m for m in (2.0, 1.5, 1.0, 0.5, 0.0) if court_fits(cam, m)), -1.0)
    r["airspace_in_frame"] = airspace_fraction(cam)
    for spot, (x, y) in SPOTS.items():
        r[f"visible_height_{spot}_m"] = visible_height(cam, y)
        ok = bool(in_frame(cam, [[x, y, 0.0]])[0])
        p = position_precision(cam, (x, y)) if ok else {}
        for k in ("m_per_px_across", "m_per_px_along", "err_m"):
            r[f"{spot}_{k}"] = p.get(k, float("nan"))
        r[f"{spot}_player_px"] = person_px(cam, x, y)
        r[f"{spot}_shuttle_px"] = shuttle_px(cam, (x, y, 1.5))
    # worst spot on the court, whichever way the camera looks
    grid = np.array([(x, y) for x in np.linspace(0, WIDTH, 5) for y in np.linspace(0, LENGTH, 9)])
    vis = in_frame(cam, np.column_stack([grid, np.zeros(len(grid))]))
    r["worst_err_m"] = max((position_precision(cam, g)["err_m"] for g in grid[vis]), default=float("nan"))
    r["court_floor_visible"] = float(vis.mean())
    heights = [person_px(cam, x, y) for x, y in grid]
    r["min_player_px"] = float(np.nanmin(heights)) if np.isfinite(heights).any() else float("nan")
    r["far_shuttle_tracknet_px"] = r["far_shuttle_px"] * TRACKNET_WIDTH / w
    r["partner_hidden_m"] = partner_hidden_m(cam)
    r.update(flight_stats(cam, fps_list=(30, 60)))
    r["disp_px_median"], r["disp_px_p90"] = r[f"disp_px_{fps}fps_median"], r[f"disp_px_{fps}fps_p90"]
    mask = airspace_mask_for_camera(cam)
    r["mask_keeps_frame"] = float(mask.mean())
    gap = 2.0
    side = [neighbour_removed(cam, (sx * (WIDTH + gap), 0.0), mask) for sx in (-1, 1)]
    vis_side = [v for v, _ in side]
    r["side_court_in_frame"] = float(np.mean(vis_side))
    r["side_court_removed"] = (float(np.average([m for _, m in side if np.isfinite(m)],
                                                weights=[v for v, m in side if np.isfinite(m)]))
                               if any(v > 0 for v in vis_side) else float("nan"))
    r["end_court_in_frame"], r["end_court_removed"] = neighbour_removed(cam, (0.0, LENGTH + 2 * gap), mask)
    if lens in F_NUMBER:
        r["iso_1/500_500lux"] = iso_needed(1 / 500, F_NUMBER[lens])
    r["grade"] = grade(r)
    return r


def grade(r: dict) -> str:
    """'recommended', 'ok' (minimum acceptable) or 'poor'; thresholds are explained in
    docs/recording_guide.md."""
    if (r["lines_fit"] and r["visible_height_near_m"] >= 2.5 and r["far_err_m"] <= 0.30 and r["far_player_px"] >= 100
            and r["shuttle_in_frame"] >= 0.8 and r["partner_hidden_m"] <= 1.0 and r["fps"] >= 60):
        return "recommended"
    if (r["player_coverage"] >= 0.95 and r["far_err_m"] <= 0.60 and r["far_player_px"] >= 70
            and r["shuttle_in_frame"] >= 0.6):
        return "ok"
    return "poor"


def evaluate_setup(s: Setup) -> dict:
    r = evaluate(make_camera(s), s.fps, s.name, s.lens if s.hfov_deg is None else None)
    r.update(lens=s.lens if s.hfov_deg is None else f"{s.hfov:.0f}deg", res=s.res if s.size is None else "custom",
             aim_z=s.aim[2])
    return r


def best_aim(s: Setup, zs=np.arange(-6.0, 6.01, 0.25), slack: float = 0.02, give_up: float = 0.10) -> Setup:
    """Tilt the camera (aim point above / below the court centre) the way a player
    should. All four court lines must be in frame if any tilt manages that (otherwise
    as much of the players' area as possible); among those tilts keep as much of the
    shuttle's flight in frame as possible. A tilt that also shows 0.3 m of floor
    around the lines (feet behind the baseline) is preferred when it costs at most
    `give_up` of the shuttle's flight time."""
    cands = [replace(s, aim=(s.aim[0], s.aim[1], float(z))) for z in zs]
    cams = [make_camera(c) for c in cands]

    def pick(idx):
        seen = [flight_stats(cams[i], fps_list=(), shutters=())["shuttle_in_frame"] for i in idx]
        k = int(np.argmax(seen))
        return cands[idx[k]], seen[k]

    lines = np.flatnonzero([court_fits(c, 0.0) for c in cams])
    if not len(lines):
        cover = np.array([player_coverage(c) for c in cams])
        return pick(np.flatnonzero(cover >= cover.max() - slack))[0]
    best, seen = pick(lines)
    roomy = np.flatnonzero([court_fits(c, 0.3) for c in cams])
    if len(roomy):
        best_roomy, seen_roomy = pick(roomy)
        if seen_roomy >= seen - give_up:
            return best_roomy
    return best


def grid_setups() -> list[Setup]:
    out = []
    for lens in ("1x", "0.5x"):
        for d in (1, 2, 3, 4, 5):
            for z in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
                out.append(Setup(f"behind {d} m, {z} m high, {lens}", CX, -float(d), z, lens))
    for lens in ("1x", "0.5x"):
        for z in (1.5, 3.0):
            out.append(Setup(f"side at net, 3 m out, {z} m high, {lens}", -3.0, NET_Y, z, lens))
            out.append(Setup(f"near corner, 1.5 m out, {z} m high, {lens}", WIDTH + 1.0, -1.5, z, lens))
            out.append(Setup(f"behind, 2 m off-centre, 2 m back, {z} m high, {lens}", CX + 2.0, -2.0, z, lens))
    return out


def named_setups() -> tuple[Setup, Setup]:
    """(recommended, minimum acceptable) of docs/recording_guide.md, before tilting."""
    return (Setup("recommended: 3 m behind, 3 m high, 0.5x", CX, -3.0, 3.0, "0.5x", "4k", 60),
            Setup("minimum: 2 m behind, 2 m high, 0.5x", CX, -2.0, 2.0, "0.5x", "4k", 60))


# --------------------------------------------------------------------------- validation

def validate(data_dir: Path, outputs_dir: Path) -> list[dict]:
    """Recover the camera of each real video from its court calibration, describe it as a
    planner Setup, re-synthesize it and compare the planner's numbers with measurements."""
    from badminton.tracking import measurement_cov

    rows = []
    for stem, fps, shuttle_csv, skel in (
        ("IMG_8923_stab", 30, outputs_dir / "IMG_8923" / "IMG_8923.shuttle.csv",
         outputs_dir / "IMG_8923" / "IMG_8923.skeleton.npz"),
        ("singles", 30, outputs_dir / "shuttle" / "singles.shuttle.csv", data_dir / "singles.skeleton.npz"),
    ):
        path = data_dir / f"{stem}.court.json"
        if not path.exists():
            continue
        calib = CourtCalibration.load(path)
        real = camera_from_calibration(calib)
        setup = setup_from_camera(real, stem, fps)
        cam = make_camera(setup)
        names = list(calib.image_points)
        clicked = np.array([calib.image_points[n] for n in names])
        from badminton.court import MARKERS

        proj = cam.project(np.array([(*MARKERS[n], 0.0) for n in names]))
        r = evaluate(cam, fps, stem)
        row = {"video": stem, "x": setup.x, "y": setup.y, "z": setup.z, "hfov_deg": setup.hfov,
               "focal_px": cam.focal_px, "pitch_deg": pitch_deg(cam), "roll_deg": setup.roll_deg,
               "marker_reproj_px": float(np.hypot(*(proj - clicked).T).mean()),
               "calib_reproj_px": calib.reprojection_error_px}
        # position error: planner (pinhole) vs the pipeline's own homography Jacobian
        for spot, xy in SPOTS.items():
            cov = measurement_cov(np.array([xy]), np.array([True]), calib)[0]
            row[f"{spot}_err_planner_m"] = r[f"{spot}_err_m"]
            row[f"{spot}_err_homography_m"] = float(np.sqrt(np.trace(cov)))
            row[f"{spot}_m_per_px_along"] = r[f"{spot}_m_per_px_along"]
        # far half height in the image: net line to far baseline at the centre line
        a, b = cam.project([[CX, NET_Y, 0.0], [CX, LENGTH, 0.0]])
        a2, b2 = calib.to_image(np.array([[CX, NET_Y], [CX, LENGTH]]))
        row["far_half_px_planner"], row["far_half_px_homography"] = float(abs(a[1] - b[1])), float(abs(a2[1] - b2[1]))
        row["far_player_px_planner"] = r["far_player_px"]
        row["disp_px_30fps_median_planner"], row["disp_px_30fps_p90_planner"] = (
            r["disp_px_30fps_median"], r["disp_px_30fps_p90"])
        row["shuttle_in_frame_planner"] = r["shuttle_in_frame"]
        row["far_shuttle_tracknet_px"] = r["far_shuttle_tracknet_px"]
        if shuttle_csv.exists():
            row.update(_measured_shuttle(shuttle_csv))
        if skel.exists():
            row.update(_measured_players(skel, cam))
        rows.append(row)
    return rows


def _measured_shuttle(path: Path) -> dict:
    with open(path) as f:
        rec = [(int(r["Frame"]), int(r["Visibility"]), float(r["X"]), float(r["Y"])) for r in csv.DictReader(f)]
    a = np.array(rec)
    both = (a[1:, 1] == 1) & (a[:-1, 1] == 1) & (a[1:, 0] - a[:-1, 0] == 1)
    d = np.hypot(a[1:, 2] - a[:-1, 2], a[1:, 3] - a[:-1, 3])[both]
    d = d[d > 1.0]  # a shuttle at rest (in hand, on the floor) is not a flight
    return {"disp_px_30fps_median_measured": float(np.median(d)), "disp_px_30fps_p90_measured": float(np.percentile(d, 90))}


def _measured_players(path: Path, cam: Camera) -> dict:
    """Measured nose-to-ankle pixel height of far-half players against the planner's
    prediction for a person standing at the same floor point."""
    d = np.load(path, allow_pickle=True)
    kp, sc, xy, valid = d["keypoints"], d["scores"], d["court_xy"], d["valid"]
    meas, pred = [], []
    for f, p in zip(*np.nonzero(valid)):
        x, y = xy[f, p]
        if not (np.isfinite(x) and NET_Y + 1 < y < LENGTH + 1 and 0 < x < WIDTH):
            continue
        if min(sc[f, p, 0], sc[f, p, 15], sc[f, p, 16]) < 0.5:
            continue
        meas.append(kp[f, p, 15:17, 1].mean() - kp[f, p, 0, 1])
        pred.append(person_px(cam, x, y, 1.0))  # px per metre of height at that spot
    if len(meas) < 20:
        return {}
    meas, pred = np.array(meas), np.array(pred)
    ok = np.isfinite(pred) & (pred > 0)
    ratio = float(np.median(meas[ok] / pred[ok]))  # implied nose height in metres
    return {"far_player_nose_to_ankle_px_measured": float(np.median(meas[ok])),
            "far_player_implied_nose_height_m": ratio, "far_player_samples": int(ok.sum())}


# --------------------------------------------------------------------------- plots

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    names = {f.name for f in font_manager.fontManager.ttflist}
    zh = next((n for n in ("PingFang SC", "Hiragino Sans GB", "Heiti SC", "Noto Sans CJK SC", "Microsoft YaHei",
                           "Arial Unicode MS") if n in names), None)
    if zh:
        rcParams["font.sans-serif"] = [zh] + list(rcParams["font.sans-serif"])
        rcParams["font.family"] = "sans-serif"
    rcParams.update({"axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False})
    return zh is not None


def _draw_view(ax, cam: Camera, title: str, note: str):
    """What the frame looks like: court lines, net, a 1.7 m player at each baseline, the
    box 8 m over the court, and a baseline-to-baseline clear."""
    from matplotlib.patches import Rectangle

    w, h = cam.size
    ax.add_patch(Rectangle((0, 0), w, h, fc="#f0efec", ec=AXIS, lw=1))

    def line(p, q, **kw):
        pts = np.linspace(np.asarray(p, float), np.asarray(q, float), 60)
        pts = pts[cam.depth(pts) > 0.2]
        if len(pts) > 1:
            uv = cam.project(pts)
            ax.plot(uv[:, 0], uv[:, 1], **kw)

    for (a, b) in court_lines():
        line((*a, 0), (*b, 0), color=INK2, lw=0.9)
    for z in (0.0, 1.55):
        line((0, NET_Y, z), (WIDTH, NET_Y, z), color=INK, lw=1.2)
    for x in (0, WIDTH):
        line((x, NET_Y, 0), (x, NET_Y, 1.55), color=INK, lw=1.2)
    # airspace box, 8 m
    for x in (0, WIDTH):
        for y in (0, LENGTH):
            line((x, y, 0), (x, y, AIR_HEIGHT), color=BLUE, lw=0.8, ls=(0, (4, 3)))
    for a, b in (((0, 0), (WIDTH, 0)), ((WIDTH, 0), (WIDTH, LENGTH)), ((WIDTH, LENGTH), (0, LENGTH)), ((0, LENGTH), (0, 0))):
        line((*a, AIR_HEIGHT), (*b, AIR_HEIGHT), color=BLUE, lw=0.8, ls=(0, (4, 3)))
    for y in (0.3, LENGTH - 0.3):
        line((CX + 1.2, y, 0), (CX + 1.2, y, PLAYER_HEIGHT), color=ORANGE, lw=3.2, solid_capstyle="round")
    clear = next(tr for n, side, tr in standard_flights() if n == "clear" and side == "near" and tr[0, 0] != CX)
    ok = cam.depth(clear) > 0.2
    uv = cam.project(clear[ok])
    ax.plot(uv[:, 0], uv[:, 1], color=AQUA, lw=1.6)
    ax.set_xlim(-0.04 * w, 1.04 * w)
    ax.set_ylim(1.04 * h, -0.04 * h)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", pad=16)
    ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")


def plot_views(cams: list[tuple[str, str, Camera]], path: Path, zh: bool):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(cams)
    cols = 2
    fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(11, 3.75 * math.ceil(n / cols)))
    for ax, (title, note, cam) in zip(np.ravel(axes), cams):
        _draw_view(ax, cam, title, note)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    lab = (("球场线和网", "1.7 米高的人（两条底线）", "球场上空 8 米的范围", "一个底线到底线的高远球")
           if zh else ("court lines and net", "1.7 m player (both baselines)", "8 m over the court", "a clear"))
    fig.legend([Line2D([], [], color=INK2, lw=1.2), Line2D([], [], color=ORANGE, lw=3.2),
                Line2D([], [], color=BLUE, lw=1, ls=(0, (4, 3))), Line2D([], [], color=AQUA, lw=1.6)],
               lab, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grid(rows: list[dict], path: Path, zh: bool):
    """Small multiples over (height, distance behind the baseline): lens by metric."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"])
    ds, zs = (1, 2, 3, 4, 5), (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    metrics = (
        ("far_err_m", "远端底线位置误差（米）" if zh else "far-baseline position error (m)", "{:.2f}", (0.05, 0.6), True),
        ("far_player_px", "远端球员身高（像素）" if zh else "far player height (px)", "{:.0f}", (40, 160), False),
        ("shuttle_in_frame", "球在画面内的时间占比" if zh else "shuttle flight time in frame", "{:.0%}", (0.5, 1.0), False),
        ("partner_hidden_m", "双打前后站位被挡住的高度（米）" if zh else "doubles partner hidden (m)", "{:.1f}", (0, 1.75), True),
    )
    lenses = sorted({r["lens"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(len(lenses), len(metrics), figsize=(13.5, 3.5 * len(lenses)), squeeze=False)
    for i, lens in enumerate(lenses):
        for j, (key, title, fmt, (lo, hi), bad_high) in enumerate(metrics):
            ax = axes[i, j]
            ax.set_xticks(range(len(ds)), [str(d) for d in ds])
            ax.set_yticks(range(len(zs)), [str(z) for z in zs])
            for a, z in enumerate(zs):
                for b, d in enumerate(ds):
                    r = next(r for r in rows if r["lens"] == lens and abs(r["dist"] - d) < 1e-6 and abs(r["z"] - z) < 1e-6)
                    v = r[key]
                    t = (v - lo) / (hi - lo)
                    t = float(np.clip(t if bad_high else 1 - t, 0, 1))  # dark = worse
                    usable = r["lines_fit"]
                    fc = cmap(t) if usable else "#f0efec"
                    ax.add_patch(plt.Rectangle((b - 0.48, a - 0.48), 0.96, 0.96, fc=fc, ec="none"))
                    txt = fmt.format(v) if np.isfinite(v) else "-"
                    ax.text(b, a, txt, ha="center", va="center", fontsize=9,
                            color=MUTED if not usable else ("#ffffff" if t > 0.55 else INK))
                    if usable and r["grade"] == "recommended":
                        ax.add_patch(plt.Rectangle((b - 0.48, a - 0.48), 0.96, 0.96, fc="none", ec=ORANGE, lw=2))
            ax.set_xlim(-0.5, len(ds) - 0.5)
            ax.set_ylim(-0.5, len(zs) - 0.5)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{lens} · {title}", loc="left", fontsize=10)
            if j == 0:
                ax.set_ylabel("机位高度（米）" if zh else "camera height (m)")
            if i == len(lenses) - 1:
                ax.set_xlabel("底线后距离（米）" if zh else "distance behind baseline (m)")
    note = ("颜色越深越差；灰色 = 画面装不下整片场地的边线；橙色框 = 达到「推荐」标准。" if zh else
            "Darker = worse; grey = court lines do not fit; orange outline = meets 'recommended'.")
    fig.text(0.01, 0.005, note, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_blur(r: dict, path: Path, zh: bool, label: str):
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [3, 2]})
    tags = [f"1/{round(1 / s)}" for s in SHUTTERS]
    x = np.arange(len(tags))
    series = ((("一般飞行（中位数）" if zh else "typical flight (median)"), [r[f"blur_px_{t}_median"] for t in tags], BLUE),
              (("较快的 10%" if zh else "fastest 10 %"), [r[f"blur_px_{t}_p90"] for t in tags], ORANGE),
              (("杀球出手后 1/30 秒" if zh else "smash, first 1/30 s"), [r[f"smash_blur_px_{t}"] for t in tags], AQUA))
    bw = 0.26
    for k, (lab, vals, col) in enumerate(series):
        bars = a1.bar(x + (k - 1) * bw, vals, bw - 0.03, color=col, label=lab)
        for b, v in zip(bars, vals):
            a1.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK2)
    size = r["net_shuttle_px"]
    a1.axhline(size, color=INK2, lw=1, ls=(0, (4, 3)))
    a1.text(len(tags) - 0.5, size, ("球本身的大小 %.0f px" if zh else "shuttle itself %.0f px") % size, ha="right",
            va="top", fontsize=8.5, color=INK2)
    a1.set_xticks(x, [t + (" 秒" if zh else " s") for t in tags])
    a1.set_ylabel("拖影长度（像素）" if zh else "blur streak (px)")
    a1.set_title("快门越慢，球拖得越长" if zh else "Slower shutter, longer streak", loc="left")
    a1.legend(frameon=False, loc="upper right")
    a1.grid(axis="y", color=GRID, lw=0.6)
    a1.set_axisbelow(True)

    fx = np.arange(2)
    med = [r["disp_px_30fps_median"], r["disp_px_60fps_median"]]
    p90 = [r["disp_px_30fps_p90"], r["disp_px_60fps_p90"]]
    for k, (lab, vals, col) in enumerate(((("中位数" if zh else "median"), med, BLUE),
                                          (("较快的 10%" if zh else "fastest 10 %"), p90, ORANGE))):
        bars = a2.bar(fx + (k - 0.5) * 0.36, vals, 0.33, color=col, label=lab)
        for b, v in zip(bars, vals):
            a2.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    a2.set_xticks(fx, ["30 fps", "60 fps"])
    a2.set_ylabel("相邻两帧之间球移动的距离（像素）" if zh else "shuttle displacement per frame (px)")
    a2.set_title("帧率翻倍，每帧位移减半" if zh else "Double the frame rate, half the jump", loc="left")
    a2.legend(frameon=False, loc="upper right")
    a2.grid(axis="y", color=GRID, lw=0.6)
    a2.set_axisbelow(True)
    fig.text(0.01, 0.005, label, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- output

TABLE_COLS = (
    ("name", "setup", "{}"), ("res", "res", "{}"), ("fps", "fps", "{}"), ("pitch_deg", "pitch", "{:.0f}"),
    ("lines_fit", "lines", "{}"), ("player_coverage", "players in", "{:.0%}"),
    ("visible_height_near_m", "h near", "{:.1f}"), ("visible_height_net_m", "h net", "{:.1f}"),
    ("shuttle_in_frame", "shuttle in", "{:.0%}"),
    ("near_err_m", "err near", "{:.3f}"), ("net_err_m", "err net", "{:.3f}"), ("far_err_m", "err far", "{:.2f}"),
    ("far_m_per_px_along", "far m/px", "{:.3f}"), ("far_player_px", "far player px", "{:.0f}"),
    ("far_shuttle_px", "far shuttle px", "{:.1f}"), ("partner_hidden_m", "hidden m", "{:.2f}"),
    ("disp_px_median", "disp med", "{:.0f}"), ("disp_px_p90", "disp p90", "{:.0f}"),
    ("blur_px_1/120_p90", "blur 1/120", "{:.0f}"), ("blur_px_1/500_p90", "blur 1/500", "{:.1f}"),
    ("side_court_removed", "side masked", "{:.0%}"), ("grade", "grade", "{}"),
)


def _fmt(v, fmt):
    if isinstance(v, bool) or isinstance(v, np.bool_):
        return "yes" if v else "no"
    if isinstance(v, float) and not np.isfinite(v):
        return "-"
    return fmt.format(v)


def markdown_table(rows: list[dict], cols=TABLE_COLS) -> str:
    out = ["| " + " | ".join(h for _, h, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(k, float("nan")), f) for k, _, f in cols) + " |")
    return "\n".join(out)


def write_csv(rows: list[dict], path: Path):
    keys = [k for k in rows[0] if not isinstance(rows[0][k], dict)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def parse_setup(text: str) -> Setup:
    x, y, z, lens, res, fps = text.split(",")
    return Setup(f"({x}, {y}, {z}) {lens} {res} {fps}fps", float(x), float(y), float(z), lens, res.lower(), int(fps))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--setup", action="append", default=[], metavar="x,y,z,lens,res,fps",
                    help="evaluate one setup (court metres; lens 1x|0.5x; res 1080p|4k), e.g. 3.05,-2,3,0.5x,4k,60")
    ap.add_argument("--keep-aim", action="store_true", help="aim at the court centre instead of the best tilt")
    ap.add_argument("--data", default="data", help="folder with the *.court.json of the real videos")
    ap.add_argument("--outputs", default="outputs", help="folder with the analysis outputs of the real videos")
    ap.add_argument("--out", default="outputs/recording_guide")
    ap.add_argument("--docs", default=None, help="also save the key plots here (e.g. docs)")
    args = ap.parse_args(argv)

    if args.setup:
        for text in args.setup:
            s = parse_setup(text)
            s = s if args.keep_aim else best_aim(s)
            r = evaluate_setup(s)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                              if not isinstance(v, dict)}, ensure_ascii=False, indent=1))
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zh = _style()

    val = validate(Path(args.data), Path(args.outputs))
    (out / "validation.json").write_text(json.dumps(val, indent=1))
    for row in val:
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, indent=1))

    rows = []
    for s in grid_setups():
        for aim_mode in ("centre", "best"):
            s2 = s if aim_mode == "centre" else best_aim(s)
            for res, fps in (("1080p", 60), ("1080p", 30), ("4k", 60), ("4k", 30)):
                r = evaluate_setup(replace(s2, res=res, fps=fps))
                r.update(aim=aim_mode, dist=-s.y if s.x == CX else float("nan"))
                rows.append(r)
    for s in named_setups():
        for res, fps in (("4k", 60), ("4k", 30), ("1080p", 60), ("1080p", 30)):
            r = evaluate_setup(replace(best_aim(s), res=res, fps=fps))
            r.update(aim="best", dist=float("nan"))
            rows.append(r)
    write_csv(rows, out / "setups.csv")
    main_rows = [r for r in rows if r["aim"] == "best" and r["fps"] == 60]
    (out / "named.json").write_text(json.dumps([r for r in rows if r["name"][:3] in ("rec", "min")], indent=1))
    (out / "setups.md").write_text(
        "Best tilt, 60 fps. Lengths in metres, errors 1 sigma for a 4 px (at 1080p) keypoint error.\n\n"
        + markdown_table(main_rows) + "\n")

    behind = [r for r in rows if r["aim"] == "best" and r["fps"] == 60 and r["res"] == "4k" and np.isfinite(r["dist"])]
    plot_grid(behind, out / "height_distance_4k.png", zh)
    plot_grid([r for r in rows if r["aim"] == "best" and r["fps"] == 60 and r["res"] == "1080p"
               and np.isfinite(r["dist"])], out / "height_distance_1080p.png", zh)

    rec, mini = (best_aim(s) for s in named_setups())
    cams = []
    data = Path(args.data)
    if (data / "IMG_8923_stab.court.json").exists():
        c = camera_from_calibration(CourtCalibration.load(data / "IMG_8923_stab.court.json"))
        cams.append(("第一段手机视频（实测机位）" if zh else "First phone video (measured)",
                     "场边角上 1.4 米高、斜着拍：远端半场只有约 60 像素高" if zh else "1.4 m high at the corner", c))
    cams.append(("最低要求：底线后 2 米、高 2 米、0.5x" if zh else "Minimum: 2 m behind, 2 m high, 0.5x",
                 "近端底线贴着画面下沿，底角的人会出画一部分" if zh else "near baseline on the bottom edge", make_camera(mini)))
    cams.append(("推荐：底线后 3 米、高 3 米、0.5x" if zh else "Recommended: 3 m behind, 3 m high, 0.5x",
                 "近端底线离画面下沿一点距离，上方留给高球" if zh else "near baseline just above the bottom edge", make_camera(rec)))
    if (data / "singles.court.json").exists():
        c = camera_from_calibration(CourtCalibration.load(data / "singles.court.json"))
        cams.append(("电视转播（对照）" if zh else "Broadcast (reference)",
                     "底线后 16 米、高 5.7 米、长焦" if zh else "16 m behind, 5.7 m high, tele", c))
    plot_views(cams, out / "views.png", zh)
    r = evaluate_setup(rec)
    print(markdown_table([x for x in rows if x["name"][:3] in ("rec", "min")]))
    label = ("推荐机位（底线后 3 米、高 3 米、0.5x、4K）下的模拟结果；1080p 的像素数减半。" if zh else
             "Simulated for the recommended setup (3 m behind, 3 m high, 0.5x, 4K); halve for 1080p.")
    plot_blur(r, out / "blur_fps.png", zh, label)
    if args.docs:
        import shutil

        docs = Path(args.docs) / "recording_guide"
        docs.mkdir(parents=True, exist_ok=True)
        for n in ("views.png", "height_distance_4k.png", "blur_fps.png"):
            shutil.copy(out / n, docs / n)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
