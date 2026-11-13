"""Automatic court calibration: find the painted court lines and fit the court model.

Steps (all classical image processing, no model to download):

1. ``line_mask``: white, thin structures. The darkest colour channel is bright
   only where the pixel is white, and a morphological top-hat keeps what is
   narrower than a painted line can be (shirts and lamps are too wide).
2. ``detect_lines``: straight lines through the mask's centre lines
   (probabilistic Hough, collinear pieces merged).
3. The lines are grouped by vanishing point. A court has two groups: the
   lines along the court (side lines, centre line) and those across it (base
   lines, service lines). With the horizon known, the position of each line
   inside its group is an affine function of its court coordinate, so two
   image lines assigned to two model lines predict where every other model
   line must be; assignments that explain many lines are kept.
4. Each pair of assignments (one per group) gives a homography. It is scored
   by how much of the projected court lies on the mask and how much does not;
   the best ones are refined by fitting the homography to the mask pixels next
   to each projected line, and the best refined court wins.

Neighbouring courts fit the same model equally well. The court that covers
the most line pixels wins (normally the one the camera is pointed at); a
``hint`` pixel inside the wanted court overrides that.

Lens distortion (ultra-wide lenses, see lens.py) bends the court lines. The
search above needs straight lines, so it runs on a picture straightened with
a lens model. With ``lens="auto"`` the model is estimated from the lines
themselves: a first guess from how well the line pieces join up into long
straight lines (``coarse_k1``), then a joint fit of homography and lens to
the line pixels (``refine_lens``). The lens is only kept when the court then
lies clearly closer to the white lines; otherwise the result is exactly the
plain homography.

The court is symmetric, so the fit cannot tell left from right or near from
far. The result follows the convention of ``court.py``: y = 0 is the base line
nearer to the camera and x = 0 is on the left as the camera sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .court import LENGTH, MARKERS, NET_Y, WIDTH, CourtCalibration, court_lines
from .lens import Lens, rectify

REF_WIDTH = 1920  # pixel tolerances below are for a frame this wide
MIN_CONFIDENCE = 0.7
MIN_LINES = 6  # painted lines that must be found (of 12), at least 2 in each direction
# A lens estimated from the lines is kept only when it is at least this strong
# (0.004 = corner pixels move by 0.4 % of the half diagonal, 4 px at 1080p) ...
MIN_LENS_K = 0.004
# ... and brings the court lines this much closer to the middle of the white strokes
# (ratio of mean distances; pixel rounding alone leaves about half a pixel).
LENS_GAIN = 0.9


class AutoCalibrationError(RuntimeError):
    """The court could not be found reliably; calibrate by hand instead."""


# ---------------------------------------------------------------- court model

def painted_lines() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Court line segments that are painted on the floor (the net is not)."""
    return [(a, b) for a, b in court_lines() if not (a[1] == NET_Y and b[1] == NET_Y)]


_SEGMENTS = np.array(painted_lines(), dtype=np.float64)  # (12, 2, 2)
_ALONG = np.isclose(_SEGMENTS[:, 0, 0], _SEGMENTS[:, 1, 0])  # x = const: runs along the court
LONG_X = np.unique(_SEGMENTS[_ALONG][:, 0, 0])  # 0, 0.46, 3.05, 5.64, 6.10
CROSS_Y = np.unique(_SEGMENTS[~_ALONG][:, 0, 1])  # 0, 0.76, 4.72, 8.68, 12.64, 13.40


def _model_samples(step: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Points every `step` metres along the painted lines, and the line index of each."""
    pts, idx = [], []
    for i, (a, b) in enumerate(_SEGMENTS):
        n = max(2, int(round(np.hypot(*(b - a)) / step)) + 1)
        pts.append(np.linspace(a, b, n))
        idx.append(np.full(n, i))
    return np.concatenate(pts), np.concatenate(idx)


_SAMPLES, _SAMPLE_LINE = _model_samples()


# ---------------------------------------------------------------- image side

def median_frame(frames: list[np.ndarray], max_shift_px: float = 1.5) -> np.ndarray:
    """Per-pixel median of a few frames, which removes the moving players.

    Frames in which the picture has shifted against the first one (the camera
    moved, or a cut) are left out, because a median over a moving camera
    smears the lines. With fewer than 3 usable frames the first is returned.
    """
    if len(frames) < 3:
        return frames[0]

    def small(f):  # half size is enough to see a shift of a pixel
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA).astype(np.float32)

    first = small(frames[0])
    window = cv2.createHanningWindow(first.shape[::-1], cv2.CV_32F)
    keep = [frames[0]]
    for f in frames[1:]:
        (dx, dy), response = cv2.phaseCorrelate(first, small(f), window)
        if 2 * np.hypot(dx, dy) <= max_shift_px * frames[0].shape[1] / REF_WIDTH and response > 0.2:
            keep.append(f)
    if len(keep) < 3:
        return frames[0]
    return np.median(np.stack(keep), axis=0).astype(np.uint8)


def line_mask(frame: np.ndarray, scale: float | None = None) -> np.ndarray:
    """uint8 mask (0/255) of white structures about as thin as a court line.

    `scale` (here and below) is the size of the picture relative to one
    REF_WIDTH wide; the default takes it from the frame's width. A
    straightened picture is wider than the recording it was made from and
    passes the recording's scale, since its lines are no thicker.
    """
    scale = scale or frame.shape[1] / REF_WIDTH
    white = frame.min(axis=2)  # high only where all of B, G, R are high
    k = int(round(17 * scale)) | 1
    top = cv2.morphologyEx(white, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (max(k, 5),) * 2))
    return ((top > 25) & (white > 100)).astype(np.uint8) * 255


def _centre_lines(mask: np.ndarray) -> np.ndarray:
    """Ridge of the mask: the pixels in the middle of each stroke."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-3) & (mask > 0)
    return ridge.astype(np.uint8) * 255


@dataclass
class ImageLine:
    """A straight line in the image: a x + b y + c = 0 with (a, b) a unit vector."""

    abc: np.ndarray
    p0: np.ndarray  # the two ends of the part that was seen
    p1: np.ndarray
    support: float  # pixels of mask along it

    @property
    def mid(self) -> np.ndarray:
        return (self.p0 + self.p1) / 2

    @property
    def direction(self) -> np.ndarray:
        return np.array([-self.abc[1], self.abc[0]])


def detect_lines(mask: np.ndarray, max_lines: int = 60, scale: float | None = None) -> list[ImageLine]:
    """Long straight lines in the mask, strongest first."""
    h, w = mask.shape
    scale = scale or w / REF_WIDTH
    segs = cv2.HoughLinesP(_centre_lines(mask), 1, np.pi / 360, int(40 * scale),
                           minLineLength=50 * scale, maxLineGap=12 * scale)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(np.float64)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    tol = 3.0 * scale
    groups: list[dict] = []
    for i in np.argsort(-length):
        ends = segs[i].reshape(2, 2)
        for g in groups:
            if np.abs(ends @ g["abc"][:2] + g["abc"][2]).max() <= tol:
                g["ends"].append(ends)
                g["len"].append(length[i])
                break
        else:
            d = (ends[1] - ends[0]) / length[i]
            n = np.array([-d[1], d[0]])
            groups.append({"abc": np.array([n[0], n[1], -n @ ends[0]]), "ends": [ends], "len": [length[i]]})
    lines = []
    for g in groups:
        pts = np.concatenate(g["ends"])
        wts = np.repeat(g["len"], 2)
        abc = _fit_line(pts, wts)
        d = np.array([-abc[1], abc[0]])
        t = pts @ d
        # Hough returns overlapping pieces: count each stretch of the line once.
        cover = np.zeros(int(t.max() - t.min()) + 2, bool)
        for e in g["ends"]:
            a, b = sorted((e @ d - t.min()).astype(int))
            cover[a:b + 1] = True
        foot = lambda s: d * s - abc[:2] * abc[2]  # noqa: E731 - point of the line at position s
        lines.append(ImageLine(abc, foot(t.min()), foot(t.max()), float(cover.sum())))
    lines = [ln for ln in lines if ln.support >= 80 * scale]
    lines.sort(key=lambda ln: -ln.support)
    return lines[:max_lines]


def _fit_line(pts: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """Total least squares line through weighted points."""
    c = (pts * wts[:, None]).sum(0) / wts.sum()
    q = (pts - c) * np.sqrt(wts)[:, None]
    n = np.linalg.svd(q, full_matrices=False)[2][-1]
    return np.array([n[0], n[1], -n @ c])


# ---------------------------------------------------------------- line groups

def _families(lines: list[ImageLine], max_families: int = 3, angle_deg: float = 2.0) -> list[tuple[np.ndarray, list[int]]]:
    """Groups of lines that meet in one vanishing point: [(vp, line indices)], biggest first.

    Coordinates are pixels; the vanishing point is homogeneous, so parallel
    image lines (vanishing point at infinity) need no special case.
    """
    L = np.array([ln.abc for ln in lines])
    mid = np.array([ln.mid for ln in lines])
    dirs = np.array([ln.direction for ln in lines])
    weight = np.array([ln.support for ln in lines])
    # Work around the image centre in units of ~1 so the cross products are well scaled.
    c0, s0 = mid.mean(0), np.abs(mid - mid.mean(0)).max() + 1.0
    Ln = np.column_stack([L[:, 0], L[:, 1], (L[:, 2] + L[:, :2] @ c0) / s0])
    midn = (mid - c0) / s0
    free = np.ones(len(lines), bool)
    out = []
    cos_tol = np.cos(np.radians(angle_deg))
    for _ in range(max_families):
        idx = np.flatnonzero(free)[:30]
        if len(idx) < 2:
            break
        i, j = np.triu_indices(len(idx), 1)
        vps = np.cross(Ln[idx[i]], Ln[idx[j]])
        norm = np.linalg.norm(vps, axis=1)
        ok = norm > 1e-9
        vps = vps[ok] / norm[ok, None]
        # direction from every line's middle to every vanishing point
        to = vps[:, None, :2] - vps[:, None, 2:3] * midn[None]
        to /= np.linalg.norm(to, axis=2, keepdims=True) + 1e-12
        inl = (np.abs((to * dirs[None]).sum(2)) > cos_tol) & free[None]
        score = (inl * weight[None]).sum(1)
        best = int(np.argmax(score))
        members = np.flatnonzero(inl[best])
        if len(members) < 2:
            break
        vp = vps[best]
        out.append((np.array([vp[0] * s0 + vp[2] * c0[0], vp[1] * s0 + vp[2] * c0[1], vp[2]]), members.tolist()))
        free[members] = False
    return out


def _offsets(lines: list[ImageLine], members: list[int], horizon: np.ndarray, origin: np.ndarray, unit: float) -> np.ndarray:
    """Position of each line of one group after the horizon is sent to infinity.

    Once the horizon is at infinity the floor is only affinely distorted, the
    group's lines are parallel, and their positions along any transversal are
    an affine function of the court coordinate.
    """
    L = np.array([lines[m].abc for m in members])
    # move the origin to a floor point so the horizon does not pass through it
    Ln = np.column_stack([L[:, 0], L[:, 1], (L[:, 2] + L[:, :2] @ origin) / unit])
    hn = np.array([horizon[0], horizon[1], (horizon[2] + horizon[:2] @ origin) / unit])
    hn = hn / hn[2]
    # rectifying map Hp = [[1,0,0],[0,1,0],[h]]; lines transform with Hp^-T
    R = Ln.copy()
    R[:, 0] -= hn[0] * Ln[:, 2]
    R[:, 1] -= hn[1] * Ln[:, 2]
    n = R[:, :2] / np.linalg.norm(R[:, :2], axis=1, keepdims=True)
    ref = n[0]
    n[(n @ ref) < 0] *= -1
    u = n.mean(0)
    u /= np.linalg.norm(u)
    return -R[:, 2] / (R[:, :2] @ u)


@dataclass
class _Assignment:
    """Which model line each image line of a group is: {image line index: court coordinate}."""

    lines: dict[int, float]
    score: float


def _assignments(lines: list[ImageLine], members: list[int], rho: np.ndarray, model: np.ndarray,
                 tol_m: float, keep: int = 40) -> list[_Assignment]:
    """Ways to read one group of image lines as the model lines at court coordinates `model`."""
    n = len(members)
    if n < 2:
        return []
    weight = np.array([lines[m].support for m in members])
    weight = np.minimum(weight / np.median(weight), 1.0)
    ii, jj = np.nonzero(~np.eye(n, dtype=bool))
    ma, mb = np.triu_indices(len(model), 1)
    # every ordered pair of image lines against every pair of model lines
    I = np.repeat(ii, len(ma))
    J = np.repeat(jj, len(ma))
    A = np.tile(model[ma], len(ii))
    B = np.tile(model[mb], len(ii))
    alpha = (rho[J] - rho[I]) / (B - A)
    ok = np.abs(alpha) > 1e-9
    I, J, A, B, alpha = I[ok], J[ok], A[ok], B[ok], alpha[ok]
    beta = rho[I] - alpha * A
    coord = (rho[None] - beta[:, None]) / alpha[:, None]  # (hyp, n) court coordinate of every line
    resid = np.abs(coord[:, :, None] - model[None, None])  # (hyp, n, model)
    near = resid.argmin(2)
    rmin = resid.min(2)
    matched = rmin < tol_m
    hit = np.zeros((len(alpha), len(model)), bool)
    for k in range(len(model)):
        hit[:, k] = (matched & (near == k)).any(1)
    inside = (coord > model[0] + tol_m) & (coord < model[-1] - tol_m) & ~matched
    score = hit.sum(1) - 0.5 * (inside * weight[None]).sum(1) - 0.5 * np.where(matched, rmin, 0).sum(1) / tol_m / n
    out: dict[tuple, _Assignment] = {}
    for h in np.argsort(-score):
        if hit[h].sum() < 2:
            break
        chosen: dict[int, tuple[int, float]] = {}  # model index -> (image line, residual): the closest one
        for li in np.flatnonzero(matched[h]):
            k = int(near[h, li])
            if k not in chosen or rmin[h, li] < chosen[k][1]:
                chosen[k] = (li, rmin[h, li])
        key = tuple(sorted((members[li], k) for k, (li, _) in chosen.items()))
        if key not in out:
            out[key] = _Assignment({members[li]: float(model[k]) for k, (li, _) in chosen.items()}, float(score[h]))
            if len(out) >= keep:
                break
    return list(out.values())


def _homography(lines: list[ImageLine], along: _Assignment, across: _Assignment) -> np.ndarray | None:
    """Court -> image homography from the crossings of the assigned lines."""
    img, court = [], []
    for i, x in along.lines.items():
        for j, y in across.lines.items():
            p = np.cross(lines[i].abc, lines[j].abc)
            if abs(p[2]) < 1e-9:
                return None
            img.append(p[:2] / p[2])
            court.append((x, y))
    H, _ = cv2.findHomography(np.array(court), np.array(img), 0)
    return H


# ---------------------------------------------------------------- scoring

def _project(Hc: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Court points through court->image homographies (..., 3, 3): pixels and depth sign."""
    ph = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    q = np.einsum("...ij,nj->...ni", Hc, ph)
    w = q[..., 2]
    centre = np.einsum("...ij,j->...i", Hc, np.array([WIDTH / 2, LENGTH / 2, 1.0]))[..., 2]
    w = w * np.sign(centre)[..., None]  # positive = in front of the camera
    safe = np.where(np.abs(q[..., 2:3]) < 1e-12, 1e-12, q[..., 2:3])
    return q[..., :2] / safe, w


def _overlap(Hc: np.ndarray, dist: np.ndarray, tol: float, use: np.ndarray | None = None,
             valid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """How well projected court lines lie on the mask, for a stack of homographies.

    `dist` is the distance of every pixel to the nearest mask pixel. Samples
    are weighted by the pixel length they stand for, so the result is in
    pixels of line: `on` matched, `off` visible but unmatched. `use` switches
    individual samples off; `valid` marks the pixels that show anything at
    all (a straightened picture has empty borders), the rest counts as out of
    the picture.
    """
    Hc = np.asarray(Hc, dtype=np.float64).reshape(-1, 3, 3)
    h, w = dist.shape
    uv, depth = _project(Hc, _SAMPLES)  # (C, N, 2)
    step = np.linalg.norm(np.diff(uv, axis=1), axis=2)
    step = np.concatenate([step, step[:, -1:]], 1)
    last = np.flatnonzero(np.diff(_SAMPLE_LINE, append=-1) != 0)
    step[:, last] = step[:, last - 1]
    visible = (depth > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1)
    if use is not None:
        visible &= use[None]
    ix = np.clip(np.nan_to_num(uv[..., 0]), 0, w - 1).round().astype(np.intp)
    iy = np.clip(np.nan_to_num(uv[..., 1]), 0, h - 1).round().astype(np.intp)
    if valid is not None:
        visible &= valid[iy, ix]
    step = np.where(visible, np.minimum(step, 0.05 * w), 0.0)
    d = dist[iy, ix]
    match = np.clip(1.5 - d / tol, 0, 1)  # full credit within tol/2, none beyond 1.5 tol
    on = (step * match).sum(1)
    total = step.sum(1)
    per_line_on = np.stack([(step * match)[:, _SAMPLE_LINE == i].sum(1) for i in range(len(_SEGMENTS))], 1)
    per_line = np.stack([step[:, _SAMPLE_LINE == i].sum(1) for i in range(len(_SEGMENTS))], 1)
    return {
        "on": on, "off": total - on, "total": total,
        "line_on": per_line_on, "line_total": per_line,
        "mean_dist": (step * np.minimum(d, 4 * tol)).sum(1) / np.maximum(total, 1e-9),
        "visible": visible.mean(1),
    }


def _canonical(Hc: np.ndarray) -> np.ndarray:
    """The same court with the axes named as court.py wants them.

    y = 0 at the end nearer to the camera, x = 0 on the camera's left.
    """
    def flip(H, fx, fy):
        S = np.eye(3)
        if fx:
            S[0, 0], S[0, 2] = -1, WIDTH
        if fy:
            S[1, 1], S[1, 2] = -1, LENGTH
        return H @ S

    ends = np.array([(WIDTH / 2, 0.0), (WIDTH / 2, LENGTH)])
    _, depth = _project(Hc, ends)
    if depth[0] > depth[1]:  # y = 0 must be the nearer end
        Hc = flip(Hc, True, True)
    # With the image's y pointing down, a floor seen from above is mirrored.
    c = np.array([(WIDTH / 2, LENGTH / 2), (WIDTH / 2 + 0.1, LENGTH / 2), (WIDTH / 2, LENGTH / 2 + 0.1)])
    p, _ = _project(Hc, c)
    ex, ey = p[1] - p[0], p[2] - p[0]
    if ex[0] * ey[1] - ex[1] * ey[0] > 0:
        Hc = flip(Hc, True, False)
    return Hc / Hc[2, 2] if abs(Hc[2, 2]) > 1e-12 else Hc


def _plausible(Hc: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """True for homographies that a real camera above the floor could produce.

    Same reasoning as camera.py: with square pixels and the principal point
    in the image centre, the two floor axes must come out orthogonal and
    equally long, which fixes the focal length. Fits that squash the court
    into a sliver, or onto a single white line, give an absurd focal length,
    camera height or badly non-orthogonal axes.
    """
    Hc = np.asarray(Hc, dtype=np.float64).reshape(-1, 3, 3)
    w, h = size
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
    G = T @ Hc
    G = G / np.linalg.norm(G[:, :, :2], axis=(1, 2))[:, None, None]
    h1, h2 = G[:, :, 0], G[:, :, 1]
    a = np.stack([h1[:, 0] * h2[:, 0] + h1[:, 1] * h2[:, 1],
                  h1[:, 0] ** 2 + h1[:, 1] ** 2 - h2[:, 0] ** 2 - h2[:, 1] ** 2], 1)
    b = np.stack([h1[:, 2] * h2[:, 2], h1[:, 2] ** 2 - h2[:, 2] ** 2], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_f2 = -(a * b).sum(1) / (a * a).sum(1)
        f = 1 / np.sqrt(inv_f2)
        k = np.stack([1 / f, 1 / f, np.ones_like(f)], 1)  # K^-1 with the principal point removed
        r1, r2, t = h1 * k, h2 * k, G[:, :, 2] * k
        n1, n2 = np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)
        skew = np.abs((r1 * r2).sum(1)) / (n1 * n2)
        ratio = n1 / n2
        normal = np.cross(r1 / n1[:, None], r2 / n2[:, None])
        height = np.abs((normal * t).sum(1)) / ((n1 + n2) / 2)
        ok = (inv_f2 > 0) & (f > 0.25 * w) & (f < 8 * w) & (height > 0.3) & (height < 60)
        ok &= (skew < 0.3) & (ratio > 0.7) & (ratio < 1.4)
    return ok & np.isfinite(f)


# ---------------------------------------------------------------- refinement

def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    t = np.clip(((p - a) @ d) / max(float(d @ d), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * d), axis=1)


def _lm(fn, theta: np.ndarray, wts: np.ndarray, huber: float, iters: int = 12) -> np.ndarray:
    """Levenberg-Marquardt with Huber weights and a numeric Jacobian.

    `fn(theta)` returns the residuals already multiplied by `wts`; the Huber
    threshold applies to the unweighted residuals.
    """
    n = len(theta)
    lam = 1e-3
    for _ in range(iters):
        r0 = fn(theta)
        raw = np.abs(r0 / wts)
        rob = np.sqrt(np.where(raw <= huber, 1.0, huber / np.maximum(raw, 1e-12)))
        J = np.empty((len(r0), n))
        for k in range(n):
            dth = theta.copy()
            eps = 1e-6 * max(1.0, abs(theta[k]))
            dth[k] += eps
            J[:, k] = (fn(dth) - r0) / eps
        Jw, rw = J * rob[:, None], r0 * rob
        A, g = Jw.T @ Jw, Jw.T @ rw
        cost0 = float(rw @ rw)
        improved = False
        step = np.zeros(n)
        for _ in range(8):
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r1 = fn(theta + step) * rob
            if np.isfinite(r1).all() and float(r1 @ r1) < cost0:
                theta, lam, improved = theta + step, lam / 3, True
                break
            lam *= 10
        if not improved or np.abs(step).max() < 1e-7:
            break
    return theta


def refine(Hc: np.ndarray, points: np.ndarray, size: tuple[int, int],
           radii: tuple[float, ...] = (12.0, 8.0, 5.0, 3.0), scale: float | None = None) -> np.ndarray:
    """Fit the homography to the mask pixels that lie next to the projected lines.

    `points` are (x, y) pixels on the centre of the white strokes. Each round
    gives every pixel within `radius` of a projected court line to that line
    and then moves the homography to bring the pixels onto their lines
    (Levenberg-Marquardt with Huber weights). The radius shrinks so a coarse
    start can still pull in and the final fit ignores clutter.
    """
    w, h = size
    scale = scale or w / REF_WIDTH
    # normalised coordinates keep the 8 parameters at similar magnitudes
    Ti = np.array([[2 / w, 0, -1], [0, 2 / w, -h / w], [0, 0, 1.0]])
    Tc = np.array([[1 / LENGTH, 0, -WIDTH / 2 / LENGTH], [0, 1 / LENGTH, -0.5], [0, 0, 1.0]])
    court_lines_h = np.cross(np.c_[_SEGMENTS[:, 0], np.ones(len(_SEGMENTS))], np.c_[_SEGMENTS[:, 1], np.ones(len(_SEGMENTS))])
    lines_n = court_lines_h @ np.linalg.inv(Tc)  # l' = T^-T l
    Hn = Ti @ Hc @ np.linalg.inv(Tc)
    if abs(Hn[2, 2]) < 1e-9:
        return Hc
    theta = (Hn / Hn[2, 2]).ravel()[:8]
    pn_all = np.c_[points, np.ones(len(points))] @ Ti.T

    def to_h(th):
        return np.append(th, 1.0).reshape(3, 3)

    def residuals(th, pn, which, wts):
        Hinv = np.linalg.inv(to_h(th))
        L = lines_n @ Hinv  # image lines of the model lines
        L = L / np.linalg.norm(L[:, :2], axis=1, keepdims=True)
        return (L[which] * pn).sum(1) * wts

    corners = np.array([(0, 0), (WIDTH, 0), (WIDTH, LENGTH), (0, LENGTH)], dtype=np.float64)
    for radius in radii + (radii[-1],) * 4:  # the last radius again until nothing moves
        r = radius * scale
        H_now = np.linalg.inv(Ti) @ to_h(theta) @ Tc
        before = theta.copy()
        ends, depth = _project(H_now, _SEGMENTS.reshape(-1, 2))
        ends, depth = ends.reshape(-1, 2, 2), depth.reshape(-1, 2)
        best = np.full(len(points), np.inf)
        which = np.full(len(points), -1)
        for i, (a, b) in enumerate(ends):
            if depth[i].min() <= 0:
                continue
            d = _segment_distance(points, a, b)
            closer = d < best
            best[closer], which[closer] = d[closer], i
        sel = best <= r
        if sel.sum() < 30 or len(np.unique(which[sel])) < 4:
            break
        pn, wh = pn_all[sel], which[sel]
        counts = np.bincount(wh, minlength=len(_SEGMENTS)).astype(np.float64)
        wts = 1.0 / np.sqrt(counts[wh])  # a long near line must not drown out a short far one
        huber = 1.5 * scale * 2 / w
        theta = _lm(lambda th: residuals(th, pn, wh, wts), theta, wts, huber)
        moved = _project(np.linalg.inv(Ti) @ to_h(theta) @ Tc, corners)[0] - _project(np.linalg.inv(Ti) @ to_h(before) @ Tc, corners)[0]
        if radius == radii[-1] and np.abs(moved).max() < 0.05:
            break
    return np.linalg.inv(Ti) @ to_h(theta) @ Tc


# ---------------------------------------------------------------- lens

def coarse_k1(mask: np.ndarray, max_segments: int = 400) -> tuple[float, float]:
    """First guess of the lens distortion from the line pieces alone: (k1, gain).

    Short pieces of a bent line are still straight. With the right k1 the
    pieces of one court line fall on one straight line, so k1 is chosen to
    make as many pairs of pieces collinear as possible (weighted by length).
    `gain` is that score relative to k1 = 0; near 1 means the picture gives
    no reason to assume any distortion.
    """
    h, w = mask.shape
    scale = w / REF_WIDTH
    segs = cv2.HoughLinesP(_centre_lines(mask), 1, np.pi / 360, int(40 * scale),
                           minLineLength=50 * scale, maxLineGap=12 * scale)
    if segs is None or len(segs) < 8:
        return 0.0, 1.0
    ends = segs.reshape(-1, 2, 2).astype(np.float64)
    length = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
    keep = np.argsort(-length)[:max_segments]
    ends, length = ends[keep], length[keep]
    centre, R = np.array([w / 2, h / 2]), np.hypot(w, h) / 2
    r2 = (np.linalg.norm(ends.mean(1) - centre, axis=1) / R) ** 2
    weight = length[:, None] * length[None]
    np.fill_diagonal(weight, 0.0)
    tol = 2.0 * scale

    def score(k: float) -> float:
        try:
            lens = Lens(k)
        except ValueError:
            return -np.inf
        u = lens.undistort(ends.reshape(-1, 2), (w, h)).reshape(-1, 2, 2)
        d = u[:, 1] - u[:, 0]
        n = np.stack([-d[:, 1], d[:, 0]], 1) / np.linalg.norm(d, axis=1, keepdims=True)
        c = -(n * u[:, 0]).sum(1)
        dist = np.abs(np.einsum("id,jed->ije", n, u) + c[:, None, None]).max(2)
        dist = dist / (1 + k * r2)[None]  # back to picture pixels, so shrinking the picture gains nothing
        return float((weight * np.exp(-(dist / tol) ** 2)).sum())

    base = score(0.0)
    if base <= 0:
        return 0.0, 1.0
    grid = np.linspace(-0.2, 0.8, 51)
    best = grid[int(np.argmax([score(k) for k in grid]))]
    fine = np.linspace(best - 0.02, best + 0.02, 21)
    best = fine[int(np.argmax([score(k) for k in fine]))]
    return float(best), score(best) / base


def refine_lens(Hc: np.ndarray, lens: Lens | None, points: np.ndarray, size: tuple[int, int],
                terms: int = 1, radii: tuple[float, ...] = (8.0, 5.0, 3.0, 3.0)) -> tuple[np.ndarray, Lens]:
    """Fit homography and lens together to the line pixels of the recorded picture.

    Like `refine`, but the pixels (`points`, in the picture as recorded) are
    straightened by the lens before their distance to the projected court
    lines is taken, and the first `terms` lens parameters (k1, k2) move along
    with the homography; with `terms` = 0 the lens stays as given. `Hc` maps the court to straightened pixels. Straight lines
    stay straight under any homography, so the bend of the lines determines
    the lens independently of where the court is.
    """
    w, h = size
    scale = w / REF_WIDTH
    lens = lens or Lens()
    Ti = np.array([[2 / w, 0, -1], [0, 2 / w, -h / w], [0, 0, 1.0]])  # picture centre -> origin
    Tc = np.array([[1 / LENGTH, 0, -WIDTH / 2 / LENGTH], [0, 1 / LENGTH, -0.5], [0, 0, 1.0]])
    court_lines_h = np.cross(np.c_[_SEGMENTS[:, 0], np.ones(len(_SEGMENTS))], np.c_[_SEGMENTS[:, 1], np.ones(len(_SEGMENTS))])
    lines_n = court_lines_h @ np.linalg.inv(Tc)
    Hn = Ti @ Hc @ np.linalg.inv(Tc)
    if abs(Hn[2, 2]) < 1e-9:
        return Hc, lens
    theta = np.concatenate([(Hn / Hn[2, 2]).ravel()[:8], [lens.k1, lens.k2][:terms]])
    pn_all = (np.c_[points, np.ones(len(points))] @ Ti.T)[:, :2]
    r2_all = (pn_all ** 2).sum(1) / (1 + (h / w) ** 2)  # (r / half diagonal)^2

    def unpack(th):
        k = [*th[8:], *[lens.k1, lens.k2][terms:]]
        return np.append(th[:8], 1.0).reshape(3, 3), float(k[0]), float(k[1])

    def residuals(th, pn, r2, which, wts):
        Hm, k1, k2 = unpack(th)
        try:
            Lens(k1, k2)
        except ValueError:  # folds the picture: not a lens
            return np.full(len(pn), np.inf)
        g = 1 + k1 * r2 + k2 * r2 * r2
        L = lines_n @ np.linalg.inv(Hm)
        L = L / np.linalg.norm(L[:, :2], axis=1, keepdims=True)
        # divided by g: distances in picture pixels, so shrinking the picture gains nothing
        return ((L[which, :2] * pn).sum(1) + L[which, 2] / g) * wts

    for radius in radii:
        Hm, k1, k2 = unpack(theta)
        now = Lens(k1, k2)
        H_now = np.linalg.inv(Ti) @ Hm @ Tc
        flat = now.undistort(points, size)
        ends, depth = _project(H_now, _SEGMENTS.reshape(-1, 2))
        ends, depth = ends.reshape(-1, 2, 2), depth.reshape(-1, 2)
        best = np.full(len(points), np.inf)
        which = np.full(len(points), -1)
        for i, (a, b) in enumerate(ends):
            if depth[i].min() <= 0:
                continue
            d = _segment_distance(flat, a, b)
            closer = d < best
            best[closer], which[closer] = d[closer], i
        sel = best <= radius * scale
        if sel.sum() < 30 or len(np.unique(which[sel])) < 4:
            break
        pn, r2, wh = pn_all[sel], r2_all[sel], which[sel]
        counts = np.bincount(wh, minlength=len(_SEGMENTS)).astype(np.float64)
        wts = 1.0 / np.sqrt(counts[wh])
        theta = _lm(lambda th: residuals(th, pn, r2, wh, wts), theta, wts, 1.5 * scale * 2 / w, iters=20)
    Hm, k1, k2 = unpack(theta)
    return np.linalg.inv(Ti) @ Hm @ Tc, Lens(k1, k2)


# ---------------------------------------------------------------- top level

@dataclass
class AutoCalibration:
    calib: CourtCalibration
    confidence: float  # 0..1: share of the visible court lines that lie on white pixels
    mean_distance_px: float  # projected lines to the nearest white pixel (capped at 12 px)
    lines_found: int  # painted court lines confirmed in the image
    lines_visible: int
    candidates: int
    mask: np.ndarray = field(repr=False, default=None)


def _summary(Hc: np.ndarray, dist: np.ndarray, tol: float, valid: np.ndarray | None = None) -> dict:
    """Final quality numbers for one homography.

    Stricter than the ranking score: where two model lines project onto the
    same pixels (within 2 tol), only the first may claim the white line under
    them. Otherwise a court squashed flat scores well by laying several of its
    lines on one stroke.
    """
    uv, depth = _project(Hc, _SAMPLES)
    ends, end_depth = _project(Hc, _SEGMENTS.reshape(-1, 2))
    ends, end_depth = ends.reshape(-1, 2, 2), end_depth.reshape(-1, 2)
    own = np.ones(len(_SAMPLES), bool)
    for i, (a, b) in enumerate(ends):
        if end_depth[i].min() > 0:
            later = _SAMPLE_LINE > i
            own[later] &= _segment_distance(uv[later], a, b) > 2 * tol
    o = _overlap(Hc, dist, tol, own, valid)
    lt, lo = o["line_total"][0], o["line_on"][0]
    seen = lt > 20 * tol
    found = seen & (lo >= 0.5 * lt)
    return {
        "confidence": float(o["on"][0] / max(o["total"][0], 1e-9)),
        "mean_dist": float(o["mean_dist"][0]),
        "found": found, "seen": seen,
        "score": float(o["on"][0] - 0.5 * o["off"][0]),
    }


def _marker_points(Hc: np.ndarray, found: np.ndarray, lens: Lens | None = None,
                   size: tuple[int, int] | None = None) -> dict[str, tuple[float, float]]:
    """Image positions of the named markers whose two lines were both found.

    `Hc` maps the court to straightened pixels; with a lens the positions are
    bent back into the recorded picture of `size`.
    """
    seg_along = {round(float(s[0, 0]), 3) for s, ok in zip(_SEGMENTS[_ALONG], found[_ALONG]) if ok}
    seg_across = {round(float(s[0, 1]), 3) for s, ok in zip(_SEGMENTS[~_ALONG], found[~_ALONG]) if ok}
    names = [n for n, (x, y) in MARKERS.items() if round(x, 3) in seg_along and round(y, 3) in seg_across]
    spread = {MARKERS[n][0] for n in names}, {MARKERS[n][1] for n in names}
    if len(names) < 4 or min(len(s) for s in spread) < 2:
        names = [n for n, (_, y) in MARKERS.items() if y != NET_Y]
    uv, _ = _project(Hc, np.array([MARKERS[n] for n in names]))
    if lens is not None:
        uv = lens.distort(uv, size)
    return {n: (round(float(u), 2), round(float(v), 2)) for n, (u, v) in zip(names, uv)}


@dataclass
class _Found:
    """A court found in one (downscaled) frame, before it is turned into a calibration."""

    Hc: np.ndarray  # court -> straightened pixels of the downscaled frame
    lens: Lens | None
    summary: dict
    mean_dist: float  # px of the downscaled frame
    candidates: int
    mask: np.ndarray

    @property
    def ok(self) -> bool:
        found = self.summary["found"]
        return bool(self.summary["confidence"] >= MIN_CONFIDENCE and found.sum() >= MIN_LINES
                    and found[_ALONG].sum() >= 2 and found[~_ALONG].sum() >= 2)


def _search(frame: np.ndarray, hint: tuple[float, float] | None, refine_top: int,
            valid: np.ndarray | None = None, scale: float | None = None) -> tuple[np.ndarray, dict, int, np.ndarray]:
    """The court in a frame without lens distortion: (court -> image, summary, candidates, mask)."""
    h, w = frame.shape[:2]
    scale = scale or w / REF_WIDTH
    tol = 3.0 * scale
    mask = line_mask(frame, scale)
    lines = detect_lines(mask, scale=scale)
    families = _families(lines) if len(lines) >= 4 else []
    if len(families) < 2:
        raise AutoCalibrationError("画面里找不到两个方向的球场线")
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    origin = np.array([ln.mid for ln in lines]).mean(0)

    cands = []
    for a in range(len(families)):
        for b in range(a + 1, len(families)):
            (vp_a, mem_a), (vp_b, mem_b) = families[a], families[b]
            horizon = np.cross(vp_a, vp_b)
            if np.linalg.norm(horizon[:2]) < 1e-12:
                continue
            horizon = horizon / np.linalg.norm(horizon[:2])
            rho_a = _offsets(lines, mem_a, horizon, origin, w)
            rho_b = _offsets(lines, mem_b, horizon, origin, w)
            for (m1, r1), (m2, r2) in (((mem_a, rho_a), (mem_b, rho_b)), ((mem_b, rho_b), (mem_a, rho_a))):
                along = _assignments(lines, m1, r1, LONG_X, 0.15)
                across = _assignments(lines, m2, r2, CROSS_Y, 0.22)
                for u in along:
                    for v in across:
                        H = _homography(lines, u, v)
                        if H is not None and np.isfinite(H).all():
                            cands.append(H)
    if not cands:
        raise AutoCalibrationError("检测到的线拼不出一个球场")
    cands = np.array(cands)
    coarse = _overlap(cands, dist, 3 * tol, valid=valid)
    score = coarse["on"] - 0.5 * coarse["off"]
    score[(coarse["visible"] < 0.25) | ~_plausible(cands, (w, h))] = -np.inf
    if hint is not None:
        for i, H in enumerate(cands):
            xy = cv2.perspectiveTransform(np.array([[hint]], dtype=np.float64), np.linalg.inv(H))[0, 0]
            if not (-0.2 <= xy[0] <= WIDTH + 0.2 and -0.2 <= xy[1] <= LENGTH + 0.2):
                score[i] = -np.inf

    points = np.argwhere(_centre_lines(mask) > 0)[:, ::-1].astype(np.float64)
    if len(points) > 40000:
        points = points[np.random.default_rng(0).choice(len(points), 40000, replace=False)]
    best = None
    tried: list[np.ndarray] = []
    corners = np.array([(0, 0), (WIDTH, 0), (WIDTH, LENGTH), (0, LENGTH)], dtype=np.float64)
    for i in np.argsort(-score):
        if not np.isfinite(score[i]) or len(tried) >= refine_top:
            break
        H0 = _canonical(cands[i])
        c0, _ = _project(H0, corners)
        if any(np.abs(c0 - t).max() < 6 * tol for t in tried):  # same court as one already refined
            continue
        tried.append(c0)
        H1 = _canonical(refine(H0, points, (w, h), scale=scale))
        s = _summary(H1, dist, tol, valid)
        if not _plausible(H1, (w, h))[0]:
            continue
        if best is None or s["score"] > best[1]["score"]:
            best = (H1, s)
    if best is None:
        raise AutoCalibrationError("提示点不在任何找到的球场里" if hint is not None else "检测到的线拼不出一个球场")
    return best[0], best[1], len(cands), mask


def _find(frame: np.ndarray, hint: tuple[float, float] | None, lens: Lens | None, refine_top: int) -> _Found:
    """The court in a frame whose lens is known (None = no distortion)."""
    flat, A, valid = rectify(frame, lens)
    if hint is not None and lens is not None:
        q = np.linalg.solve(A, np.append(lens.undistort([hint], frame.shape[1::-1])[0], 1.0))
        hint = (float(q[0] / q[2]), float(q[1] / q[2]))
    Hc, s, n, mask = _search(flat, hint, refine_top, valid, frame.shape[1] / REF_WIDTH)
    return _Found(A @ Hc, lens, s, s["mean_dist"], n, mask)


def _with_lens(frame: np.ndarray, Hc: np.ndarray, lens: Lens | None, candidates: int) -> _Found | None:
    """Score a court + lens from `refine_lens` the same way as a plain search result."""
    scale = frame.shape[1] / REF_WIDTH
    flat, A, valid = rectify(frame, lens)
    h, w = flat.shape[:2]
    mask = line_mask(flat, scale)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    points = np.argwhere(_centre_lines(mask) > 0)[:, ::-1].astype(np.float64)
    if len(points) > 40000:
        points = points[np.random.default_rng(0).choice(len(points), 40000, replace=False)]
    H1 = _canonical(refine(_canonical(np.linalg.solve(A, Hc)), points, (w, h), scale=scale))
    if not _plausible(H1, (w, h))[0]:
        return None
    s = _summary(H1, dist, 3.0 * scale, valid)
    return _Found(A @ H1, lens, s, s["mean_dist"], candidates, mask)


def _stroke_distance(f: _Found, points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Distance of every stroke-centre pixel of the recorded picture to the nearest court line."""
    flat = points if f.lens is None else f.lens.undistort(points, size)
    ends, depth = _project(f.Hc, _SEGMENTS.reshape(-1, 2))
    ends, depth = ends.reshape(-1, 2, 2), depth.reshape(-1, 2)
    best = np.full(len(points), np.inf)
    for i, (a, b) in enumerate(ends):
        if depth[i].min() > 0:
            best = np.minimum(best, _segment_distance(flat, a, b))
    return best


def _clearly_better(new: _Found, old: _Found | None, points: np.ndarray, size: tuple[int, int]) -> bool:
    """Is the court with a lens model enough of an improvement to believe the lens?

    Compared on the stroke-centre pixels near the court lines of either fit
    (the distance to the nearest white pixel, as used for the confidence,
    hides a bend that is smaller than the stroke is wide).
    """
    if old is None or (new.ok and not old.ok):
        return True
    # (a half-hidden line may drop just below the "found" mark; more than one is a worse court)
    if not new.ok or new.summary["found"].sum() < old.summary["found"].sum() - 1:
        return False
    if new.summary["confidence"] < old.summary["confidence"] - 0.02:
        return False
    r = 6.0 * size[0] / REF_WIDTH
    d_new, d_old = _stroke_distance(new, points, size), _stroke_distance(old, points, size)
    near = (d_new < r) | (d_old < r)
    if near.sum() < 200:
        return False
    return float(np.minimum(d_new[near], r).mean()) <= LENS_GAIN * float(np.minimum(d_old[near], r).mean())


def _estimate_lens(frame: np.ndarray, plain: _Found | None, hint, refine_top: int, coarse: bool = True) -> _Found | None:
    """The best court with a lens estimated from the lines, or `plain` when no lens is called for."""
    h, w = frame.shape[:2]
    mask = line_mask(frame)
    points = np.argwhere(_centre_lines(mask) > 0)[:, ::-1].astype(np.float64)
    if len(points) > 40000:
        points = points[np.random.default_rng(0).choice(len(points), 40000, replace=False)]
    starts: list[_Found] = [plain] if plain is not None else []
    k0, gain = coarse_k1(mask) if coarse else (0.0, 1.0)
    if abs(k0) >= 0.03 and gain > 1.2:  # visibly bent lines: search again on a straightened picture
        try:
            starts.append(_find(frame, hint, Lens(k0), refine_top))
        except AutoCalibrationError:
            pass
    best = bar = plain  # `bar` is what a lens has to beat; without a lens the answer stays `plain` itself
    if plain is not None:
        # The same extra fitting without a lens, so that a lens is not credited
        # with what a few more rounds on the homography alone would have gained.
        again = _with_lens(frame, refine_lens(plain.Hc, None, points, (w, h), terms=0)[0], None, plain.candidates)
        if again is not None and again.ok >= plain.ok and again.summary["score"] > plain.summary["score"]:
            bar = again
    for start in starts:
        Hc, lens = refine_lens(start.Hc, start.lens, points, (w, h))
        if abs(lens.k1) < MIN_LENS_K:
            continue
        cand = _with_lens(frame, Hc, lens, start.candidates)
        if cand is None or not _clearly_better(cand, bar, points, (w, h)):
            continue
        best = bar = cand
        # k2 has to earn its place over k1 alone, the same way k1 did over no lens
        Hc, lens = refine_lens(Hc, lens, points, (w, h), terms=2)
        cand = _with_lens(frame, Hc, lens, start.candidates)
        if cand is not None and _clearly_better(cand, bar, points, (w, h)):
            best = bar = cand
    return best


def estimate_lens(frame: np.ndarray, calib: CourtCalibration) -> Lens | None:
    """Lens distortion of a frame whose court is already known roughly (clicked by hand).

    The clicked court tells which white pixels belong to which court line;
    the lens is then fitted to the bend of those lines. None when the lines
    are not clearly closer to a court with a lens than to one without, or
    when the court lines cannot be seen well enough to tell.
    """
    full_h, full_w = frame.shape[:2]
    shrink = min(1.0, REF_WIDTH / full_w)
    if shrink < 1:
        frame = cv2.resize(frame, (int(round(full_w * shrink)), int(round(full_h * shrink))), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    mask = line_mask(frame)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    points = np.argwhere(_centre_lines(mask) > 0)[:, ::-1].astype(np.float64)
    if len(points) > 40000:
        points = points[np.random.default_rng(0).choice(len(points), 40000, replace=False)]
    Hc = _canonical(refine(_canonical(np.diag([shrink, shrink, 1.0]) @ calib.H_inv), points, (w, h)))
    s = _summary(Hc, dist, 3.0 * w / REF_WIDTH)
    plain = _Found(Hc, None, s, s["mean_dist"], 1, mask)
    best = _estimate_lens(frame, plain, None, 0, coarse=False)
    return best.lens if best.ok else None  # the court could not be seen well: no opinion


def auto_calibrate(frame: np.ndarray, hint: tuple[float, float] | None = None,
                   refine_top: int = 6, lens: Lens | str | None = "auto") -> AutoCalibration:
    """Find the court in a frame (BGR). Raises AutoCalibrationError when unsure.

    `hint` is a pixel inside the wanted court, for halls with several courts.
    `lens`: "auto" estimates the lens distortion from the court lines and
    uses it only when it clearly helps; None assumes there is none; a `Lens`
    is used as given.
    """
    full_h, full_w = frame.shape[:2]
    shrink = min(1.0, REF_WIDTH / full_w)
    if shrink < 1:
        frame = cv2.resize(frame, (int(round(full_w * shrink)), int(round(full_h * shrink))), interpolation=cv2.INTER_AREA)
        hint = (hint[0] * shrink, hint[1] * shrink) if hint else None
    if isinstance(lens, str):
        if lens != "auto":
            raise ValueError(f"lens 应为 'auto'、None 或 Lens，得到 {lens!r}")
        plain, failure = None, None
        try:
            plain = _find(frame, hint, None, refine_top)
        except AutoCalibrationError as e:
            failure = e
        best = _estimate_lens(frame, plain, hint, refine_top)
        if best is None:
            raise failure
    else:
        best = _find(frame, hint, lens, refine_top)
    res = _result(best, shrink, (full_w, full_h))
    if not best.ok:
        s = best.summary
        err = AutoCalibrationError(
            f"结果不可靠（线条重合度 {s['confidence']:.0%}，只对上 {int(s['found'].sum())}/{int(s['seen'].sum())} 条线）")
        err.result = res
        raise err
    return res


def _result(f: _Found, shrink: float, full_size: tuple[int, int]) -> AutoCalibration:
    S = np.diag([1 / shrink, 1 / shrink, 1.0])
    s = f.summary
    calib = CourtCalibration(_marker_points(S @ f.Hc, s["found"], f.lens, full_size), full_size, f.lens)
    return AutoCalibration(calib, s["confidence"], f.mean_dist / shrink, int(s["found"].sum()), int(s["seen"].sum()),
                           f.candidates, f.mask)


def track(frame: np.ndarray, calib: CourtCalibration, min_confidence: float = 0.8) -> AutoCalibration | None:
    """Find the same court in another frame of the video (camera drift).

    First only the refinement runs, starting from `calib`; it follows shifts
    of up to about ten pixels. When that does not land on the lines, the full
    search runs with the old court centre as the hint. None when both fail.
    """
    full_h, full_w = frame.shape[:2]
    shrink = min(1.0, REF_WIDTH / full_w)
    small = frame
    if shrink < 1:
        small = cv2.resize(frame, (int(round(full_w * shrink)), int(round(full_h * shrink))), interpolation=cv2.INTER_AREA)
    scale = small.shape[1] / REF_WIDTH
    flat, A, valid = rectify(small, calib.lens)  # the lens belongs to the camera: it does not drift
    h, w = flat.shape[:2]
    mask = line_mask(flat, scale)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    points = np.argwhere(_centre_lines(mask) > 0)[:, ::-1].astype(np.float64)
    S = np.diag([shrink, shrink, 1.0])
    Hc = _canonical(refine(np.linalg.solve(A, S @ calib.H_inv), points, (w, h), scale=scale))
    s = _summary(Hc, dist, 3.0 * scale, valid)
    if s["confidence"] >= min_confidence:
        return _result(_Found(A @ Hc, calib.lens, s, s["mean_dist"], 1, mask), shrink, (full_w, full_h))
    centre = calib.to_image(np.array([(WIDTH / 2, LENGTH / 4)]))[0]  # near half: large in the picture
    try:
        return auto_calibrate(frame, (float(centre[0]), float(centre[1])), lens=calib.lens)
    except AutoCalibrationError:
        return None


def intersection_distance_px(a: CourtCalibration, b: CourtCalibration, in_frame: bool = False) -> tuple[float, float]:
    """Mean and max pixel distance between two calibrations at the court line crossings.

    All 30 crossings of the painted lines are compared; with `in_frame` only
    those that `a` puts inside the picture (inf, inf when there are none).
    """
    grid = np.array([(x, y) for x in LONG_X for y in CROSS_Y])
    pa, pb = a.to_image(grid), b.to_image(grid)
    if in_frame:
        w, h = a.frame_size
        keep = (pa[:, 0] >= 0) & (pa[:, 0] < w) & (pa[:, 1] >= 0) & (pa[:, 1] < h)
        pa, pb = pa[keep], pb[keep]
    if len(pa) == 0:
        return float("inf"), float("inf")
    d = np.linalg.norm(pa - pb, axis=1)
    return float(d.mean()), float(d.max())


def mask_agreement(calib: CourtCalibration, frame: np.ndarray) -> tuple[float, float]:
    """(share of the visible court lines on white pixels, mean distance in px) for any calibration."""
    flat, A, valid = rectify(frame, calib.lens)
    mask = line_mask(flat, frame.shape[1] / REF_WIDTH)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    s = _summary(np.linalg.solve(A, calib.H_inv), dist, 3.0 * frame.shape[1] / REF_WIDTH, valid)
    return s["confidence"], s["mean_dist"]
