"""Step 3: doubles positioning report for each pair.

Uses the cleaned, smoothed positions of tracking.py / movement.py and only
in-play time. Per pair (the two players on one half):

  * formation: the angle of the line between the partners. Steep (> 60 deg
    from the net) = front-back (attacking shape), flat (< 30 deg) =
    side-by-side (defensive shape), in between = transition;
  * rotations: who is the front player in front-back, who is on the left in
    side-by-side; a change that lasts >= ROLE_MIN_S counts as one rotation;
  * partner spacing: distance between the two (too close < 1.5 m: both cover
    the same shuttle; too far > 4.5 m: a hole opens between them);
  * open court: for every frame the point of the half that is farthest from
    both players (the biggest hole), how far it is, and in which zone;
  * coverage: share of the half that is closer to each player (Voronoi).

Without the shuttle the report cannot say who was attacking; step 5/6 will
add that (and per-shot rotations) from the shuttle trajectory.
"""

from __future__ import annotations

import numpy as np

from .court import LENGTH, NET_Y, WIDTH
from .movement import _runs, smooth_track

FRONT_BACK_DEG = 60.0
SIDE_BY_SIDE_DEG = 30.0
ROLE_MIN_S = 0.5  # a role change must last this long to count as a rotation
CLOSE_M, FAR_M = 1.5, 4.5
GRID_M = 0.25
ZONES = [(d, s) for d in ("front", "mid", "back") for s in ("left", "right")]
ZONE_ZH = {"front": "前场", "mid": "中场", "back": "后场", "left": "左", "right": "右"}


def _half_grid(side: int) -> np.ndarray:
    """Grid points (M, 2) covering one half (singles + doubles area, inside the lines)."""
    xs = np.arange(GRID_M / 2, WIDTH, GRID_M)
    ys = np.arange(GRID_M / 2, NET_Y, GRID_M) if side == 0 else np.arange(NET_Y + GRID_M / 2, LENGTH, GRID_M)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], 1)


def _zone(points: np.ndarray, side: int) -> list[tuple[str, str]]:
    """(depth, lateral) zone from the pair's own view facing the net."""
    depth = NET_Y - points[:, 1] if side == 0 else points[:, 1] - NET_Y
    d = np.clip((depth // (NET_Y / 3)).astype(int), 0, 2)
    left = points[:, 0] < WIDTH / 2 if side == 0 else points[:, 0] > WIDTH / 2
    return [(("front", "mid", "back")[k], "left" if lft else "right") for k, lft in zip(d, left)]


def _debounced_changes(role: np.ndarray, valid: np.ndarray, t: np.ndarray, min_s: float) -> list[int]:
    """Indices where `role` (0/1 per frame) switches to a value that is then held
    for at least `min_s` of valid time. Frames where `valid` is False (another
    formation, not in play) are skipped and do not count towards the duration,
    so passing through front-back for a moment while crossing sideways is not
    a front/back rotation."""
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return []
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    changes, current = [], None
    k = 0
    while k < len(idx):
        r = role[idx[k]]
        m = k
        while m + 1 < len(idx) and role[idx[m + 1]] == r:
            m += 1
        held = (m - k + 1) * dt
        if held >= min_s or current is None and m == len(idx) - 1:
            if current is not None and r != current:
                changes.append(int(idx[k]))
            current = r
        k = m + 1
    return changes


def pair_report(a_xy: np.ndarray, b_xy: np.ndarray, side: int, t: np.ndarray, in_play: np.ndarray,
                ids: tuple[int, int], names: tuple[str, str]) -> dict:
    """a_xy, b_xy: smoothed positions (F, 2) of the two partners, NaN when unknown."""
    ok = in_play & np.isfinite(a_xy).all(1) & np.isfinite(b_xy).all(1)
    n = int(ok.sum())
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    base = {"ids": list(ids), "names": list(names), "side": "near" if side == 0 else "far",
            "analysed_s": round(n * dt, 2)}
    if n == 0:
        return base | {"note": "回合内两人同时可见的时间为 0"}

    d = b_xy - a_xy
    dist = np.hypot(d[:, 0], d[:, 1])
    angle = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))  # 0 = side by side, 90 = front-back
    fb = ok & (angle > FRONT_BACK_DEG)
    sbs = ok & (angle < SIDE_BY_SIDE_DEG)
    trans = ok & ~fb & ~sbs

    # roles: who is nearer the net / who is on the left (pair's own view)
    a_front = (a_xy[:, 1] > b_xy[:, 1]) if side == 0 else (a_xy[:, 1] < b_xy[:, 1])
    a_left = (a_xy[:, 0] < b_xy[:, 0]) if side == 0 else (a_xy[:, 0] > b_xy[:, 0])
    fb_changes = _debounced_changes(a_front.astype(int), fb, t, ROLE_MIN_S)
    lr_changes = _debounced_changes(a_left.astype(int), sbs, t, ROLE_MIN_S)

    # biggest hole: grid point farthest from both players
    grid = _half_grid(side)
    gap_dist = np.full(len(t), np.nan)
    gap_pt = np.full((len(t), 2), np.nan)
    share_a = np.full(len(t), np.nan)
    for i in np.flatnonzero(ok):
        da = np.hypot(*(grid - a_xy[i]).T)
        db = np.hypot(*(grid - b_xy[i]).T)
        near = np.minimum(da, db)
        k = int(np.argmax(near))
        gap_dist[i], gap_pt[i] = near[k], grid[k]
        share_a[i] = np.mean(da < db)
    zones = _zone(gap_pt[ok], side)
    zone_pct = {f"{dz}_{lz}": round(100 * sum(z == (dz, lz) for z in zones) / n, 1) for dz, lz in ZONES}
    gap_hist, xe, ye = np.histogram2d(gap_pt[ok, 0], gap_pt[ok, 1],
                                      bins=[np.arange(0, WIDTH + 1e-9, 0.5),
                                            np.arange(0, NET_Y + 1e-9, 0.5) + (0 if side == 0 else NET_Y)])

    def pct(m):
        return round(100 * float(m.sum()) / n, 1)

    minutes = n * dt / 60
    return base | {
        "formation_pct": {"front_back": pct(fb), "side_by_side": pct(sbs), "transition": pct(trans)},
        "front_time_pct": {names[0]: round(100 * float((fb & a_front).sum()) / max(1, fb.sum()), 1),
                           names[1]: round(100 * float((fb & ~a_front).sum()) / max(1, fb.sum()), 1)},
        "rotations": {"front_back_swaps": len(fb_changes), "left_right_swaps": len(lr_changes),
                      "per_minute": round((len(fb_changes) + len(lr_changes)) / minutes, 2) if minutes else 0.0,
                      "front_back_swap_times_s": [round(float(t[i]), 2) for i in fb_changes],
                      "left_right_swap_times_s": [round(float(t[i]), 2) for i in lr_changes]},
        "spacing_m": {"mean": round(float(dist[ok].mean()), 2), "median": round(float(np.median(dist[ok])), 2),
                      "p10": round(float(np.percentile(dist[ok], 10)), 2),
                      "p90": round(float(np.percentile(dist[ok], 90)), 2),
                      "too_close_pct": pct(ok & (dist < CLOSE_M)), "too_far_pct": pct(ok & (dist > FAR_M))},
        "open_court": {"mean_gap_m": round(float(np.nanmean(gap_dist[ok])), 2),
                       "p90_gap_m": round(float(np.nanpercentile(gap_dist[ok], 90)), 2),
                       "biggest_gap_zone_pct": zone_pct,
                       "heatmap": {"x_edges": xe.tolist(), "y_edges": ye.tolist(), "counts": gap_hist.T.astype(int).tolist()}},
        "coverage_pct": {names[0]: round(100 * float(np.nanmean(share_a[ok])), 1),
                         names[1]: round(100 * (1 - float(np.nanmean(share_a[ok]))), 1)},
        "_series": {"angle": angle, "dist": dist, "ok": ok, "gap_pt": gap_pt},
    }


def analyse_pairs(tracking, in_play: np.ndarray) -> list[dict]:
    """One report per half that has exactly two tracked players."""
    t = tracking.time_s
    out = []
    for side in (0, 1):
        tracks = [tr for tr in tracking.tracks if tr.side == side]
        if len(tracks) != 2:
            continue
        a, b = tracks
        pa, _ = smooth_track(a.xy, t)
        pb, _ = smooth_track(b.xy, t)
        out.append(pair_report(pa, pb, side, t, in_play, (a.id, b.id),
                               (a.name or f"球员 {a.id}", b.name or f"球员 {b.id}")))
    return out


def summary_lines(rep: dict) -> list[str]:
    if "formation_pct" not in rep:
        return [f"{' / '.join(rep['names'])}：{rep.get('note', '')}"]
    f, r, s, o = rep["formation_pct"], rep["rotations"], rep["spacing_m"], rep["open_court"]
    zone = max(o["biggest_gap_zone_pct"], key=o["biggest_gap_zone_pct"].get)
    dz, lz = zone.split("_")
    return [
        f"{' / '.join(rep['names'])}（{'近端' if rep['side'] == 'near' else '远端'}，回合内 {rep['analysed_s']:.0f} 秒）",
        f"  站位：前后 {f['front_back']:.0f}%，左右 {f['side_by_side']:.0f}%，过渡 {f['transition']:.0f}%",
        f"  轮转：前后换位 {r['front_back_swaps']} 次，左右换位 {r['left_right_swaps']} 次（每分钟 {r['per_minute']:.1f} 次）",
        f"  搭档间距：平均 {s['mean']:.1f} 米（太近 {s['too_close_pct']:.0f}%，太远 {s['too_far_pct']:.0f}% 的时间）",
        f"  最大空当：平均离最近的人 {o['mean_gap_m']:.1f} 米，最常出现在{ZONE_ZH[dz]}{ZONE_ZH[lz]}侧"
        f"（{o['biggest_gap_zone_pct'][zone]:.0f}% 的时间）",
        f"  各自覆盖：" + "，".join(f"{k} {v:.0f}%" for k, v in rep["coverage_pct"].items()),
    ]


def json_ready(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if not k.startswith("_")}


def render_pair_card(rep: dict, out_path) -> None:
    """Half court with the biggest-gap heatmap, plus the key numbers."""
    import matplotlib

    matplotlib.use("Agg")
    import cv2
    import matplotlib.pyplot as plt

    from .card import _setup_font, draw_court

    zh = _setup_font()
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_axes([0.02, 0.03, 0.45, 0.9])
    draw_court(ax)
    if "open_court" in rep:
        hm = rep["open_court"]["heatmap"]
        c = np.asarray(hm["counts"], float)
        if c.max() > 0:
            dens = cv2.GaussianBlur(c, (0, 0), 1.0)
            dens /= dens.max()
            ax.pcolormesh(hm["x_edges"], hm["y_edges"], np.ma.masked_where(dens < 0.05, dens), cmap="Blues",
                          alpha=0.85, zorder=2, vmin=0, vmax=1)
    fig.text(0.52, 0.9, " / ".join(rep["names"]), fontsize=18, weight="semibold")
    side = ("近端" if rep["side"] == "near" else "远端") if zh else rep["side"]
    fig.text(0.52, 0.85, ("双打站位报告 · " if zh else "Doubles positioning · ") + side, fontsize=10, color="#555")
    if "formation_pct" in rep:
        f, r, s_, o = rep["formation_pct"], rep["rotations"], rep["spacing_m"], rep["open_court"]
        zone = max(o["biggest_gap_zone_pct"], key=o["biggest_gap_zone_pct"].get)
        dz, lz = zone.split("_")
        cov = list(rep["coverage_pct"].values())
        rows = [
            ("前后站位" if zh else "Front-back", f"{f['front_back']:.0f}%"),
            ("左右站位" if zh else "Side-by-side", f"{f['side_by_side']:.0f}%"),
            ("轮转" if zh else "Rotations", f"{r['front_back_swaps'] + r['left_right_swaps']}" +
             (f" 次（{r['per_minute']:.1f}/分钟）" if zh else f" ({r['per_minute']:.1f}/min)")),
            ("搭档间距" if zh else "Partner spacing", f"{s_['mean']:.1f} " + ("米" if zh else "m")),
            ("太近 / 太远" if zh else "Too close / far", f"{s_['too_close_pct']:.0f}% / {s_['too_far_pct']:.0f}%"),
            ("最大空当" if zh else "Biggest gap", f"{o['mean_gap_m']:.1f} " + ("米" if zh else "m")),
            ("空当常在" if zh else "Gap usually", (ZONE_ZH[dz] + ZONE_ZH[lz] + "侧") if zh else f"{dz} {lz}"),
            ("各自覆盖" if zh else "Coverage", f"{cov[0]:.0f}% / {cov[1]:.0f}%"),
        ]
        y = 0.76
        for k, v in rows:
            fig.text(0.52, y, k, fontsize=11, color="#555")
            fig.text(0.95, y, v, fontsize=14, weight="semibold", ha="right")
            y -= 0.07
    fig.text(0.52, 0.04, ("蓝色：最大空当出现的位置（离两人都最远的点）；只统计回合内时间" if zh else
                          "Blue: where the biggest open space was; in-play time only"), fontsize=8, color="#888")
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
