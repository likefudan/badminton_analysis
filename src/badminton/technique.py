"""How each stroke was played: arm, swing, body turn, jump, stance, per stroke type.

Phase 1 of the technique analysis (plan: docs/technique.md). For every stroke
from `badminton strokes` this takes the hitter's skeleton from 0.8 s before to
0.4 s after the hit and works out a handful of numbers a player understands:

  overhead strokes (杀球, 高远/平高, 吊球)
    contact_height    wrist above the shoulders at the hit, in torso lengths
    elbow_deg         shoulder-elbow-wrist angle at the hit (as seen in the picture)
    backswing_elbow   how high the racket elbow got above the shoulders while preparing
    chain_ms          when hip, shoulder, elbow and wrist each moved fastest, ms from the hit
    chain_order_ok    those peaks come from the body outwards (hip first, wrist last)
    wrist_speed       fastest wrist speed, torso lengths per second (2D swing speed)
    turn              how far the shoulder line swung round (seen as its sideways width)
    jump              how far the ankles rose before the hit, in torso lengths
    follow_through    how far the wrist swept across the body after the hit
  low and mid-height strokes (放网, 勾对角, 挑球, 推扑, 抽挡)
    contact_height, wrist_speed,
    stance            distance between the ankles at the hit
    knee_deg          the more bent knee's angle at the hit
    trunk_lean        trunk angle from upright at the hit

Everything is measured in the picture of one camera, so angles are what the
camera sees, not the true joint angles (see the plan doc for what 2D can and
cannot tell). Lengths are in torso lengths (mid-shoulder to mid-hip) so people
of different size and at different distances compare. Only strokes whose
joints were seen well enough get numbers; the near-court player by default,
since the far player is too small in the picture.

Drops also get `disguise`: how far the preparation (0.6 s to 0.1 s before the
hit) is from the same player's own smash preparation, compared with how much
their smashes differ from each other (about 1 or less: the drop looks like a smash).

With a reference (`badminton technique-ref`, built from other videos, e.g. pro
matches) every number also gets its percentile there, and numbers outside the
10th..90th percentile are flagged. Without one only a few fixed rules flag
(draft thresholds, to be checked against labelled strokes).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .movement import savgol_coeffs

SCHEMA = "badminton.technique/1"
REF_SCHEMA = "badminton.technique_ref/1"

# COCO-17
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16

OVERHEAD = ("杀球", "高远/平高", "吊球")
LOW = ("放网", "勾对角", "挑球", "推扑", "抽挡")
GROUPS = OVERHEAD + LOW

BEFORE_S, AFTER_S = 0.8, 0.4  # window around the hit
KPT_THR = 0.3  # keypoint score below this counts as not seen
MIN_SEEN = 0.7  # share of the window the core joints must be seen in
CHAIN_WIN = (-0.4, 0.1)  # where the speed peaks of the kinetic chain are looked for (s from the hit)
PREP_WIN = (-0.6, -0.1)  # preparation, for backswing and disguise
TURN_WIN = (-0.6, 0.3)
JUMP_BASE = (-0.8, -0.5)  # ankles before the take-off
JUMP_TOP = (-0.15, 0.1)
FOLLOW_S = 0.25
PREP_SAMPLES = 10
MIN_SMASHES = 3  # a player needs this many smashes for the disguise number

# draft rules used when there is no reference (torso lengths / ms); to be calibrated on labelled strokes
RULES = {
    "contact_low": 1.0,  # overhead: wrist less than one torso length above the shoulders
    "turn_small": 0.25,  # overhead: shoulder line hardly swung round
    "chain_tol_ms": 20,  # a peak may come this much before the previous joint's and still count as in order
}

METRIC_NAMES = {
    "contact_height": "击球点高度（躯干长）",
    "elbow_deg": "击球时肘角（画面里）",
    "backswing_elbow": "引拍时肘高（躯干长）",
    "wrist_speed": "手腕最快速度（躯干长/秒）",
    "turn": "转体幅度",
    "jump": "起跳高度（躯干长）",
    "follow_through": "随挥幅度（躯干长）",
    "stance": "两脚间距（躯干长）",
    "knee_deg": "弯曲那条腿的膝角",
    "trunk_lean": "躯干前倾（度）",
    "disguise": "吊球像杀球的程度（越小越像）",
}
# metric -> True if larger is the "more" direction worth a flag text (for reference flags both ends are reported)
NUMERIC = list(METRIC_NAMES)


# ---------------------------------------------------------------- geometry helpers


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b in degrees."""
    u, v = a - b, c - b
    n = np.linalg.norm(u) * np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / n, -1, 1))))


def _mid(kp: np.ndarray, i: int, j: int) -> np.ndarray:
    return (kp[..., i, :] + kp[..., j, :]) / 2


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def fill_and_smooth(kp: np.ndarray, seen: np.ndarray, fps: float) -> np.ndarray:
    """(n, 17, 2) keypoints with (n, 17) seen mask: linear fill of the gaps (ends held),
    then a Savitzky-Golay smooth over about 0.1 s. Joints never seen stay NaN."""
    out = kp.astype(float).copy()
    n = len(out)
    t = np.arange(n)
    win = min(_odd(max(5, round(0.1 * fps))), n if n % 2 else n - 1)
    coef = savgol_coeffs(win, 2) if win >= 5 else None
    for j in range(out.shape[1]):
        ok = seen[:, j] & np.isfinite(out[:, j]).all(-1)
        if not ok.any():
            out[:, j] = np.nan
            continue
        for d in range(2):
            out[:, j, d] = np.interp(t, t[ok], out[ok, j, d])
            if coef is not None:
                pad = np.pad(out[:, j, d], win // 2, mode="edge")
                out[:, j, d] = np.convolve(pad, coef, mode="valid")
    return out


def _speed(xy: np.ndarray, fps: float) -> np.ndarray:
    """(n, 2) -> (n,) speed in units per second."""
    return np.linalg.norm(np.gradient(xy, axis=0), axis=-1) * fps


# ---------------------------------------------------------------- one stroke


def stroke_window(data: dict, row_of: np.ndarray, det_index: np.ndarray, hit_frame: int, fps: float,
                  before_s: float = BEFORE_S, after_s: float = AFTER_S):
    """Keypoints, scores and times (s from the hit) of one tracked player around a hit.
    Frames the player was not detected in come back NaN / score 0."""
    n_frames = len(row_of)
    f = np.arange(hit_frame - round(before_s * fps), hit_frame + round(after_s * fps) + 1)
    f = f[(f >= 0) & (f < n_frames)]
    rows = row_of[f]
    kp = np.full((len(f), 17, 2), np.nan)
    sc = np.zeros((len(f), 17))
    for k, r in enumerate(rows):
        if r < 0 or r >= len(det_index):
            continue
        d = det_index[r]
        if d < 0:
            continue
        kp[k] = data["keypoints"][r, d]
        sc[k] = data["scores"][r, d]
    return kp, sc, (f - hit_frame) / fps


def racket_side(kp: np.ndarray, t: np.ndarray) -> int:
    """Which wrist holds the racket in this stroke: the one that moved fastest near the hit.
    Returns the COCO index of the right (R_WR) or left (L_WR) wrist."""
    near = (t >= CHAIN_WIN[0]) & (t <= CHAIN_WIN[1])
    fps = 1 / np.median(np.diff(t)) if len(t) > 1 else 30.0
    peaks = [np.nanmax(_speed(kp[:, w], fps)[near]) if near.any() else 0 for w in (R_WR, L_WR)]
    return R_WR if peaks[0] >= peaks[1] else L_WR


def _at(t: np.ndarray, s: float) -> int:
    return int(np.argmin(np.abs(t - s)))


def _between(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (t >= lo - 1e-9) & (t <= hi + 1e-9)


def stroke_metrics(kp_raw: np.ndarray, sc: np.ndarray, t: np.ndarray, fps: float, group: str,
                   wrist: int | None = None) -> dict:
    """The numbers of one stroke (see the module doc). `wrist`: racket wrist, guessed if None.
    Returns {"ok": bool, "reason": str, "racket_hand": "R"/"L", "metrics": {...}, "prep": (k, 17, 2) or None}."""
    seen = sc >= KPT_THR
    core = [L_SH, R_SH, L_HIP, R_HIP]
    kp = fill_and_smooth(kp_raw, seen, fps)
    if len(t) < 5 or not np.isfinite(kp[:, core]).all():
        return {"ok": False, "reason": "击球前后没拍到这个人", "metrics": {}}
    wrist = wrist if wrist is not None else racket_side(kp, t)
    elbow, shoulder = (R_EL, R_SH) if wrist == R_WR else (L_EL, L_SH)
    other_sh = L_SH if shoulder == R_SH else R_SH
    need = core + [elbow, wrist]
    seen_share = float(seen[:, need].all(1).mean())
    hand = "R" if wrist == R_WR else "L"
    if seen_share < MIN_SEEN or not np.isfinite(kp[:, need]).all():
        return {"ok": False, "reason": f"关节只有 {seen_share:.0%} 的时间看得清", "racket_hand": hand, "metrics": {},
                "seen_share": round(seen_share, 2)}

    up = -1.0  # image y grows downwards
    msh, mhip = _mid(kp, L_SH, R_SH), _mid(kp, L_HIP, R_HIP)
    torso = float(np.nanmedian(np.linalg.norm(msh - mhip, axis=-1)))
    if not torso > 0:
        return {"ok": False, "reason": "量不出躯干长度", "metrics": {}}
    h = _at(t, 0.0)
    m: dict = {}
    m["contact_height"] = up * (kp[h, wrist, 1] - msh[h, 1]) / torso
    speed_w = _speed(kp[:, wrist], fps) / torso
    chain = _between(t, *CHAIN_WIN)
    m["wrist_speed"] = float(np.max(speed_w[chain]))

    if group in OVERHEAD:
        m["elbow_deg"] = _angle(kp[h, shoulder], kp[h, elbow], kp[h, wrist])
        prep = _between(t, *PREP_WIN)
        m["backswing_elbow"] = float(np.max(up * (kp[prep, elbow, 1] - msh[prep, 1]))) / torso if prep.any() else None
        parts = {"hip": mhip, "shoulder": kp[:, shoulder], "elbow": kp[:, elbow], "wrist": kp[:, wrist]}
        idx = np.flatnonzero(chain)
        peaks = {k: float(t[idx[np.argmax(_speed(v, fps)[idx])]]) for k, v in parts.items()}
        m["chain_ms"] = {k: round(v * 1000) for k, v in peaks.items()}
        order = [peaks[k] for k in ("hip", "shoulder", "elbow", "wrist")]
        tol = RULES["chain_tol_ms"] / 1000 + 0.5 / fps
        m["chain_order_ok"] = bool(all(b >= a - tol for a, b in zip(order, order[1:])))
        turn = _between(t, *TURN_WIN)
        sx = (kp[turn, shoulder, 0] - kp[turn, other_sh, 0]) / torso
        m["turn"] = float(np.ptp(sx)) if turn.any() else None
        ank = (kp[:, L_AN, 1] + kp[:, R_AN, 1]) / 2
        if np.isfinite(ank).all() and _between(t, *JUMP_BASE).any():
            base = np.median(ank[_between(t, *JUMP_BASE)])
            top = np.min(ank[_between(t, *JUMP_TOP)])
            m["jump"] = max(0.0, float(base - top) / torso)
        else:
            m["jump"] = None
        after = _between(t, 1e-9, FOLLOW_S)
        if t[-1] >= FOLLOW_S - 1 / fps and after.any():
            # across the body = towards the other shoulder's side, taken before the turn starts
            across = -(np.sign(kp[0, shoulder, 0] - kp[0, other_sh, 0]) or 1.0)
            m["follow_through"] = float(np.max(across * (kp[after, wrist, 0] - kp[h, wrist, 0])) / torso)
        else:
            m["follow_through"] = None
    else:
        a, b = kp[h, L_AN], kp[h, R_AN]
        m["stance"] = float(np.linalg.norm(a - b) / torso) if np.isfinite([a, b]).all() else None
        knees = [_angle(kp[h, hip], kp[h, kn], kp[h, an]) for hip, kn, an in ((L_HIP, L_KN, L_AN), (R_HIP, R_KN, R_AN))]
        knees = [k for k in knees if np.isfinite(k)]
        m["knee_deg"] = float(min(knees)) if knees else None
        v = msh[h] - mhip[h]
        m["trunk_lean"] = float(np.degrees(np.arctan2(abs(v[0]), -v[1])))

    prep_pose = None
    sel = _between(t, *PREP_WIN)
    if sel.sum() >= 2:
        ts = np.linspace(PREP_WIN[0], PREP_WIN[1], PREP_SAMPLES)
        rel = (kp - mhip[:, None, :]) / torso
        if hand == "L":
            rel[..., 0] *= -1
        prep_pose = np.stack([[np.interp(ts, t, rel[:, j, d]) for d in range(2)] for j in range(17)]).transpose(2, 0, 1)
    m = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}
    return {"ok": True, "reason": "", "racket_hand": hand, "metrics": m, "seen_share": round(seen_share, 2),
            "torso_px": round(torso, 1), "prep": prep_pose}


# ---------------------------------------------------------------- disguise (drops vs own smashes)


def add_disguise(results: list[dict]) -> None:
    """Per player: distance of each drop's preparation to the mean smash preparation,
    divided by the typical distance of the smashes themselves to that mean."""
    by_player: dict = {}
    for r in results:
        if r.get("prep") is not None:
            by_player.setdefault(r["player_id"], []).append(r)
    for rs in by_player.values():
        smashes = np.array([r["prep"] for r in rs if r["group"] == "杀球"])
        if len(smashes) < MIN_SMASHES:
            continue
        mean = np.nanmean(smashes, 0)
        dist = lambda p: float(np.nanmean(np.linalg.norm(p - mean, axis=-1)))  # noqa: E731
        spread = float(np.median([dist(p) for p in smashes]))
        if not spread > 0:
            continue
        for r in rs:
            if r["group"] == "吊球":
                r["metrics"]["disguise"] = round(dist(r["prep"]) / spread, 2)


# ---------------------------------------------------------------- flags


def percentile_of(value: float, ref: dict) -> float | None:
    """Rough percentile of value in a reference {p10, p25, p50, p75, p90} (linear between, clipped 0..100)."""
    pts = [(p, ref.get(f"p{p}")) for p in (10, 25, 50, 75, 90)]
    pts = [(p, v) for p, v in pts if v is not None]
    if len(pts) < 2 or value is None:
        return None
    ps, vs = zip(*pts)
    if vs[-1] == vs[0]:
        return 50.0
    return float(np.clip(np.interp(value, vs, ps, left=0, right=100), 0, 100))


def flags_for(group: str, m: dict, reference: dict | None) -> tuple[list[str], dict]:
    """Plain-language notes and per-metric percentiles."""
    notes, pct = [], {}
    if group in OVERHEAD:
        if m.get("chain_order_ok") is False:
            c = m.get("chain_ms", {})
            notes.append("发力顺序：" + "、".join(f"{k} {v:+d}ms" for k, v in c.items()) + "，不是从髋到手腕依次加速")
        if reference is None:
            if m.get("contact_height") is not None and m["contact_height"] < RULES["contact_low"]:
                notes.append(f"击球点偏低：手腕只高出肩 {m['contact_height']:.2f} 个躯干长")
            if m.get("turn") is not None and m["turn"] < RULES["turn_small"]:
                notes.append("转体很少：肩线几乎没转过来")
    ref = (reference or {}).get("groups", {}).get(group, {})
    for k, v in m.items():
        if k in ref and isinstance(v, (int, float)) and not isinstance(v, bool):
            p = percentile_of(v, ref[k])
            if p is None:
                continue
            pct[k] = round(p)
            if p <= 10 or p >= 90:
                where = "偏低" if p <= 10 else "偏高"
                notes.append(f"{METRIC_NAMES.get(k, k)} {v} {where}（参考的第 {round(p)} 百分位，中位数 {ref[k]['p50']}）")
    return notes, pct


# ---------------------------------------------------------------- whole video


def frame_rows(frames: np.ndarray) -> np.ndarray:
    """Video frame -> row in the skeleton arrays (-1 if that frame was not processed)."""
    frames = np.asarray(frames, int)
    row_of = np.full(int(frames[-1]) + 1, -1)
    row_of[frames] = np.arange(len(frames))
    return row_of


def analyse(data: dict, tracking, strokes: list[dict], fps: float, sides=("near",), include_review: bool = False,
            reference: dict | None = None) -> list[dict]:
    """One result per stroke of the chosen groups and sides."""
    row_of = frame_rows(data["frame"])
    by_id = {t.id: t for t in tracking.tracks}
    out = []
    picks = [s for s in strokes if s.get("group") in GROUPS and s.get("side") in sides
             and (include_review or not s.get("needs_review")) and s.get("player_id") in by_id]
    # racket hand per player: majority over their strokes
    hands: dict = {}
    windows = []
    for s in picks:
        tr = by_id[s["player_id"]]
        kp, sc, t = stroke_window(data, row_of, tr.det_index, int(s["frame"]), fps)
        windows.append((kp, sc, t))
        smooth = fill_and_smooth(kp, sc >= KPT_THR, fps)
        if np.isfinite(smooth[:, [L_WR, R_WR]]).all() and len(t) >= 5:
            hands.setdefault(s["player_id"], []).append(racket_side(smooth, t))
    hand_of = {p: max(set(v), key=v.count) for p, v in hands.items()}
    for s, (kp, sc, t) in zip(picks, windows):
        r = stroke_metrics(kp, sc, t, fps, s["group"], hand_of.get(s["player_id"]))
        tr = by_id[s["player_id"]]
        row = row_of[min(int(s["frame"]), len(row_of) - 1)]
        airborne = bool(tr.airborne[row]) if row >= 0 and getattr(tr, "airborne", None) is not None else None
        out.append({"frame": int(s["frame"]), "time_s": s.get("time_s", round(s["frame"] / fps, 3)),
                    "player_id": s["player_id"], "player_name": s.get("player_name", ""), "side": s["side"],
                    "type": s["type"], "group": s["group"], "type_prob": s.get("prob"),
                    "type_needs_review": bool(s.get("needs_review")), "airborne_at_hit": airborne, **r})
    add_disguise(out)
    for r in out:
        if r["ok"]:
            r["notes"], r["percentiles"] = flags_for(r["group"], r["metrics"], reference)
    return out


def summary(results: list[dict]) -> dict:
    """Per group: per player mean / std / n of every number, and how steady a player is
    (within-player spread divided by the spread over everyone; small = repeatable)."""
    out: dict = {}
    for g in GROUPS:
        rs = [r for r in results if r["ok"] and r["group"] == g]
        if not rs:
            continue
        gd: dict = {"n": len(rs), "players": {}, "steadiness": {}}
        keys = [k for k in NUMERIC if any(isinstance(r["metrics"].get(k), (int, float)) for r in rs)]
        for pid in sorted({r["player_id"] for r in rs}, key=str):
            mine = [r for r in rs if r["player_id"] == pid]
            pd = {"n": len(mine), "name": mine[0].get("player_name", "")}
            for k in keys:
                v = np.array([r["metrics"][k] for r in mine if isinstance(r["metrics"].get(k), (int, float))], float)
                if len(v):
                    pd[k] = {"mean": round(float(v.mean()), 3), "std": round(float(v.std()), 3), "n": len(v)}
            if g in OVERHEAD:
                pd["chain_order_ok_share"] = round(float(np.mean([bool(r["metrics"].get("chain_order_ok")) for r in mine])), 2)
            gd["players"][str(pid)] = pd
        for k in keys:
            allv = np.array([r["metrics"][k] for r in rs if isinstance(r["metrics"].get(k), (int, float))], float)
            within = [p[k]["std"] for p in gd["players"].values() if k in p and p[k]["n"] >= 3]
            if len(allv) >= 6 and within and allv.std() > 0:
                gd["steadiness"][k] = round(float(np.mean(within) / allv.std()), 2)
        out[g] = gd
    return out


def build_reference(docs: list[dict], min_n: int = 10) -> dict:
    """Percentiles of every number per group over the strokes of several technique.json files."""
    groups: dict = {}
    for doc in docs:
        for r in doc["strokes"]:
            if not r.get("ok"):
                continue
            for k, v in r["metrics"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    groups.setdefault(r["group"], {}).setdefault(k, []).append(v)
    out = {}
    for g, ms in groups.items():
        out[g] = {k: {"n": len(v), **{f"p{p}": round(float(np.percentile(v, p)), 3) for p in (10, 25, 50, 75, 90)}}
                  for k, v in ms.items() if len(v) >= min_n}
    return {"schema": REF_SCHEMA, "sources": [d.get("video", "") for d in docs], "groups": out}


# ---------------------------------------------------------------- files


def technique_json(results: list[dict], fps: float, video: str = "", inputs: dict | None = None,
                   reference: str | None = None) -> dict:
    strokes = [{k: v for k, v in r.items() if k != "prep"} for r in results]
    return {"schema": SCHEMA, "video": video, "fps": fps, "inputs": inputs or {}, "reference": reference,
            "rules": RULES, "window_s": [-BEFORE_S, AFTER_S], "metric_names": METRIC_NAMES,
            "summary": summary(results), "strokes": strokes}


def write_csv(results: list[dict], path: str | Path) -> None:
    cols = ["frame", "time_s", "player_id", "player_name", "side", "type", "group", "ok", "reason",
            "racket_hand", *NUMERIC, "chain_order_ok", "chain_hip_ms", "chain_shoulder_ms", "chain_elbow_ms",
            "chain_wrist_ms", "airborne_at_hit", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            m = r.get("metrics", {})
            row = {k: r.get(k, "") for k in cols}
            row.update({k: m.get(k, "") for k in NUMERIC})
            row["chain_order_ok"] = m.get("chain_order_ok", "")
            for k, v in (m.get("chain_ms") or {}).items():
                row[f"chain_{k}_ms"] = v
            row["notes"] = "；".join(r.get("notes", []))
            w.writerow(row)


def run(skeleton_npz, strokes_json_path, court, players_per_side: int, sides=("near",),
        include_review: bool = False, reference: dict | None = None) -> tuple[list[dict], float]:
    """players_per_side must be what `badminton strokes` used, so the player ids match."""
    from .tracking import load_npz, track_players

    doc = json.loads(Path(strokes_json_path).read_text())
    data = load_npz(skeleton_npz)
    fps = float(data["fps"])
    tracking = track_players(data, court, players_per_side)
    return analyse(data, tracking, doc["strokes"], fps, sides, include_review, reference), fps
