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
CHAIN_WIN = (-0.5, 0.05)  # where the speed peaks of the kinetic chain are looked for (s from the contact):
#   up to the contact only, since after it the body keeps turning and its peak is not part of the swing
SHUTTLE_CLEAR = 0.75  # the racket wrist is the one nearer the shuttle when it is at most this share of the other's distance
CONTACT_SEARCH_S = 0.25  # overhead: the contact is the racket wrist's highest point this close to the detected hit
ARM_SPEED_S = 0.15  # low strokes without the shuttle: the racket wrist moves more this close to the hit
SHUTTLE_NEAR_T = 1.2  # ... and the shuttle rule is only used when the nearer wrist is within this many torso lengths
WRIST_PEAK_WIN = (-0.15, 0.05)  # the wrist's own peak is looked for this close to the contact
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
# numbers compared with a reference and written into notes. Left out, after the pro run on FineBadminton
# (14 broadcast matches): turn (shoulder-line width depends on camera height), jump (moving forward / back
# shows as ankles rising), elbow_deg (p10 of pro smashes 42-78 deg: pose errors), and the swing order
# (only 31% of pro overhead strokes come out "in order" at 25-30 fps). Also left out after checking frames
# of the user's clips: wrist_speed, follow_through and backswing_elbow, since the broadcast camera films from
# high behind, where the forward swing is mostly depth and shows up small, while a phone at the side sees it
# across the picture. They stay in the JSON.
FLAG_METRICS = ("contact_height", "stance", "knee_deg", "trunk_lean", "disguise")
# (group, metric) pairs never noted: 推扑 mixes pushes (below the shoulder) with net kills (above it), so
# its contact height says more about which of the two it was than about the stroke being done well.
UNNOTED = {("推扑", "contact_height")}


def is_noted(group: str, metric: str) -> bool:
    return metric in FLAG_METRICS and (group, metric) not in UNNOTED


DEFAULT_REFERENCE = Path(__file__).with_name("technique_ref_pro_near.json")
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
    Only frames the skeleton was run on are returned (with --stride 2 every other frame), so
    the samples are 1 / effective_fps apart. Frames the player was not detected in come back
    NaN / score 0."""
    n_frames = len(row_of)
    f = np.arange(hit_frame - round(before_s * fps), hit_frame + round(after_s * fps) + 1)
    f = f[(f >= 0) & (f < n_frames)]
    f = f[row_of[f] >= 0]
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


def _track_pair(P: np.ndarray, S: np.ndarray, k0: int, first_is_racket: bool) -> tuple[np.ndarray, np.ndarray]:
    """Two points per frame (n, 2, 2) whose labels may swap from frame to frame -> the same two
    points as (racket, other), kept apart by continuity (constant-velocity prediction) going out
    both ways from frame k0, where the racket one is given."""
    n = len(P)
    out = np.full((n, 2, 2), np.nan)
    outS = np.zeros((n, 2), bool)
    order = [0, 1] if first_is_racket else [1, 0]
    out[k0], outS[k0] = P[k0, order], S[k0, order]
    for step in (1, -1):
        last = out[k0].copy()
        prev = np.full((2, 2), np.nan)
        k = k0 + step
        while 0 <= k < n:
            pred = np.where(np.isfinite(prev), 2 * last - prev, last)
            pts, ok = P[k], S[k] & np.isfinite(P[k]).all(-1)
            if ok.all():
                keep = np.nansum(np.linalg.norm(pts - pred, axis=-1))
                swap = np.nansum(np.linalg.norm(pts[::-1] - pred, axis=-1))
                assigned = pts if keep <= swap else pts[::-1]
                got = np.array([True, True])
            elif ok.any():
                p = pts[ok.argmax()]
                d = np.linalg.norm(pred - p, axis=-1)
                slot = int(np.nanargmin(d)) if np.isfinite(d).any() else 0
                assigned = np.full((2, 2), np.nan)
                assigned[slot] = p
                got = np.arange(2) == slot
            else:
                got = np.array([False, False])
                assigned = np.full((2, 2), np.nan)
            out[k], outS[k] = assigned, got
            for j in range(2):
                if got[j]:
                    prev[j], last[j] = last[j], assigned[j]
                else:
                    prev[j] = np.nan  # no velocity across a gap
            k += step
    return out, outS


def assign_arms(kp: np.ndarray, seen: np.ndarray, t: np.ndarray, group: str,
                shuttle: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """The pose model's left / right labels flip when a player turns side- or back-on, so the
    racket arm is not taken from the labels. At the frame nearest the hit where both wrists are
    seen, the racket wrist is the one clearly nearer the shuttle (pixel position at the hit, if
    known), else the higher one (overhead strokes) or the one moving faster around the hit
    (others: in a lunge the free arm reaches back, so "farther out" picks the wrong one); from there each joint pair is followed by continuity. Returns copies where
    R_SH / R_EL / R_WR are the racket arm and L_* the other arm, plus the label ("R" / "L") the
    pose model gave the racket wrist at that frame (for information only)."""
    kp, seen = kp.copy(), seen.copy()
    both = seen[:, [L_WR, R_WR]].all(1) & np.isfinite(kp[:, [L_WR, R_WR]]).all((1, 2))
    if not both.any():
        return kp, seen, "?"
    k0 = int(np.flatnonzero(both)[np.argmin(np.abs(t[both]))])
    wl, wr = kp[k0, L_WR], kp[k0, R_WR]
    r_is_racket = None
    if shuttle is not None and np.isfinite(shuttle).all():
        dl, dr = np.linalg.norm(wl - shuttle), np.linalg.norm(wr - shuttle)
        torso = np.linalg.norm(_mid(kp[k0], L_SH, R_SH) - _mid(kp[k0], L_HIP, R_HIP))
        near_enough = np.isfinite(torso) and min(dl, dr) <= SHUTTLE_NEAR_T * torso  # not a point across the net
        if near_enough and min(dl, dr) <= SHUTTLE_CLEAR * max(dl, dr):
            r_is_racket = bool(dr < dl)
    if r_is_racket is None and group in OVERHEAD:
        r_is_racket = bool(wr[1] <= wl[1])
    if r_is_racket is None:  # faster wrist around the hit; labels may flip, so speed of the nearer-labelled path
        near = np.flatnonzero(_between(t, -ARM_SPEED_S, ARM_SPEED_S) & both)
        if len(near) >= 2:
            seg = kp[near][:, [L_WR, R_WR]]
            moved = np.linalg.norm(np.diff(seg, axis=0), axis=-1).sum(0)  # per label, (2,)
            r_is_racket = bool(moved[1] >= moved[0])
        else:
            r_is_racket = bool(wr[1] <= wl[1])
    label = "R" if r_is_racket else "L"
    chain = None  # racket point of the previous joint in the chain, per frame
    for li, ri in ((L_WR, R_WR), (L_EL, R_EL), (L_SH, R_SH)):
        P = kp[:, [ri, li]]
        S = seen[:, [ri, li]]
        if chain is None:
            first = bool(r_is_racket)
            k = k0
        else:  # the joint of this pair nearer the racket arm's previous joint, where both are seen
            ok = S.all(1) & np.isfinite(P).all((1, 2)) & np.isfinite(chain).all(1)
            if not ok.any():
                continue
            k = int(np.flatnonzero(ok)[np.argmin(np.abs(t[ok]))])
            first = np.linalg.norm(P[k, 0] - chain[k]) <= np.linalg.norm(P[k, 1] - chain[k])
        pair, pairS = _track_pair(P, S, k, first)
        kp[:, ri], kp[:, li] = pair[:, 0], pair[:, 1]
        seen[:, ri], seen[:, li] = pairS[:, 0], pairS[:, 1]
        chain = pair[:, 0]
    return kp, seen, label


def _at(t: np.ndarray, s: float) -> int:
    return int(np.argmin(np.abs(t - s)))


def _between(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (t >= lo - 1e-9) & (t <= hi + 1e-9)


def stroke_metrics(kp_raw: np.ndarray, sc: np.ndarray, t: np.ndarray, fps: float, group: str,
                   wrist: int | None = None, shuttle: np.ndarray | None = None) -> dict:
    """The numbers of one stroke (see the module doc). `wrist`: force the racket wrist by its
    pose-model label (tests); by default the racket arm is found by assign_arms, since the labels flip.
    Returns {"ok": bool, "reason": str, "racket_hand": "R"/"L", "metrics": {...}, "prep": (k, 17, 2) or None};
    racket_hand is the pose model's label of the racket wrist near the hit, for information only."""
    seen = sc >= KPT_THR
    if wrist is None:
        kp_raw, seen, hand = assign_arms(kp_raw, seen, t, group, shuttle)
    else:
        hand = "R" if wrist == R_WR else "L"
        if wrist == L_WR:  # move the racket arm into the R_* slots
            sw = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR]
            kp_raw, seen = kp_raw.copy(), seen.copy()
            kp_raw[:, sw] = kp_raw[:, [R_SH, L_SH, R_EL, L_EL, R_WR, L_WR]]
            seen[:, sw] = seen[:, [R_SH, L_SH, R_EL, L_EL, R_WR, L_WR]]
    wrist, elbow, shoulder, other_sh = R_WR, R_EL, R_SH, L_SH
    core = [L_SH, R_SH, L_HIP, R_HIP]
    kp = fill_and_smooth(kp_raw, seen, fps)
    if len(t) < 5 or not np.isfinite(kp[:, core]).all():
        return {"ok": False, "reason": "击球前后没拍到这个人", "metrics": {}}
    need = core + [elbow, wrist]
    seen_share = float(seen[:, need].all(1).mean())
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
    if group in OVERHEAD:  # the detected hit can be a few frames off: take the racket wrist's top
        near = np.flatnonzero(_between(t, -CONTACT_SEARCH_S, CONTACT_SEARCH_S))
        top = int(near[np.argmin(kp[near, wrist, 1])])
        if top in (near[0], near[-1]):  # still rising / falling at the edge: no clear top, keep the detected hit
            m["contact_shift_ms"], m["contact_unclear"] = 0, True
        else:
            h = top
            m["contact_shift_ms"], m["contact_unclear"] = round(float(t[h]) * 1000), False
            t = t - t[h]
        if up * (kp[h, wrist, 1] - msh[h, 1]) < 0 and not m["contact_unclear"]:
            return {"ok": False, "reason": "最高点时持拍手还在肩下面：多半不是高手位球，类型可能判错", "racket_hand": hand,
                    "metrics": {}, "seen_share": round(seen_share, 2)}
    unclear = bool(m.get("contact_unclear"))  # then numbers taken at the contact are left out
    m["contact_height"] = None if unclear else up * (kp[h, wrist, 1] - msh[h, 1]) / torso
    speed_w = _speed(kp[:, wrist], fps) / torso
    chain = _between(t, *CHAIN_WIN)
    m["wrist_speed"] = float(np.max(speed_w[chain]))

    if group in OVERHEAD:
        m["elbow_deg"] = None if unclear else _angle(kp[h, shoulder], kp[h, elbow], kp[h, wrist])
        prep = _between(t, *PREP_WIN)
        m["backswing_elbow"] = float(np.max(up * (kp[prep, elbow, 1] - msh[prep, 1]))) / torso if prep.any() else None
        parts = {"hip": mhip, "shoulder": kp[:, shoulder], "elbow": kp[:, elbow], "wrist": kp[:, wrist]}
        idx = np.flatnonzero(chain)
        widx = np.flatnonzero(_between(t, *WRIST_PEAK_WIN)) if not unclear else np.array([], int)
        peaks = {k: float(t[idx[np.argmax(_speed(v, fps)[idx])]]) for k, v in parts.items() if k != "wrist"}
        if len(widx):
            sw = _speed(kp[:, wrist], fps)
            peaks["wrist"] = float(t[widx[np.argmax(sw[widx])]])
            # the fastest wrist moment of the whole swing came well before the contact (dropping the racket
            # in the backswing, or jitter): the order cannot be read from this stroke
            m["wrist_peak_early"] = bool(np.max(sw[idx]) > 1.2 * np.max(sw[widx]))
        m["chain_ms"] = {k: round(v * 1000) for k, v in peaks.items()}
        if len(peaks) == 4 and not m.get("wrist_peak_early"):
            order = [peaks[k] for k in ("hip", "shoulder", "elbow", "wrist")]
            tol = RULES["chain_tol_ms"] / 1000 + 0.5 / fps
            # outwards in time, and the wrist clearly after the hip (all four at once is not a chain)
            m["chain_order_ok"] = bool(all(b >= a - tol for a, b in zip(order, order[1:]))
                                       and order[3] - order[0] >= 1.5 / fps)
        else:
            m["chain_order_ok"] = None
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
        if t[-1] >= FOLLOW_S - 1 / fps and after.any() and not unclear:
            # how far the wrist keeps going, sideways in the picture, the way it was moving into the contact
            across = np.sign(kp[h, wrist, 0] - kp[max(0, h - 2), wrist, 0]) or 1.0
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
        if reference is None:
            if m.get("contact_height") is not None and m["contact_height"] < RULES["contact_low"]:
                notes.append(f"击球点偏低：手腕只高出肩 {m['contact_height']:.2f} 个躯干长")
            if m.get("turn") is not None and m["turn"] < RULES["turn_small"]:
                notes.append("转体很少：肩线几乎没转过来")
    ref = (reference or {}).get("groups", {}).get(group, {})
    for k, v in m.items():
        if k in ref and isinstance(v, (int, float)) and not isinstance(v, bool) and (group, k) not in UNNOTED:
            p = percentile_of(v, ref[k])
            if p is None:
                continue
            pct[k] = round(p)
            if is_noted(group, k) and (p <= 10 or p >= 90):
                where = "偏低" if p <= 10 else "偏高"
                notes.append(f"{METRIC_NAMES.get(k, k)} {v} {where}（参考的第 {round(p)} 百分位，中位数 {ref[k]['p50']}）")
    return notes, pct


# ---------------------------------------------------------------- whole video


def load_reference(path: str | Path | None = "auto") -> tuple[dict | None, str | None]:
    """"auto": the pro reference shipped with the package (near-court players of 14 FineBadminton
    matches), if present; "none" / None: no reference; else a technique-ref JSON file."""
    if path in (None, "none"):
        return None, None
    p = DEFAULT_REFERENCE if path == "auto" else Path(path)
    if not p.exists():
        if path == "auto":
            return None, None
        raise FileNotFoundError(p)
    return json.loads(p.read_text()), str(p)


def frame_rows(frames: np.ndarray) -> np.ndarray:
    """Video frame -> row in the skeleton arrays (-1 if that frame was not processed)."""
    frames = np.asarray(frames, int)
    row_of = np.full(int(frames[-1]) + 1, -1)
    row_of[frames] = np.arange(len(frames))
    return row_of


def shuttle_at(shuttle_px: np.ndarray | None, frame: int, fps: float, half_s: float = 0.05) -> np.ndarray | None:
    """Median shuttle pixel position within half_s of a frame, None if not seen there."""
    if shuttle_px is None:
        return None
    k = max(1, round(half_s * fps))
    seg = shuttle_px[max(0, frame - k):frame + k + 1]
    seg = seg[np.isfinite(seg).all(1)]
    return np.median(seg, 0) if len(seg) else None


def analyse(data: dict, tracking, strokes: list[dict], fps: float, sides=("near",), include_review: bool = False,
            reference: dict | None = None, shuttle_px: np.ndarray | None = None) -> list[dict]:
    """One result per stroke of the chosen groups and sides."""
    row_of = frame_rows(data["frame"])
    # skeleton run with --stride k: windows hold every k-th frame, so speeds and smoothing use fps / k
    step = float(np.median(np.diff(data["frame"]))) if len(data["frame"]) > 1 else 1.0
    fps_eff = fps / max(step, 1.0)
    by_id = {t.id: t for t in tracking.tracks}
    out = []
    picks = [s for s in strokes if s.get("group") in GROUPS and s.get("side") in sides
             and (include_review or not s.get("needs_review")) and s.get("player_id") in by_id]
    windows = []
    for s in picks:
        tr = by_id[s["player_id"]]
        kp, sc, t = stroke_window(data, row_of, tr.det_index, int(s["frame"]), fps)
        windows.append((kp, sc, t))
    for s, (kp, sc, t) in zip(picks, windows):
        r = stroke_metrics(kp, sc, t, fps_eff, s["group"], shuttle=shuttle_at(shuttle_px, int(s["frame"]), fps))
        tr = by_id[s["player_id"]]
        row = int(np.argmin(np.abs(np.asarray(data["frame"]) - int(s["frame"]))))  # nearest processed frame
        airborne = bool(tr.airborne[row]) if row >= 0 and getattr(tr, "airborne", None) is not None else None
        out.append({"frame": int(s["frame"]), "time_s": s.get("time_s", round(s["frame"] / fps, 3)),
                    "player_id": s["player_id"], "player_name": s.get("player_name", ""), "side": s["side"],
                    "type": s["type"], "group": s["group"], "type_prob": s.get("prob"),
                    "type_needs_review": bool(s.get("needs_review")), "airborne_at_hit": airborne,
                    "box": player_box(kp, sc), **r})
    add_disguise(out)
    for r in out:
        if r["ok"]:
            r["notes"], r["percentiles"] = flags_for(r["group"], r["metrics"], reference)
    return out


def player_box(kp: np.ndarray, sc: np.ndarray) -> list[int] | None:
    """[x0, y0, x1, y1] in video pixels around every joint seen in the stroke window: where the
    player is, for cutting the pictures of the stroke out of the video (stroke_shots)."""
    seen = (sc >= KPT_THR) & np.isfinite(kp).all(-1)
    if seen.sum() < 4:
        return None
    pts = kp[seen]
    return [int(np.floor(pts[:, 0].min())), int(np.floor(pts[:, 1].min())),
            int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max()))]


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
                known = [r["metrics"]["chain_order_ok"] for r in mine if r["metrics"].get("chain_order_ok") is not None]
                if known:
                    pd["chain_order_ok_share"] = round(float(np.mean(known)), 2)
                    pd["chain_order_known"] = len(known)
            gd["players"][str(pid)] = pd
        for k in keys:
            allv = np.array([r["metrics"][k] for r in rs if isinstance(r["metrics"].get(k), (int, float))], float)
            within = [p[k]["std"] for p in gd["players"].values() if k in p and p[k]["n"] >= 3]
            # needs two players: with one, within-player and overall spread are the same strokes (always 1.0)
            if len(allv) >= 6 and len(within) >= 2 and allv.std() > 0:
                gd["steadiness"][k] = round(float(np.mean(within) / allv.std()), 2)
        out[g] = gd
    return out


def build_reference(docs: list[dict], min_n: int = 10) -> dict:
    """Percentiles of every technique number (METRIC_NAMES) per group over the strokes of several
    technique.json files. Bookkeeping numbers such as contact_shift_ms are left out: they say how
    far the detected hit was off, not how the stroke was played."""
    groups: dict = {}
    for doc in docs:
        for r in doc["strokes"]:
            if not r.get("ok"):
                continue
            for k, v in r["metrics"].items():
                if k in METRIC_NAMES and isinstance(v, (int, float)) and not isinstance(v, bool):
                    groups.setdefault(r["group"], {}).setdefault(k, []).append(v)
    out = {}
    for g, ms in groups.items():
        out[g] = {k: {"n": len(v), **{f"p{p}": round(float(np.percentile(v, p)), 3) for p in (10, 25, 50, 75, 90)}}
                  for k, v in ms.items() if len(v) >= min_n}
    return {"schema": REF_SCHEMA, "sources": [d.get("video", "") for d in docs], "groups": out}


# ---------------------------------------------------------------- files


def reference_summary(reference: dict | None) -> dict:
    """p10 / p50 / p90 of the noted numbers only, per group: what the web page and the review show."""
    out: dict = {}
    for g, ms in ((reference or {}).get("groups") or {}).items():
        keep = {k: {q: v[q] for q in ("p10", "p50", "p90", "n") if q in v} for k, v in ms.items() if is_noted(g, k)}
        if keep:
            out[g] = keep
    return out


def technique_json(results: list[dict], fps: float, video: str = "", inputs: dict | None = None,
                   reference: str | None = None, reference_doc: dict | None = None) -> dict:
    strokes = [{k: v for k, v in r.items() if k != "prep"} for r in results]
    return {"schema": SCHEMA, "video": video, "fps": fps, "inputs": inputs or {}, "reference": reference,
            "rules": RULES, "window_s": [-BEFORE_S, AFTER_S], "metric_names": METRIC_NAMES,
            "noted_metrics": list(FLAG_METRICS), "reference_summary": reference_summary(reference_doc),
            "summary": summary(results), "strokes": strokes}


def _median(vals: list) -> float | None:
    v = [x for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return round(float(np.median(v)), 2) if v else None


def digest(doc: dict, max_examples: int = 4) -> dict:
    """Short version of a technique.json for the AI review: per player and stroke group the number of
    strokes, the median of each noted number next to the pro median, how many strokes fell outside the
    pros' 10..90% range on it, and a few example notes with their time. Only the noted numbers."""
    ref = doc.get("reference_summary") or {}
    players: dict = {}
    for r in doc.get("strokes", []):
        if not r.get("ok"):
            continue
        pid = str(r["player_id"])
        p = players.setdefault(pid, {"id": r["player_id"], "name": r.get("player_name") or "", "side": r.get("side"),
                                     "groups": {}, "examples": []})
        g = p["groups"].setdefault(r["group"], {"n": 0, "values": {}, "outside": {}})
        g["n"] += 1
        for k in FLAG_METRICS:
            if not is_noted(r["group"], k):
                continue
            v = r["metrics"].get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                g["values"].setdefault(k, []).append(v)
            pc = (r.get("percentiles") or {}).get(k)
            if pc is not None and (pc <= 10 or pc >= 90):
                g["outside"][k] = g["outside"].get(k, 0) + 1
        if r.get("notes") and len(p["examples"]) < max_examples:
            p["examples"].append({"time_s": r["time_s"], "type": r["type"], "notes": r["notes"]})
    for p in players.values():
        for gname, g in p["groups"].items():
            g["metrics"] = {k: {"median": _median(v), "pro_median": ref.get(gname, {}).get(k, {}).get("p50"),
                                "outside_pro_range": g["outside"].get(k, 0)} for k, v in g.pop("values").items()}
            g.pop("outside")
    skipped = sum(1 for r in doc.get("strokes", []) if not r.get("ok"))
    return {"players": list(players.values()), "skipped": skipped, "has_reference": bool(ref),
            "metric_names": {k: METRIC_NAMES[k] for k in FLAG_METRICS}}


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
        include_review: bool = False, reference: dict | None = None, shuttle_csv=None) -> tuple[list[dict], float]:
    """players_per_side must be what `badminton strokes` used, so the player ids match.
    shuttle_csv (optional): picks the racket arm as the one nearer the shuttle at the hit."""
    from .tracking import load_npz, track_players

    doc = json.loads(Path(strokes_json_path).read_text())
    data = load_npz(skeleton_npz)
    fps = float(data["fps"])
    tracking = track_players(data, court, players_per_side)
    shuttle_px = None
    if shuttle_csv and Path(shuttle_csv).exists():
        from .bst import read_shuttle_csv

        shuttle_px = read_shuttle_csv(shuttle_csv, int(data["frame"][-1]) + 1)
    return analyse(data, tracking, doc["strokes"], fps, sides, include_review, reference, shuttle_px), fps
