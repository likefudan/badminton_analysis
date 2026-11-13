"""Stroke type of every hit, per player, in singles and doubles.

Takes the hits (`badminton hits`: frame and hitter) and classifies each one
with BST (see bst.py) using the released weights that keep ShuttleSet's 17
stroke types apart (`bst_CG_AP_JnB_bone.pt`, 35 classes: far x 17, near x 17,
unknown). Then:

* the two players fed to BST are the hitter and one opponent, taken from the
  player tracks (so people at the side of the court are never picked, and
  doubles works without retraining: BST only ever sees two players);
* only the hitter's 17 classes (and "unknown") are kept, since the hitter is
  already known;
* ShuttleSet's types are mapped to the names players use, grouped as
  杀球 / 吊球 / 放网 / 勾对角 / 推扑 / 挑球 / 高远·平高 / 抽挡 / 发球;
* two attributes come from where the next hitter stands, not from BST:
  直线 / 对角 (how far across the court the shuttle went) and, for clears,
  平高 / 高远 (how fast the shuttle got there).

Not covered (ShuttleSet does not label them, they need our own labels):
劈杀 / 劈吊, 收吊, 搓球（展搓）.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from . import bst
from .court import CourtCalibration
from .provenance import DERIVED

SCHEMA = "badminton.strokes/1"

# BST's unmerged class order (shuttleset_dataset.get_stroke_types, side="Both"):
# Top (far) x 17, Bottom (near) x 17, then 未知球種.
SHUTTLESET_TYPES = [
    "放小球", "擋小球", "殺球", "點扣", "挑球", "防守回挑",
    "長球", "平球", "後場抽平球", "切球", "過渡切球", "推球",
    "撲球", "防守回抽", "勾球", "發短球", "發長球",
]
N_TYPES = len(SHUTTLESET_TYPES)
N_CLASSES_FULL = 2 * N_TYPES + 1
UNKNOWN = 2 * N_TYPES

# ShuttleSet type -> (our name, group, English)
NAMES = {
    "放小球": ("放网", "放网", "net shot"),
    "擋小球": ("挡网", "抽挡", "net block"),
    "殺球": ("杀球", "杀球", "smash"),
    "點扣": ("点杀", "杀球", "wrist smash"),
    "挑球": ("挑球", "挑球", "lift"),
    "防守回挑": ("防守挑球", "挑球", "defensive lift"),
    "長球": ("高远球", "高远/平高", "clear"),
    "平球": ("平抽", "抽挡", "drive"),
    "後場抽平球": ("后场抽平", "抽挡", "rear-court drive"),
    "切球": ("吊球", "吊球", "drop"),
    "過渡切球": ("过渡吊球", "吊球", "passive drop"),
    "推球": ("推球", "推扑", "push"),
    "撲球": ("扑球", "推扑", "net kill"),
    "防守回抽": ("防守抽球", "抽挡", "defensive drive"),
    "勾球": ("勾对角", "勾对角", "cross-court net shot"),
    "發短球": ("发短球", "发球", "short serve"),
    "發長球": ("发长球", "发球", "long serve"),
}
UNKNOWN_NAME = ("未知", "未知", "unknown")
FLAT_CLEAR = ("平高球", "高远/平高", "flat clear")

DEFAULT_WEIGHTS = Path.home() / "codebase/third_party/BST_weights/weight/bst_CG_AP_JnB_bone.pt"

# rules (metres, seconds); rough, to be checked against labelled hits
CROSS_M = 2.0  # the next hitter is at least this far across from the hitter: 对角
STRAIGHT_M = 1.2  # at most this far across: 直线 (in between: no call)
FLAT_CLEAR_MPS = 8.0  # a clear whose shuttle reaches the next hitter at this ground speed or faster: 平高
NEXT_HIT_MAX_S = 2.5  # the next hit only tells where this one went if it comes within this time
OUTSIDE_KEEP = 0.15  # BST drops frames with a player outside the court; keep up to this share of the
#   court length / width outside (about 2 m behind the baseline) and clip the position to the edge:
#   amateurs often hit from behind the baseline, and dropping those frames left half the clip empty
GAP_FILL = 3  # bridge a player missing for at most this many samples (of 30 per second)
REVIEW_PROB = 0.5  # below this the type is marked for checking
REVIEW_OK = 0.6  # ... and when fewer than this share of the 30 samples had both players


def type_names(cls: int) -> tuple[str, str, str]:
    """Class index (0..34) -> (our name, group, English)."""
    if cls == UNKNOWN:
        return UNKNOWN_NAME
    return NAMES[SHUTTLESET_TYPES[cls % N_TYPES]]


def hitter_probs(probs: np.ndarray, sides: list[str | None]) -> np.ndarray:
    """(n, 35) -> (n, 18): the hitter's 17 types then unknown, renormalised.
    Hits with no known side keep the larger of far / near per type."""
    out = np.empty((len(probs), N_TYPES + 1))
    for i, (p, s) in enumerate(zip(probs, sides)):
        far, near = p[:N_TYPES], p[N_TYPES:2 * N_TYPES]
        mine = far if s == "far" else near if s == "near" else np.maximum(far, near)
        out[i, :N_TYPES], out[i, N_TYPES] = mine, p[UNKNOWN]
        out[i] /= out[i].sum()
    return out


# ---------------------------------------------------------------- two-player clips from the tracks


@dataclass
class Clip:
    pose: np.ndarray  # (30, 2, 72)
    pos: np.ndarray  # (30, 2, 2)
    shuttle: np.ndarray  # (30, 2)
    length: int
    ok_share: float  # share of the samples where both players were there (after bridging)


def sample_frames(hit_frame: int, fps: float, n_frames: int, seq_len: int = bst.SEQ_LEN) -> np.ndarray:
    """Half a second each side of the hit at 30 samples per second, video frame numbers,
    cut at the ends of the video. Same frames as BST's middle_in_a_sec + make_seq_len_same
    for 30 and 60 fps."""
    step = fps / 30.0
    f = np.round(hit_frame + (np.arange(seq_len) - seq_len // 2) * step).astype(int)
    return f[(f >= 0) & (f < n_frames)]


def _bridge(a: np.ndarray, have: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear fill of runs of missing samples no longer than max_gap that have data on both sides."""
    have = have.copy()
    idx = np.flatnonzero(have)
    for lo, hi in pairwise(idx):
        if 1 < hi - lo <= max_gap + 1:
            for k in range(lo + 1, hi):
                w = (k - lo) / (hi - lo)
                a[k] = (1 - w) * a[lo] + w * a[hi]
                have[k] = True
    return have


def build_clip(data: dict, rows: np.ndarray, dets: tuple[np.ndarray, np.ndarray], frames: np.ndarray,
               shuttle_px: np.ndarray, calib: CourtCalibration, seq_len: int = bst.SEQ_LEN,
               gap_fill: int | None = None, outside_keep: float | None = None) -> Clip:
    """BST arrays for one hit. rows: skeleton row of each sample frame (-1 if none);
    dets: per player (far first, then near), the detection index in each skeleton row (-1 if none)."""
    gap_fill = GAP_FILL if gap_fill is None else gap_fill
    outside_keep = OUTSIDE_KEEP if outside_keep is None else outside_keep
    w, h = (float(v) for v in data["frame_size"])
    n = len(frames)
    joints = np.zeros((n, 2, bst.N_JOINTS, 2))
    pos = np.zeros((n, 2, 2))
    have = np.zeros((n, 2), bool)
    for m, det in enumerate(dets):
        for t in range(n):
            r, d = rows[t], det[t]
            if r < 0 or d < 0:
                continue
            kp, bb = data["keypoints"][r, d], data["bbox"][r, d]
            joints[t, m] = bst.normalize_joints(kp, bb)
            pos[t, m] = bst.court_position(kp[None, 15:17], calib)[0]
            have[t, m] = True
        margin = max(outside_keep, bst.IN_COURT_EPS)
        inside = ((pos[:, m] > -margin) & (pos[:, m] < 1 + margin)).all(-1)
        have[:, m] &= inside
        pos[:, m] = np.clip(pos[:, m], 0.0, 1.0)
        flat = np.concatenate([joints[:, m].reshape(n, -1), pos[:, m]], axis=1)
        have[:, m] = _bridge(flat, have[:, m], gap_fill)
        joints[:, m] = flat[:, :-2].reshape(n, bst.N_JOINTS, 2)
        pos[:, m] = flat[:, -2:]
    ok = have.all(1)
    sh = shuttle_px[frames] if n else np.zeros((0, 2))
    shuttle = np.where(np.isfinite(sh), sh / (w, h), 0.0)
    joints[~ok], pos[~ok], shuttle[~ok] = 0, 0, 0
    j, p, s, length = bst.make_seq_len_same(seq_len, joints, pos, shuttle)
    jb = np.concatenate([j, bst.create_bones(j)], axis=-2).reshape(seq_len, 2, -1)
    return Clip(jb.astype(np.float32), p.astype(np.float32), s.astype(np.float32), length,
                float(ok.mean()) if n else 0.0)


# ---------------------------------------------------------------- per hit


def next_in_rally(hits: list[dict], i: int, fps: float) -> dict | None:
    """The following hit if it is the reply: from the other side, soon enough."""
    if i + 1 >= len(hits):
        return None
    a, b = hits[i], hits[i + 1]
    if b["frame"] - a["frame"] > NEXT_HIT_MAX_S * fps:
        return None
    if a.get("side") and b.get("side") and a["side"] == b["side"]:
        return None
    return b


def pick_opponent(hits: list[dict], i: int, fps: float, tracks: list, side_of: dict) -> int | None:
    """Track id of the opponent to show BST with the hitter: whoever hits the reply,
    else the other-side player seen most around the hit."""
    h = hits[i]
    nxt = next_in_rally(hits, i, fps)
    if nxt is not None and nxt.get("player_id") is not None and side_of.get(nxt["player_id"]) != h.get("side"):
        return nxt["player_id"]
    others = [t for t in tracks if ("near", "far")[t.side] != h.get("side")]
    if not others:
        return None
    return max(others, key=lambda t: int(t.seen[max(0, h["_row"] - 15):h["_row"] + 15].sum())).id


def _position(track, row: int) -> np.ndarray | None:
    xy = track.xy[row] if 0 <= row < len(track.xy) else np.array([np.nan, np.nan])
    return xy if np.isfinite(xy).all() else None


def attributes(hits: list[dict], i: int, fps: float, by_id: dict, type_name: str) -> dict:
    """直线 / 对角 and 平高 / 高远 from where the reply was hit."""
    out: dict = {}
    if type_name in ("发短球", "发长球"):
        return out
    h, nxt = hits[i], next_in_rally(hits, i, fps)
    if nxt is None or h.get("player_id") not in by_id or nxt.get("player_id") not in by_id:
        return out
    a = _position(by_id[h["player_id"]], h["_row"])
    b = _position(by_id[nxt["player_id"]], nxt["_row"])
    if a is None or b is None:
        return out
    across = float(abs(b[0] - a[0]))
    direction = "对角" if across >= CROSS_M else "直线" if across <= STRAIGHT_M else ""
    out["direction"] = {"value": direction or None, "across_m": round(across, 2), "source": DERIVED}
    if type_name == "高远球":
        t = (nxt["frame"] - h["frame"]) / fps
        speed = float(np.linalg.norm(b - a) / t) if t > 0 else float("nan")
        out["length"] = {"value": "平高" if speed >= FLAT_CLEAR_MPS else "高远", "ground_speed_mps": round(speed, 1),
                         "flight_s": round(t, 2), "source": DERIVED}
    return out


def classify_hits(data: dict, tracking, hits: list[dict], shuttle_px: np.ndarray, calib: CourtCalibration,
                  models: list, fps: float) -> list[dict]:
    """hits: dicts with frame, player_id, side ("near"/"far"). Returns one result dict per hit."""
    frames_np = np.asarray(data["frame"], int)
    n_frames = int(frames_np[-1]) + 1
    row_of = np.full(n_frames, -1)
    row_of[frames_np] = np.arange(len(frames_np))
    if len(frames_np) > 1 and np.median(np.diff(frames_np)) > 1:  # stride > 1: nearest processed frame
        near = np.clip(np.searchsorted(frames_np, np.arange(n_frames)), 1, len(frames_np) - 1)
        left = frames_np[near - 1]
        pick = np.where(np.arange(n_frames) - left <= frames_np[near] - np.arange(n_frames), near - 1, near)
        ok_gap = np.abs(frames_np[pick] - np.arange(n_frames)) <= 1
        row_of = np.where(ok_gap, pick, -1)
    by_id = {t.id: t for t in tracking.tracks}
    side_of = {t.id: ("near", "far")[t.side] for t in tracking.tracks}
    hits = [dict(h) for h in hits]
    for h in hits:
        h["_row"] = int(row_of[min(h["frame"], n_frames - 1)])
        if not h.get("side") and h.get("player_id") in side_of:
            h["side"] = side_of[h["player_id"]]
    shuttle_px = np.vstack([shuttle_px, np.full((max(0, n_frames - len(shuttle_px)), 2), np.nan)])

    clips, meta = [], []
    for i, h in enumerate(hits):
        opp = pick_opponent(hits, i, fps, tracking.tracks, side_of)
        frames = sample_frames(h["frame"], fps, n_frames)
        rows = row_of[frames]
        pair = {}
        for tid in (h.get("player_id"), opp):
            if tid in by_id:
                t = by_id[tid]
                pair[("near", "far")[t.side]] = np.where(rows >= 0, t.det_index[np.maximum(rows, 0)], -1)
        none = np.full(len(frames), -1)
        dets = (pair.get("far", none), pair.get("near", none))
        clips.append(build_clip(data, rows, dets, frames, shuttle_px, calib))
        meta.append(opp)

    if clips:
        import torch

        args = [torch.from_numpy(np.stack([getattr(c, k) for c in clips]))
                for k in ("pose", "shuttle", "pos")] + [torch.tensor([c.length for c in clips])]
        with torch.no_grad():
            probs = torch.stack([m(*args).softmax(-1) for m in models]).mean(0).numpy()
    else:
        probs = np.zeros((0, N_CLASSES_FULL))
    mine = hitter_probs(probs, [h.get("side") for h in hits])

    out = []
    for i, (h, c, opp) in enumerate(zip(hits, clips, meta)):
        order = np.argsort(-mine[i])
        k = int(order[0])
        cls = UNKNOWN if k == N_TYPES else k + (0 if h.get("side") == "far" else N_TYPES)
        name, group, en = type_names(cls)
        attrs = attributes(hits, i, fps, by_id, name)
        if attrs.get("length", {}).get("value") == "平高":
            name, group, en = FLAT_CLEAR
        alts = []
        for kk in order[1:3]:
            alts.append({"type": type_names(UNKNOWN if kk == N_TYPES else int(kk))[0], "prob": round(float(mine[i, kk]), 3)})
        prob = float(mine[i, k])
        out.append({
            "frame": h["frame"], "time_s": round(h["frame"] / fps, 3),
            "player_id": h.get("player_id"), "side": h.get("side"), "opponent_id": opp,
            "type": name, "group": group, "type_en": en, "shuttleset_type": None if k == N_TYPES else SHUTTLESET_TYPES[k],
            "prob": round(prob, 3), "alternatives": alts, "attributes": attrs,
            "ok_share": round(c.ok_share, 2),
            "needs_review": bool(prob < REVIEW_PROB or c.ok_share < REVIEW_OK or k == N_TYPES),
        })
    return out


# ---------------------------------------------------------------- files


def load_hits(path: str | Path) -> list[dict]:
    """hits.json from `badminton hits`, or a CSV with frame (and player / side) columns."""
    path = Path(path)
    if path.suffix == ".json":
        doc = json.loads(path.read_text())
        return [{"frame": int(e["frame"]["value"]), "player_id": e.get("player", {}).get("value"),
                 "side": e.get("player", {}).get("side"), "name": e.get("player", {}).get("name", "")}
                for e in doc["hits"]]
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            pid = r.get("player") or r.get("player_id")
            out.append({"frame": int(float(r["frame"])), "player_id": int(pid) if pid not in (None, "") else None,
                        "side": r.get("side") or None})
    return out


def strokes_json(results: list[dict], fps: float, video: str = "", inputs: dict | None = None,
                 names: dict | None = None, weights: list[str] | None = None) -> dict:
    names = names or {}
    for r in results:
        r["player_name"] = names.get(str(r["player_id"]), "")
    return {"schema": SCHEMA, "video": video, "fps": fps, "inputs": inputs or {}, "weights": weights or [],
            "rules": {"cross_m": CROSS_M, "straight_m": STRAIGHT_M, "flat_clear_mps": FLAT_CLEAR_MPS},
            "not_covered": ["劈杀", "劈吊", "收吊", "搓球"], "strokes": results}


def write_csv(results: list[dict], path: str | Path) -> None:
    cols = ["frame", "time_s", "player_id", "player_name", "side", "type", "group", "prob", "direction",
            "across_m", "clear_speed_mps", "alt1", "alt1_prob", "ok_share", "needs_review"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            d, ln = r["attributes"].get("direction", {}), r["attributes"].get("length", {})
            alt = r["alternatives"][0] if r["alternatives"] else {}
            w.writerow({"frame": r["frame"], "time_s": r["time_s"], "player_id": r["player_id"],
                        "player_name": r.get("player_name", ""), "side": r["side"], "type": r["type"],
                        "group": r["group"], "prob": r["prob"], "direction": d.get("value") or "",
                        "across_m": d.get("across_m", ""), "clear_speed_mps": ln.get("ground_speed_mps", ""),
                        "alt1": alt.get("type", ""), "alt1_prob": alt.get("prob", ""),
                        "ok_share": r["ok_share"], "needs_review": int(r["needs_review"])})


def load_models(weights: list[str | Path]) -> list:
    return [bst.load_model(w, n_class=N_CLASSES_FULL) for w in weights]


def run(shuttle_csv, skeleton_npz, court: CourtCalibration, weights: list, hits_file=None,
        players_per_side: int = 1) -> tuple[list[dict], dict]:
    """Files in, one result per hit out. Without hits_file the hits are found here (`badminton hits`)."""
    from . import hits as H
    from .tracking import load_npz, track_players

    data = load_npz(skeleton_npz)
    fps = float(data["fps"])
    n = int(data["frame"][-1]) + 1
    if hits_file:
        tracking = track_players(data, court, players_per_side)
        hits = load_hits(hits_file)
    else:
        found, ctx = H.run(shuttle_csv, skeleton_npz, court, players_per_side)
        tracking = ctx["tracking"]
        hits = [{"frame": h["frame"], "player_id": h.get("player_id"), "side": h.get("side")} for h in found]
    shuttle_px = bst.read_shuttle_csv(shuttle_csv, n)
    results = classify_hits(data, tracking, hits, shuttle_px, court, load_models(weights), fps)
    return results, {"fps": fps, "tracking": tracking, "data": data}


# ---------------------------------------------------------------- video


def label_text(r: dict) -> str:
    d = (r["attributes"].get("direction") or {}).get("value")
    text = r["type"] + (f"·{d}" if d and r["type"] != "勾对角" else "")
    return f"{text} {r['prob']:.0%}" + ("?" if r["needs_review"] else "")


def draw_video(src: str | Path, dst: str | Path, results: list[dict], data: dict, tracking,
               hold_s: float = 1.0) -> None:
    """Each stroke above the hitter's head for `hold_s` after the hit (a "?" when it needs checking)."""
    import cv2
    from PIL import Image, ImageDraw

    from .video import VideoWriter, probe, read_frames

    info = probe(src)
    font = cjk_font(max(16, round(info.height * 0.035)))
    fps = info.fps or float(data["fps"])
    hold = round(hold_s * fps)
    frames = np.asarray(data["frame"], int)
    by_id = {t.id: t for t in tracking.tracks}
    with VideoWriter(dst, info.width, info.height, fps, audio_from=src) as wr:
        for f, frame in read_frames(src):
            live = [r for r in results if 0 <= f - r["frame"] < hold]
            if live:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                d = ImageDraw.Draw(img, "RGBA")
                row = int(np.clip(np.searchsorted(frames, f), 0, len(frames) - 1))
                for r in live:
                    text = label_text(r)
                    t = by_id.get(r["player_id"])
                    det = int(t.det_index[row]) if t is not None else -1
                    if det >= 0:
                        x0, y0, x1, _ = data["bbox"][row, det]
                        xy = ((x0 + x1) / 2, max(0, y0 - font.size * 1.6))
                    else:
                        xy = (info.width / 2, info.height * 0.08)
                    box = d.textbbox(xy, text, font=font, anchor="ma")
                    pad = font.size * 0.3
                    d.rounded_rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], radius=pad,
                                        fill=(0, 0, 0, 170))
                    colour = (255, 235, 59, 255) if not r["needs_review"] else (200, 200, 200, 255)
                    d.text(xy, text, font=font, fill=colour, anchor="ma")
                frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            wr.write(frame)


def cjk_font(size: int):
    """A font with simplified Chinese glyphs. A .ttc holds several faces and the first one is not
    always the one asked for (PingFang.ttc starts with the Hong Kong face, which lacks 杀 and 远)."""
    from matplotlib import font_manager
    from PIL import ImageFont

    from .card import _CJK_FONTS

    known = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        for index in range(64 if name in known else 0):
            try:
                font = ImageFont.truetype(known[name], size, index=index)
            except OSError:
                break
            if font.getname()[0] == name:
                return font
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
