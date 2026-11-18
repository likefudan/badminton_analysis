"""FineBadminton-20K labels of one match -> strokes.json for `badminton technique`.

FineBadminton (Apache-2.0, https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)
labels every shot of 70 professional singles matches: `start_frame` (about 3 frames
before the shuttle turns, measured on match 0001), `hitter` (top / bottom of the
picture), `hit type` and a list of `subtype`s. This turns them into the strokes.json
that `badminton strokes` writes, so the pro strokes can be measured with
`badminton technique` and pooled with `badminton technique-ref` into a reference.

The skeleton is usually run on a clip of the rallies only (the frames of the full
match video listed in `<clip>.framemap.json` as `frame_map`); the labels are mapped
onto it. Without a frame map the skeleton is taken to be on the full video.

    python scripts/finebadminton_strokes.py 0013_updated.json \
        --skeleton 0013.rallies.skeleton.npz --court 0013.rallies.court.json \
        --framemap 0013.rallies.framemap.json -o 0013.rallies.strokes.json

Then:  badminton technique 0013.rallies.mp4 --court ... --data ... --singles \
           --strokes 0013.rallies.strokes.json --sides both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABEL_TO_CONTACT = 3  # FineBadminton's start_frame is about this many frames before the contact

# (hit type, subtype) -> (our name, group); subtype "" = no or unknown subtype.
# Groups are the ones `badminton strokes` uses. FineBadminton's "ball area" is where the
# shot was hit from: "push shot" is mostly from the front court (3549 of 4536), so 推扑.
TYPES = {
    ("serve", "short serve"): ("发短球", "发球"),
    ("serve", "high serve"): ("发长球", "发球"),
    ("serve", "flick serve"): ("发长球", "发球"),
    ("serve", ""): ("发球", "发球"),
    ("kill", ""): ("杀球", "杀球"),
    ("kill", "common smash"): ("杀球", "杀球"),
    ("kill", "jump smash"): ("跳杀", "杀球"),
    ("kill", "full smash"): ("重杀", "杀球"),
    ("kill", "slice smash"): ("劈杀", "杀球"),
    ("kill", "stick smash"): ("点杀", "杀球"),
    ("clear", ""): ("高远球", "高远/平高"),
    ("clear", "attacking clear"): ("平高球", "高远/平高"),
    ("drop shot", ""): ("吊球", "吊球"),
    ("drop shot", "slice drop shot"): ("劈吊", "吊球"),
    ("drop shot", "reverse slice drop shot"): ("反手位劈吊", "吊球"),
    ("drop shot", "blocked drop shot"): ("拦吊", "吊球"),
    ("drop shot", "stop drop shot"): ("收吊", "吊球"),
    ("net shot", ""): ("放网", "放网"),
    ("net shot", "spinning net"): ("搓球", "放网"),
    ("cross-court net shot", ""): ("勾对角", "勾对角"),
    ("net lift", ""): ("挑球", "挑球"),
    ("net lift", "high lift"): ("挑球", "挑球"),
    ("push shot", ""): ("推球", "推扑"),
    ("push shot", "flat lift"): ("平推", "推扑"),
    ("net kill", ""): ("扑球", "推扑"),
    ("net kill", "continuous net kills"): ("扑球", "推扑"),
    ("block", ""): ("挡网", "抽挡"),
    ("block", "high block"): ("挡网", "抽挡"),
    ("drive", ""): ("平抽", "抽挡"),
    ("drive", "flat drive"): ("平抽", "抽挡"),
    ("drive", "high drive"): ("平抽", "抽挡"),
}
SIDE = {"bottom": "near", "top": "far"}  # checked on frames of matches 0013 and 0021


def stroke_type(hit_type: str, subtypes: list) -> tuple[str, str] | None:
    """(name, group) for a labelled shot; the first subtype that is known wins. None if unknown."""
    hit_type = (hit_type or "").strip()
    for sub in list(subtypes or []) + [""]:
        t = TYPES.get((hit_type, (sub or "").strip()))
        if t:
            return t
    return None


def load_shots(label_json: str | Path) -> tuple[list[dict], float]:
    """Every labelled shot of a match: frame (full video), hitter side, type, player name."""
    doc = json.loads(Path(label_json).read_text())
    shots = []
    for r in doc:
        for h in r.get("hitting", []):
            fa = h.get("Foundational Actions Level") or {}
            if h.get("start_frame") is None or fa.get("hitter") not in SIDE:
                continue
            shots.append({"frame": int(h["start_frame"]) + LABEL_TO_CONTACT, "side": SIDE[fa["hitter"]],
                          "hit_type": fa.get("hit type") or "", "subtype": fa.get("subtype") or [],
                          "player_name": h.get("player") or ""})
    return shots, float(doc[0]["fps"]) if doc else 0.0


def nearest_track(tracks, side: str, row: int, max_rows: int) -> int | None:
    """Id of the track on that side seen nearest to skeleton row `row` (within max_rows)."""
    s = 0 if side == "near" else 1
    best, best_d = None, max_rows + 1
    for tr in tracks:
        if tr.side != s:
            continue
        seen = np.flatnonzero(np.asarray(tr.det_index) >= 0)
        if not len(seen):
            continue
        d = int(np.min(np.abs(seen - row)))
        if d < best_d:
            best, best_d = tr.id, d
    return best


def convert(shots: list[dict], data: dict, tracking, fps: float, frame_map: list[int] | None = None,
            max_gap_s: float = 0.5) -> tuple[list[dict], dict]:
    """Strokes in the layout of `badminton strokes` (frame on the skeleton's video), plus counts."""
    to_clip = {int(f): j for j, f in enumerate(frame_map) if f >= 0} if frame_map is not None else None
    frames = np.asarray(data["frame"])
    out, skipped = [], {"not_in_clip": 0, "unknown_type": 0, "no_player": 0}
    for s in shots:
        f = to_clip.get(s["frame"]) if to_clip is not None else s["frame"]
        if f is None:
            skipped["not_in_clip"] += 1
            continue
        t = stroke_type(s["hit_type"], s["subtype"])
        if t is None:
            skipped["unknown_type"] += 1
            continue
        row = int(np.argmin(np.abs(frames - f)))
        pid = nearest_track(tracking.tracks, s["side"], row, max(1, round(max_gap_s * fps)))
        if pid is None:
            skipped["no_player"] += 1
            continue
        out.append({"frame": int(f), "time_s": round(f / fps, 3), "player_id": pid, "side": s["side"],
                    "type": t[0], "group": t[1], "type_en": " / ".join([s["hit_type"], *s["subtype"]]),
                    "prob": 1.0, "needs_review": False, "model": "finebadminton", "player_name": s["player_name"]})
    out.sort(key=lambda r: r["frame"])
    return out, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", help="FineBadminton JSON of one match (NNNN_updated.json)")
    ap.add_argument("--skeleton", required=True, help="skeleton .npz (every frame) of the clip or video")
    ap.add_argument("--court", required=True, help="court.json of the same clip / video")
    ap.add_argument("--framemap", help="<clip>.framemap.json: clip frame -> full video frame")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)

    from badminton.court import CourtCalibration
    from badminton.strokes import SCHEMA
    from badminton.tracking import load_npz, track_players

    shots, _ = load_shots(a.labels)
    data = load_npz(a.skeleton)
    fps = float(data["fps"])
    tracking = track_players(data, CourtCalibration.load(a.court), 1)  # same call as `badminton technique --singles`
    fm = json.loads(Path(a.framemap).read_text())["frame_map"] if a.framemap else None
    strokes, skipped = convert(shots, data, tracking, fps, fm)
    doc = {"schema": SCHEMA, "video": "", "fps": fps, "source": "FineBadminton-20K",
           "inputs": {"labels": str(a.labels), "skeleton": str(a.skeleton), "framemap": a.framemap},
           "label_to_contact_frames": LABEL_TO_CONTACT, "skipped": skipped, "strokes": strokes}
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    by_group: dict = {}
    for s in strokes:
        by_group[s["group"]] = by_group.get(s["group"], 0) + 1
    print(f"{len(strokes)} 拍（{len(shots)} 拍有标注），没用上：{skipped}；", "、".join(f"{g} {n}" for g, n in by_group.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
