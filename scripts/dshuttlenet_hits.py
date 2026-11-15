"""Score `badminton hits` against DShuttleNet's shot-by-shot labels (professional doubles).

DShuttleNet (Kaggle anonymousay58dki/dshuttlenet, CC BY-NC-SA 4.0) has 58 BWF
doubles matches (men's, women's, mixed) with every shot of every rally: the
frame it starts at in the full broadcast video (30 fps), who hit it (A/B: the
winners, C/D: the losers), the stroke type and where the four players stood,
in court coordinates. S2DoublesDataset's three matches are DShuttleNet matches
5–7 with the same frames. See docs/hits.md, "DShuttleNet 双打对照".

The rally clip is the one scripts/bfmd_hits.py makes, so the steps are:

    python scripts/dshuttlenet_hits.py labels raw/ --match <video_name or match_id> -o out/
    python scripts/bfmd_hits.py clip videos/<video_name>.mp4 --rallies out/<video_name>.rallies.json -o out/<v>/<v>.rallies.mp4
    badminton calibrate / skeleton --stride 1 / shuttle --court / hits --court   (as for BFMD)
    python scripts/dshuttlenet_hits.py score out/*/*.rallies.mp4 [--shift N]

`score` reports, besides the moment (±3 and ±6 frames, as bfmd_hits), how
often the hitter is right for the hits found within the first tolerance: the
side (near / far half of the court) and the person. Our hitter is a track id;
it is turned into A/B/C/D by where that player stands at the hit (feet on the
court through our calibration) against where the labels put the four players
at the nearest labelled shot.

The downloaded files are untrusted data: this only reads CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bfmd_hits as B  # noqa: E402

SCHEMA = B.SCHEMA
LETTERS = "ABCD"
# DShuttleNet court coordinates: decimetres from the near left corner (x across, y towards the
# far end), the same axes as ours (badminton.court) in metres. Checked on video: the feet of our
# tracked players lie a median 0.4 m from the labelled positions this way (1.0 m with y flipped).
DM_LENGTH = 134.0
LANDED = {"Opp Grounded within Boundaries", "Failed Return", "Out of Bounds", "Judgment Error"}


def to_metres(x_dm: float, y_dm: float) -> tuple[float, float]:
    return x_dm / 10.0, y_dm / 10.0


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v: str) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def read_match(raw: str | Path, match: str) -> dict:
    """One match of the DShuttleNet CSVs: {"match": video_name, "rallies": [(s, e)], "shots": [...]}.

    `match` is the video_name or the match_id. Each shot: frame (start_frame_num),
    end_frame, player (A–D, None if not one of the four), side ("near"/"far" from where
    the hitter stands, None without a position), ball_type, set, rally, shot, and
    positions {A: (x, y) metres, …}."""
    raw = Path(raw)
    matches = _rows(raw / "match.csv")
    m = next((r for r in matches if match in (r["video_name"], r["match_id"])), None)
    if m is None:
        raise SystemExit(f"match.csv 里没有 {match}")
    sets = {r["set_id"]: r for r in _rows(raw / "set.csv") if r["match_id"] == m["match_id"]}
    rallies = {r["rally_id"]: r for r in _rows(raw / "rally.csv") if r["set_id"] in sets}
    shots = []
    for r in _rows(raw / "shot.csv"):
        if r["rally_id"] not in rallies:
            continue
        ids = {k: r[k] for k in LETTERS}
        pid = r["player"].split(".")[0] if r["player"] else ""
        who = next((k for k, v in ids.items() if v.split(".")[0] == pid and pid), None)
        pos = {}
        for k in LETTERS:
            x, y = _num(r[f"player_{k}_x"]), _num(r[f"player_{k}_y"])
            if x is not None and y is not None:
                pos[k] = to_metres(x, y)
        side = None
        if who in pos:
            side = "near" if pos[who][1] < DM_LENGTH / 20 else "far"
        ra = rallies[r["rally_id"]]
        shots.append({"frame": int(float(r["start_frame_num"])), "end_frame": int(float(r["end_frame_num"])),
                      "player": who, "side": side, "ball_type": r["ball_type"],
                      "set": int(float(sets[ra["set_id"]]["set_num"])), "rally": int(ra["rally_num"]),
                      "shot": int(r["shot_num"]), "positions": pos})
    shots.sort(key=lambda s: (s["frame"], s["shot"]))
    spans = sorted((int(float(r["start_frame"])), int(float(r["end_frame"]))) for r in rallies.values())
    # A rally's end frame is its last shot's end: where the shuttle came down. Only the reasons
    # that mean "it hit the floor" are a landing; a net fault ends on the net.
    landed = sorted(int(float(r["end_frame"])) for r in rallies.values()
                    if r.get("lose_reason", "").strip() in LANDED)
    return {"match": m["video_name"], "match_id": int(m["match_id"]), "event": int(m["event"]),
            "rallies": spans, "shots": shots, "landed": landed}


def cmd_labels(a) -> int:
    doc = read_match(a.raw, a.match)
    name = doc["match"]
    out = Path(a.out_dir)
    hits = [s["frame"] for s in doc["shots"]]
    inside = [h for h in hits if B.in_spans(h, doc["rallies"])]
    B.write_labels(hits, out / f"{name}.hits.labels.csv")
    side = {"schema": SCHEMA, "match": name, "source": f"DShuttleNet match_id {doc['match_id']}", "fps": 30.0,
            "event": {1: "MD", 2: "WD", 3: "XD"}.get(doc["event"], str(doc["event"])),
            "rallies": [list(r) for r in doc["rallies"]], "hits": hits, "hits_in_rallies": len(inside),
            "shots": doc["shots"], "events": {"bounce": doc["landed"]}}
    (out / f"{name}.rallies.json").write_text(json.dumps(side, ensure_ascii=False, indent=1))
    who = sum(s["player"] is not None for s in doc["shots"])
    print(f"{name}：{len(hits)} 拍（{who} 拍有击球人），{len(doc['rallies'])} 个回合，回合内 {len(inside)} 拍")
    return 0


# --- hitter ---------------------------------------------------------------------------------------


def track_feet(clip: Path) -> dict[int, np.ndarray]:
    """Feet of every tracked player on the court (metres), per clip frame, as `badminton hits` tracks them."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from badminton import hits as H
    from badminton.court import CourtCalibration
    from badminton.tracking import load_npz, track_players

    data = load_npz(clip.with_suffix(".skeleton.npz"))
    court = CourtCalibration.load(clip.with_suffix(".court.json"))
    tracking = track_players(data, court, 2)
    n = int(data["frame"][-1]) + 1
    out = {}
    for p in H.player_signals(data, tracking, n):
        xy = np.full((n, 2), np.nan)
        ok = np.isfinite(p.feet).all(1)
        if ok.any():
            xy[ok] = court.to_court(p.feet[ok])
        out[p.id] = xy
    return out


def nearest_letter(xy, positions: dict) -> tuple[str | None, float]:
    """Which labelled player (A–D) stands nearest to `xy` (metres), and how far (m)."""
    if xy is None or not np.isfinite(xy).all() or not positions:
        return None, float("inf")
    d = {k: float(np.hypot(xy[0] - p[0], xy[1] - p[1])) for k, p in positions.items()}
    k = min(d, key=d.get)
    return k, d[k]


def label_hitters(dets: list[dict], shots: list[dict], feet: dict[int, np.ndarray], max_dist: float = 2.0) -> None:
    """Fill each detection's `letter` (A–D or None) and keep `side`, in place.

    Positions are those of the labelled shot nearest in time; a player further than
    `max_dist` metres from every labelled player stays unassigned."""
    frames = np.array([s["frame"] for s in shots]) if shots else np.zeros(0, int)
    for d in dets:
        d["letter"], d["letter_dist"] = None, None
        if not len(frames) or d.get("track") not in feet:
            continue
        s = shots[int(np.abs(frames - d["frame"]).argmin())]
        f = feet[d["track"]]
        xy = f[d["clip_frame"]] if 0 <= d["clip_frame"] < len(f) else None
        k, dist = nearest_letter(xy, s["positions"])
        if k is not None and dist <= max_dist:
            d["letter"], d["letter_dist"] = k, round(dist, 2)


def hitter_scores(dets: list[dict], labels: list[dict], tol: int) -> dict:
    """Hitter accuracy over the hits matched within `tol` frames: side and person."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from badminton import hits as H

    out = {}
    for key in ("side", "letter"):
        r = H.score_hits([{"frame": d["frame"], "player_id": d.get(key)} for d in dets],
                         [{"frame": l["frame"], "player_id": l.get(key), "note": ""} for l in labels], tol)
        out[key] = {"checked": r["hitter_checked"], "right": r["hitter_right"]}
    return out


def detections(clip: Path) -> list[dict]:
    side = json.loads(clip.with_suffix(".framemap.json").read_text())
    fmap = side["frame_map"]
    out = []
    with open(clip.with_suffix(".hits.csv"), newline="") as f:
        for r in csv.DictReader(f):
            j = int(r["frame"])
            if 0 <= j < len(fmap) and fmap[j] >= 0:
                out.append({"frame": fmap[j], "clip_frame": j, "found_by": r.get("found_by", ""),
                            "track": int(r["player"]) if r.get("player", "").strip() else None,
                            "side": r.get("side") or None})
    return out


def score_match(clip: Path, tols=(3, 6), shift: int = 0) -> dict:
    res = B.score_match(clip, tols, shift)
    side = json.loads(clip.with_suffix(".framemap.json").read_text())
    doc = json.loads(Path(side["rallies_json"]).read_text())
    rallies = doc["rallies"]
    dets = [d for d in detections(clip) if B.in_spans(d["frame"], rallies, max(tols))]
    shots = [s for s in doc["shots"] if B.in_spans(s["frame"], rallies)]
    label_hitters(dets, shots, track_feet(clip))
    labels = [{"frame": s["frame"] + shift, "side": s["side"], "letter": s["player"]} for s in shots]
    res["event"] = doc.get("event", "")
    res["hitter"] = hitter_scores(dets, labels, tols[0])
    res["assigned"] = sum(d["letter"] is not None for d in dets)
    res["detections"] = [{k: d[k] for k in ("frame", "track", "side", "letter", "letter_dist", "found_by")} for d in dets]
    return res


def cmd_score(a) -> int:
    tols = tuple(int(t) for t in a.tol.split(","))
    rows = [score_match(Path(c), tols, a.shift) for c in a.clips]

    def prf(c):
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    def frac(h):
        return f"{B._pct(h['right'] / h['checked']) if h['checked'] else '—'}（{h['right']}/{h['checked']}）"

    print("| 比赛 | 项目 | 回合 | 对照拍数 | 识别 | " + " | ".join(f"±{t} 帧 F1" for t in tols)
          + f" | 击球人上下半场（±{tols[0]} 帧对上的） | 具体哪一个人 |")
    print("|" + " --- |" * (7 + len(tols)))
    tot = {t: {k: 0 for k in ("tp", "fp", "fn")} for t in tols}
    hit_tot = {k: {"checked": 0, "right": 0} for k in ("side", "letter")}
    for r in rows:
        for t in tols:
            for k in tot[t]:
                tot[t][k] += r[t][k]
        for k in hit_tot:
            for j in ("checked", "right"):
                hit_tot[k][j] += r["hitter"][k][j]
        print(f"| {r['match'][:50]} | {r['event']} | {r['rallies']} | {r['labels']} | {r['detected']} | "
              + " | ".join(B._pct(prf(r[t])[2]) for t in tols)
              + f" | {frac(r['hitter']['side'])} | {frac(r['hitter']['letter'])} |")
    print(f"| 合计 | | {sum(r['rallies'] for r in rows)} | {sum(r['labels'] for r in rows)} | "
          f"{sum(r['detected'] for r in rows)} | " + " | ".join(B._pct(prf(tot[t])[2]) for t in tols)
          + f" | {frac(hit_tot['side'])} | {frac(hit_tot['letter'])} |")
    for t in tols:
        p, rc, f = prf(tot[t])
        print(f"±{t} 帧：准确率 {B._pct(p)}，召回率 {B._pct(rc)}，F1 {B._pct(f)}")
    off = [o for r in rows for o in r[tols[0]]["offset_frames"]]
    if off:
        o = np.array(off)
        print(f"±{tols[0]} 帧内对上的 {len(o)} 拍：识别减标注的中位数 {np.median(o):+.0f} 帧，平均 {o.mean():+.2f} 帧")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "total": tot, "hitter": hit_tot}, ensure_ascii=False, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("labels", help="DShuttleNet CSV -> 对照 CSV（整场帧号）和回合区间 JSON（bfmd_hits clip 可用）")
    s.add_argument("raw", help="解压后的 DShuttleNet 文件夹（match.csv、set.csv、rally.csv、shot.csv）")
    s.add_argument("--match", required=True, help="video_name 或 match_id")
    s.add_argument("-o", "--out-dir", required=True)
    s = sub.add_parser("score", help="击球时刻 F1 和击球人（上下半场、具体哪一个人）准确率")
    s.add_argument("clips", nargs="+", help="bfmd_hits clip 输出的视频（旁边要有 framemap、court、skeleton、hits）")
    s.add_argument("--tol", default="3,6")
    s.add_argument("--shift", type=int, default=0, help="标注帧号整体加几帧再比")
    s.add_argument("--json", help="另存逐场结果")
    a = p.parse_args(argv)
    return {"labels": cmd_labels, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
