"""Score `badminton hits` against BFMD's hand-marked hit moments (professional doubles).

BFMD (see docs/hits.md) marks, for 7 BWF men's doubles
matches, every hit as one frame of the full broadcast video (1280x720, 30 fps),
plus Rally / replay / bounce / net_hit spans, as a Label Studio timeline
export. Doubles have no hitter and no stroke type, so scoring is on the moment
only (the Player column stays empty, and `badminton hits --labels` then skips
the hitter check).

A broadcast is mostly not play (replays, close-ups, breaks), so only the Rally
spans are processed: `clip` cuts them, with a little margin, into one video and
holds the last frame of each for a moment in between (frozen frames: no
motion, a static shuttle point that drop_static removes, and a pause longer
than hits.RALLY_GAP_S so the next rally starts afresh).

    python scripts/bfmd_hits.py labels BFMD/annotations/hits_doubles/<match>.json -o out/
    python scripts/bfmd_hits.py clip videos/<match>.mp4 --rallies out/<match>.rallies.json -o out/<match>.rallies.mp4
    badminton calibrate out/<match>.rallies.mp4 --auto
    badminton skeleton out/<match>.rallies.mp4 --court … --stride 1
    badminton shuttle out/<match>.rallies.mp4 --court …
    badminton hits out/<match>.rallies.mp4 --court … --labels out/<match>.rallies.labels.csv
    python scripts/bfmd_hits.py score out/<match>.rallies.mp4 [more clips…]

`score` maps the detections back to frames of the full video, keeps those
inside a Rally span, and reports precision / recall / F1 within 3 and 6
frames per match and in total (missed and extra frames go to a JSON file for
looking at them).

The downloaded annotation files are untrusted data: this only reads JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = "badminton.bfmd_rallies/1"


# --- annotations ------------------------------------------------------------------------------


def read_bfmd(path: str | Path) -> dict:
    """Events of one BFMD hits_doubles file: {"hit": [frames], "Rally": [(start, end)], …}.

    Every timeline label becomes a list of (start, end) frame spans, except hit,
    bounce and net_hit, which are single frames (start == end in the export)."""
    doc = json.loads(Path(path).read_text())
    if isinstance(doc, list):
        doc = doc[0]
    out: dict[str, list] = {}
    for r in doc["annotations"][0]["result"]:
        v = r.get("value", {})
        for lab in v.get("timelinelabels", []):
            for rg in v.get("ranges", []):
                out.setdefault(lab, []).append((int(rg["start"]), int(rg["end"])))
    for k in out:
        out[k].sort()
    for k in ("hit", "bounce", "net_hit"):
        if k in out:
            out[k] = [s for s, _ in out[k]]
    return out


def in_spans(frame: int, spans, pad: int = 0) -> bool:
    return any(a - pad <= frame <= b + pad for a, b in spans)


def write_labels(frames, path: str | Path, note: str = "") -> None:
    """`badminton hits --labels` format: Frame, Player (empty: hitter not marked), Note."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Player", "Note"])
        for fr in frames:
            w.writerow([fr, "", note])


def cmd_labels(a) -> int:
    ev = read_bfmd(a.json)
    name = a.name or Path(a.json).stem
    out = Path(a.out_dir)
    rallies = ev.get("Rally", [])
    hits = ev.get("hit", [])
    inside = [h for h in hits if in_spans(h, rallies)]
    write_labels(hits, out / f"{name}.hits.labels.csv")
    doc = {"schema": SCHEMA, "match": name, "source": str(a.json), "fps": 30.0,
           "rallies": [list(r) for r in rallies],
           "hits": hits, "hits_in_rallies": len(inside),
           "events": {k: v for k, v in ev.items() if k not in ("hit", "Rally")}}
    (out / f"{name}.rallies.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"{name}：{len(hits)} 拍，{len(rallies)} 个回合，回合内 {len(inside)} 拍；"
          f"其他标签 {', '.join(f'{k} {len(v)}' for k, v in doc['events'].items())}")
    return 0


# --- rally clip -------------------------------------------------------------------------------


def plan_clip(rallies, n_frames: int, margin: int, hold: int) -> list[int]:
    """Frame map of the clip: original frame per clip frame, -1 for a held (frozen) frame.

    Each rally gets `margin` frames before and after (rallies that then touch are
    joined), and `hold` frozen frames after it."""
    spans = []
    for s, e in sorted(rallies):
        s, e = max(0, s - margin), min(n_frames - 1, e + margin)
        if spans and s <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    fmap: list[int] = []
    for s, e in spans:
        fmap.extend(range(s, e + 1))
        fmap.extend([-1] * hold)
    return fmap


def cmd_clip(a) -> int:
    import cv2

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from badminton.video import VideoWriter, probe

    doc = json.loads(Path(a.rallies).read_text())
    info = probe(a.video)
    if abs(info.fps - doc["fps"]) > 0.01:
        print(f"注意：视频 {info.fps} fps，标注按 {doc['fps']} fps")
    fmap = plan_clip(doc["rallies"], info.frames, a.margin, a.hold)
    want = np.array([f for f in fmap if f >= 0])
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(a.video))
    writer = VideoWriter(out, info.width, info.height, info.fps, encoder=a.encoder, crf=a.crf)
    i, k, last = 0, 0, None  # i: next frame of the video, k: position in fmap
    while k < len(fmap):
        f = fmap[k]
        if f < 0:
            writer.write(last)
            k += 1
            continue
        while i < f:  # skip without decoding into an image
            if not cap.grab():
                raise SystemExit(f"视频在第 {i} 帧结束，回合要到第 {f} 帧")
            i += 1
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"读不到第 {f} 帧")
        i += 1
        writer.write(frame)
        last = frame
        k += 1
    writer.close()
    cap.release()
    to_clip = {f: j for j, f in enumerate(fmap) if f >= 0}
    labels = [to_clip[h] for h in doc["hits"] if h in to_clip and in_spans(h, doc["rallies"])]
    write_labels(labels, out.with_suffix(".labels.csv"))
    side = {"schema": SCHEMA, "video": str(a.video), "rallies_json": str(a.rallies), "margin": a.margin,
            "hold": a.hold, "frame_map": fmap}
    out.with_suffix(".framemap.json").write_text(json.dumps(side))
    print(f"已写入 {out}（{len(fmap)} 帧，其中原片 {len(want)} 帧），{out.with_suffix('.labels.csv').name}"
          f"（回合内 {len(labels)} 拍），{out.with_suffix('.framemap.json').name}")
    return 0


# --- scoring ----------------------------------------------------------------------------------


def detections_in_video_frames(hits_csv: str | Path, fmap: list[int]) -> list[dict]:
    """`badminton hits` CSV of the clip -> hits on the full video's frame axis (held frames dropped)."""
    out = []
    with open(hits_csv, newline="") as f:
        for r in csv.DictReader(f):
            j = int(r["frame"])
            if 0 <= j < len(fmap) and fmap[j] >= 0:
                out.append({"frame": fmap[j], "clip_frame": j, "player_id": None, "found_by": r.get("found_by", "")})
    return out


def score_match(clip: Path, tols=(3, 6), shift: int = 0) -> dict:
    """Score one clip; `shift` frames are added to every BFMD hit (see docs/hits.md: -1)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from badminton import hits as H

    side = json.loads(clip.with_suffix(".framemap.json").read_text())
    doc = json.loads(Path(side["rallies_json"]).read_text())
    rallies = doc["rallies"]
    fmap = side["frame_map"]
    hits_csv = clip.with_suffix(".hits.csv")
    dets = detections_in_video_frames(hits_csv, fmap)
    pad = max(tols)
    dets = [d for d in dets if in_spans(d["frame"], rallies, pad)]
    labels = [{"frame": h + shift, "player_id": None, "note": ""} for h in doc["hits"] if in_spans(h, rallies)]
    res = {"match": doc["match"], "rallies": len(rallies), "labels": len(labels), "detected": len(dets)}
    for t in tols:
        r = H.score_hits(dets, labels, t)
        res[t] = {k: r[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1",
                                    "missed_frames", "extra_frames", "offset_frames")}
    return res


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def cmd_score(a) -> int:
    tols = tuple(int(t) for t in a.tol.split(","))
    rows = [score_match(Path(c), tols, a.shift) for c in a.clips]
    tot = {t: {k: sum(r[t][k] for r in rows) for k in ("tp", "fp", "fn")} for t in tols}
    head = "| 比赛 | 回合 | 对照拍数 | 识别 | " + " | ".join(
        f"±{t} 帧 准确率 | 召回率 | F1" for t in tols) + " |"
    print(head)
    print("|" + " --- |" * (4 + 3 * len(tols)))

    def cells(c):
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return f"{_pct(p)} | {_pct(r)} | {_pct(f)}"

    for r in rows:
        print(f"| {r['match'][:60]} | {r['rallies']} | {r['labels']} | {r['detected']} | "
              + " | ".join(cells(r[t]) for t in tols) + " |")
    print(f"| 合计 | {sum(r['rallies'] for r in rows)} | {sum(r['labels'] for r in rows)} | "
          f"{sum(r['detected'] for r in rows)} | " + " | ".join(cells(tot[t]) for t in tols) + " |")
    off = [o for r in rows for o in r[tols[0]]["offset_frames"]]
    if off:
        o = np.array(off)
        print(f"±{tols[0]} 帧内对上的 {len(o)} 拍：识别减标注的中位数 {np.median(o):+.0f} 帧，平均 {o.mean():+.2f} 帧，"
              f"|差| ≤1 帧 {_pct(np.mean(np.abs(o) <= 1))}，≤2 帧 {_pct(np.mean(np.abs(o) <= 2))}")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "total": tot}, ensure_ascii=False, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("labels", help="BFMD hits_doubles JSON -> 对照 CSV（整场帧号）和回合区间 JSON")
    s.add_argument("json")
    s.add_argument("-o", "--out-dir", required=True)
    s.add_argument("--name", help="比赛名，默认取 JSON 文件名")
    s = sub.add_parser("clip", help="把回合区间切成一个视频，另写帧号对照表和片段帧号的对照 CSV")
    s.add_argument("video")
    s.add_argument("--rallies", required=True, help="labels 写的 <比赛>.rallies.json")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--margin", type=int, default=15, help="回合前后各多留几帧（默认 15）")
    s.add_argument("--hold", type=int, default=80, help="回合之间定格几帧（默认 80，长于 hits.RALLY_GAP_S）")
    s.add_argument("--crf", type=int, default=18)
    s.add_argument("--encoder", default="libx264")
    s = sub.add_parser("score", help="把 badminton hits 的结果换回整场帧号，按回合内打分")
    s.add_argument("clips", nargs="+", help="clip 输出的视频（旁边要有 .framemap.json 和 .hits.csv）")
    s.add_argument("--tol", default="3,6", help="容许误差（帧），逗号分隔，第一个是主口径")
    s.add_argument("--shift", type=int, default=0, help="BFMD 标注帧号整体加几帧再比（双打标注比击球晚约 1 帧，用 -1）")
    s.add_argument("--json", help="另存逐场结果（含漏检、误检帧号）")
    a = p.parse_args(argv)
    return {"labels": cmd_labels, "clip": cmd_clip, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
