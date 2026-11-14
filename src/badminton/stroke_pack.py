"""Stroke-type labelling on the phone: a short clip of every hit and the model's guess.

`badminton annotate-strokes VIDEO --court C --analysis-dir D -o OUT` uses the
`badminton strokes` result (<video>.strokes.json in D; made here when missing)
and writes to OUT:

  strokes.mp4         one clip per hit, all the same length, one after another:
                      clip k starts at k * clip.frames / clip.fps seconds, with a
                      key frame there so the page can jump to it exactly. The
                      hitter has a yellow box; the frame of the hit has a red edge.
  strokes.state.json  the clips' layout and per hit the model's type, probability,
                      alternatives, direction and whether it needs checking.

The labels made on the page come back as <video>.strokes.labels.csv with the
columns frame,player_id,type,tags,note (tags separated by ";").
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import __version__

BEFORE_S = 0.8        # clip starts this long before the hit
AFTER_S = 0.6         # and ends this long after
WIDTH = 960
TYPES = ["放网", "挡网", "杀球", "点杀", "挑球", "防守挑球", "高远球", "平高球", "平抽", "后场抽平", "吊球",
         "过渡吊球", "推球", "扑球", "防守抽球", "勾对角", "发短球", "发长球", "未知"]
TAGS = ["劈杀", "劈吊", "收吊", "搓球", "直线", "对角"]
CSV_COLUMNS = ["frame", "player_id", "type", "tags", "note"]


def find_inputs(work: Path, analysis_dir: Path | None) -> dict:
    """skeleton, shuttle, players and strokes files of the analysis, wherever they are."""
    dirs = [d for d in (analysis_dir, work.parent) if d]
    out = {}
    for key, suffix in (("data", ".skeleton.npz"), ("shuttle", ".shuttle.csv"), ("players", ".players.json"),
                        ("strokes", ".strokes.json")):
        out[key] = next((d / f"{work.stem}{suffix}" for d in dirs if (d / f"{work.stem}{suffix}").exists()), None)
    return out


def load_or_classify(work: Path, court: Path, analysis_dir: Path | None, singles: bool | None = None,
                     weights=None, run=None) -> tuple[dict, dict | None]:
    """(strokes.json document, ctx with data + tracking for drawing, or None)."""
    from . import strokes as ST
    from .court import CourtCalibration

    found = find_inputs(work, analysis_dir)
    calib = CourtCalibration.load(court)
    names = json.loads(found["players"].read_text()) if found["players"] else {}
    if found["strokes"] and found["shuttle"] and found["strokes"].stat().st_mtime >= found["shuttle"].stat().st_mtime:
        print(f"沿用 {found['strokes'].name}")
        doc = json.loads(found["strokes"].read_text())
        ctx = None
        if found["data"]:
            try:
                from .tracking import load_npz, track_players

                data = load_npz(found["data"])
                ctx = {"data": data, "tracking": track_players(data, calib, _per_side(doc, singles))}
            except Exception as e:  # boxes are a help, not a must
                print(f"画不了击球人的框：{e}")
        return doc, ctx
    for key, how in (("data", "badminton analyze"), ("shuttle", "badminton shuttle --court")):
        if not found[key]:
            raise SystemExit(f"找不到 {work.stem}{'.skeleton.npz' if key == 'data' else '.shuttle.csv'}：先运行 {how}")
    weights = [Path(w) for w in (weights or [ST.DEFAULT_WEIGHTS])]
    for w in weights:
        if not w.exists():
            raise SystemExit(f"找不到识别击球类型用的 BST 权重 {w}：下载办法见 docs/stroke_types.md")
    if singles is None:
        from .cli import _players_per_side

        per_side = _players_per_side(found["data"])
    else:
        per_side = 1 if singles else 2
    print("识别每一拍的技术类型（badminton strokes）")
    results, ctx = (run or ST.run)(found["shuttle"], found["data"], calib, weights, None, per_side)
    doc = ST.strokes_json(results, ctx["fps"], str(work), {"skeleton": str(found["data"]), "shuttle": str(found["shuttle"])},
                          names, [str(w) for w in weights])
    out = (analysis_dir or work.parent) / f"{work.stem}.strokes.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    ST.write_csv(results, out.with_suffix(".csv"))
    return doc, ctx


def _per_side(doc: dict, singles: bool | None) -> int:
    if singles is not None:
        return 1 if singles else 2
    ids = {(s.get("side"), s.get("player_id")) for s in doc.get("strokes", [])}
    return 2 if max([sum(1 for side, _ in ids if side == x) for x in ("near", "far")] or [1]) > 1 else 1


def hitter_box(ctx: dict | None, stroke: dict, frame: int) -> tuple | None:
    """Box of the hitter (video pixels) in the skeleton sample nearest to `frame`."""
    if not ctx:
        return None
    data, tracking = ctx["data"], ctx["tracking"]
    frames = np.asarray(data["frame"], int)
    row = int(np.clip(np.searchsorted(frames, frame), 0, len(frames) - 1))
    for t in tracking.tracks:
        if t.id == stroke.get("player_id"):
            det = int(t.det_index[row])
            if det >= 0:
                return tuple(float(v) for v in data["bbox"][row, det][:4])
    return None


def encoder_args(clip_frames: int) -> list[list[str]]:
    """libx264 with a key frame at every clip start; the Mac's hardware encoder as a fallback."""
    return [["-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
             "-force_key_frames", f"expr:eq(mod(n,{clip_frames}),0)", "-g", str(clip_frames)],
            ["-c:v", "h264_videotoolbox", "-b:v", "2500k", "-g", str(clip_frames)]]


def write_clips(video: Path, strokes: list[dict], out: Path, ctx: dict | None, width: int = WIDTH,
                progress=None) -> dict:
    """`progress(done, total, elapsed_s)` as pipeline.print_progress."""
    import cv2

    from .video import probe

    info = probe(video)
    fps = info.fps or 30.0
    before, after = round(BEFORE_S * fps), round(AFTER_S * fps)
    n_clip = before + after + 1
    w = min(width, info.width)
    w -= w % 2
    h = round(info.height * w / info.width)
    h -= h % 2
    s = w / info.width
    if not shutil.which("ffmpeg"):
        raise SystemExit("需要 ffmpeg 来写击球小视频")
    total = len(strokes) * n_clip
    t0 = time.monotonic()
    for args in encoder_args(n_clip):
        tmp = out.with_name(out.name + ".tmp.mp4")
        proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                 "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", *args, "-pix_fmt", "yuv420p",
                                 "-movflags", "+faststart", str(tmp)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        cap = cv2.VideoCapture(str(video))
        done = 0
        try:
            for st in strokes:
                f0 = int(st["frame"]) - before
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0))
                last = None
                for k in range(n_clip):
                    f = f0 + k
                    img = None
                    if f >= 0:
                        ok, img = cap.read()
                        img = img if ok else None
                    if img is None:  # before the start or after the end: hold the nearest frame
                        img = last if last is not None else np.zeros((info.height, info.width, 3), np.uint8)
                    last = img
                    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    box = hitter_box(ctx, st, max(0, f))
                    if box:
                        x0, y0, x1, y1 = (round(v * s) for v in box)
                        cv2.rectangle(small, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 220, 255), 2)
                    if abs(k - before) <= 1:  # the hit
                        cv2.rectangle(small, (0, 0), (w - 1, h - 1), (40, 40, 230), 6)
                    proc.stdin.write(small.tobytes())
                    done += 1
                    if progress and done % 30 == 0:
                        progress(done, total, time.monotonic() - t0)
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "replace")
            code = proc.wait()
        except BrokenPipeError:
            err, code = proc.stderr.read().decode("utf-8", "replace"), proc.wait() or 1
        finally:
            cap.release()
        if code == 0 and tmp.exists() and tmp.stat().st_size:
            os.replace(tmp, out)
            if progress:
                progress(total, total, time.monotonic() - t0)
            return {"fps": fps, "frames": n_clip, "hit": before, "size": [w, h], "video_size": [info.width, info.height]}
        print(f"编码失败（{args[1]}）：{err.strip()[-300:]}")
        tmp.unlink(missing_ok=True)
    raise SystemExit("写击球小视频失败")


def pack(work: str | Path, out_dir: str | Path, court: str | Path, analysis_dir: str | Path | None = None,
         singles: bool | None = None, weights=None, progress=None, run=None) -> dict:
    work, out_dir = Path(work), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc, ctx = load_or_classify(work, Path(court), Path(analysis_dir) if analysis_dir else None, singles, weights, run)
    strokes = sorted(doc.get("strokes", []), key=lambda r: int(r["frame"]))
    if not strokes:
        raise SystemExit("这段录像里没有找到击球")
    clip = write_clips(work, strokes, out_dir / "strokes.mp4", ctx, progress=progress)
    players = {}
    for r in strokes:
        pid = r.get("player_id")
        if pid is not None and str(pid) not in players:
            players[str(pid)] = {"id": pid, "name": r.get("player_name") or "", "side": r.get("side")}
    state = {
        "version": 1, "tool": "badminton annotate-strokes", "tool_version": __version__,
        "video": {"file": work.name, "fps": clip["fps"], "size": clip["video_size"]},
        "clip": {k: clip[k] for k in ("fps", "frames", "hit", "size")},
        "schema": doc.get("schema"), "types": TYPES, "tags": TAGS,
        "players": sorted(players.values(), key=lambda p: (p["side"] or "", str(p["id"]))),
        "strokes": [{
            "frame": int(r["frame"]), "time_s": r.get("time_s"), "player_id": r.get("player_id"),
            "player_name": r.get("player_name") or "", "side": r.get("side"), "type": r.get("type"),
            "group": r.get("group"), "prob": r.get("prob"), "needs_review": bool(r.get("needs_review")),
            "direction": ((r.get("attributes") or {}).get("direction") or {}).get("value"),
            "alternatives": r.get("alternatives") or [],
        } for r in strokes],
    }
    tmp = out_dir / "strokes.state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, separators=(",", ":")))
    os.replace(tmp, out_dir / "strokes.state.json")
    return state


def write_labels_csv(path: str | Path, labels: dict) -> int:
    """labels from the site: {frame: {player_id, type, tags: [...], note}} -> CSV in frame order."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for frame in sorted(labels, key=int):
            lab = labels[frame]
            pid = lab.get("player_id")
            w.writerow([int(frame), "" if pid is None else pid, lab.get("type") or "",
                        ";".join(lab.get("tags") or []), lab.get("note") or ""])
    os.replace(tmp, path)
    return len(labels)
