"""Trial run: stroke types on our own pose + shuttle data with BST's released ShuttleSet weights.

    badminton skeleton data/singles.mp4 --court data/singles.court.json --singles --stride 1 \
        -o outputs/bst/singles.skeleton.mp4
    badminton shuttle data/singles.mp4 --court data/singles.court.json -o outputs/bst/singles.shuttle.csv
    python scripts/bst_trial.py --skeleton outputs/bst/singles.skeleton.npz \
        --shuttle outputs/bst/singles.shuttle.csv --court data/singles.court.json \
        --weights ~/codebase/third_party/BST_weights/weight/bst_CG_AP_JnB_bone_merged*.pt \
        --video outputs/bst/singles.skeleton.mp4 -o outputs/bst/singles

Writes <out>.strokes.csv (one row per hit) and <out>.inputs.npz (the arrays fed
to the network), and with --video <out>.strokes.mp4 with each stroke written
next to the player who hit it. Needs torch (any recent version). See
docs/bst_trial.md for what the numbers mean and what is still rough.

Hits come from --hits (a CSV with a `frame` column, e.g. from the hit
detector) or, without it, from the simple finder in badminton.bst.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from badminton import bst  # noqa: E402
from badminton.court import CourtCalibration  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skeleton", required=True, help="skeleton.npz from `badminton skeleton --singles --stride 1`")
    p.add_argument("--shuttle", required=True, help="CSV from `badminton shuttle` (Frame,Visibility,X,Y)")
    p.add_argument("--court", required=True, help="court.json")
    p.add_argument("--weights", required=True, nargs="+",
                   help="BST-CG-AP weights for 25 classes and seq_len 30 (bst_CG_AP_JnB_bone_merged*.pt); "
                        "several files are averaged")
    p.add_argument("--hits", help="CSV with a `frame` column; default: the simple hit finder")
    p.add_argument("--video", help="draw the result on this video (same frames as the skeleton file)")
    p.add_argument("-o", "--out", required=True, help="output prefix")
    a = p.parse_args()

    skel = dict(np.load(a.skeleton))
    calib = CourtCalibration.load(a.court)
    shuttle_px = bst.read_shuttle_csv(a.shuttle, int(skel["frame"][-1]) + 1)
    seq = bst.build_sequence(skel, shuttle_px, calib)
    height = float(skel["frame_size"][1])
    print(f"{len(seq.ok)} 帧，{seq.fps:g} fps；两名球员都在场内的帧 {seq.ok.mean():.1%}，"
          f"看得见球的帧 {np.isfinite(seq.shuttle_px).all(1).mean():.1%}")

    if a.hits:
        with open(a.hits, newline="") as f:
            frames = [int(float(r["frame"])) for r in csv.DictReader(f)]
        found = {}
    else:
        hits = bst.find_hits(seq, height)
        frames = [h.frame for h in hits]
        found = {h.frame: h for h in hits}
    print(f"{len(frames)} 次击球" + ("" if a.hits else "（简单切段：球速突变且球在球员手边）"))

    inputs = bst.model_inputs(seq, frames)
    models = [bst.load_model(w) for w in a.weights]
    probs = bst.classify(models, inputs)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{out}.inputs.npz", frame=np.array(frames), **inputs, probs=probs)
    rows = []
    for i, f in enumerate(frames):
        top = np.argsort(-probs[i])[:3]
        who, name = bst.class_name(int(top[0]))
        row = {"frame": f, "time_s": round(f / seq.fps, 2), "who": who or "", "stroke": name,
               "stroke_en": bst.class_name(int(top[0]), "en")[1], "prob": round(float(probs[i, top[0]]), 3),
               "ok_frames": int(seq.ok[slice(*bst.clip_window(f, seq.fps, len(seq.ok)))].sum())}
        for n, k in enumerate(top[1:], start=2):
            w, s = bst.class_name(int(k))
            row[f"top{n}"] = f"{w or ''}:{s}"
            row[f"prob{n}"] = round(float(probs[i, k]), 3)
        if f in found:
            row["finder_who"] = found[f].who
            row["turn"] = round(found[f].turn, 1)
            row["reach"] = round(found[f].reach, 2)
        rows.append(row)
    if rows:
        with open(f"{out}.strokes.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    side_zh = {"far": "远端", "near": "近端", "": "？"}
    for r in rows:
        agree = "" if "finder_who" not in r or not r["who"] or r["who"] == r["finder_who"] else "  (切段认为是另一名球员)"
        print(f"  {r['time_s']:6.2f}s  {side_zh[r['who']]} {r['stroke']:<5} {r['prob']:.2f}   "
              f"其次 {r['top2']} {r['prob2']:.2f}{agree}")
    print(f"已写入 {out}.strokes.csv 和 {out}.inputs.npz")

    if a.video:
        draw_video(a.video, f"{out}.strokes.mp4", rows, skel, seq.fps)
        print(f"已写入 {out}.strokes.mp4")
    return 0


def draw_video(src: str, dst: str, rows: list[dict], skel: dict, fps: float, hold_s: float = 1.0) -> None:
    """Each stroke's name above the player who (according to the model) hit it, for `hold_s` after the hit."""
    import cv2
    from PIL import Image, ImageDraw

    from badminton.video import VideoWriter, probe, read_frames

    info = probe(src)
    font = _cjk_font(max(16, round(info.height * 0.035)))
    hold = round(hold_s * fps)
    first = int(skel["frame"][0])
    with VideoWriter(dst, info.width, info.height, info.fps) as wr:
        for i, frame in read_frames(src):
            f = first + i
            live = [r for r in rows if 0 <= f - r["frame"] < hold]
            if live:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                d = ImageDraw.Draw(img, "RGBA")
                for r in live:
                    text = f"{r['stroke']} {r['prob']:.0%}"
                    side = {"far": bst.FAR, "near": bst.NEAR}.get(r["who"])
                    k = min(i, len(skel["frame"]) - 1)
                    slot = np.flatnonzero(skel["valid"][k] & (skel["side"][k] == side)) if side is not None else []
                    if len(slot):
                        x0, y0, x1, _ = skel["bbox"][k, slot[0]]
                        xy = ((x0 + x1) / 2, y0 - font.size * 1.6)
                    else:
                        xy = (info.width / 2, info.height * 0.08)
                    box = d.textbbox(xy, text, font=font, anchor="ma")
                    pad = font.size * 0.3
                    d.rounded_rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], radius=pad,
                                        fill=(0, 0, 0, 170))
                    d.text(xy, text, font=font, fill=(255, 235, 59, 255), anchor="ma")
                frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            wr.write(frame)


def _cjk_font(size: int):
    """A font with simplified Chinese glyphs. A .ttc holds several faces and the first one is not
    always the one matplotlib listed (PingFang.ttc starts with the Hong Kong face, which lacks 杀 and 远)."""
    from matplotlib import font_manager
    from PIL import ImageFont

    from badminton.card import _CJK_FONTS

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


if __name__ == "__main__":
    raise SystemExit(main())
