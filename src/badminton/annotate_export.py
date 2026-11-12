"""Write labelled rallies in the directory layout TrackNetV3's dataset.py reads.

    <out>/
      manifest.json                      what was exported (ours; TrackNetV3 ignores it)
      train/match<N>/
        median.npz                       key "median": RGB uint8 (H, W, 3), background of the whole video
        csv/<rally>_ball.csv             Frame,Visibility,X,Y ; Frame = 0..n-1
        frame/<rally>/0.png ... n-1.png  the frames; X,Y are pixels of THESE images
        frame/<rally>/median.npz         same format, median of this rally's frames
      val/match<N>/...                   same; whole rallies, never frames of a train rally

One video is one "match". A rally is cut wherever a frame has no usable label
(not reviewed, or marked occluded), because TrackNetV3 trains on windows of 8
consecutive frames; pieces shorter than a window are dropped. The pieces of one
rally always land in the same split. See docs/tracknet_finetune.md.

By default the frames are written at the model's input size (512x288), made
with the same Pillow bicubic resize as `badminton shuttle`, and the labels are
scaled to match: dataset.py then has nothing left to resize, training sees
exactly the pixels inference sees, and a rally takes ~30x less disk than
1080p PNGs. `full_size=True` keeps the video's resolution instead.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import __version__
from .annotate import REVIEWED, Label, read_labels

SEQ_LEN = 8  # TrackNet's window: the shortest usable piece
RALLY_MEDIAN_SAMPLES = 100
FULL_MEDIAN_SAMPLES = 100  # full-size frames held in memory for the video median (1080p: 0.6 GB)
CACHE_GLOBS = ("img_config_*.npz", "data_l*_s*_*.npz")  # dataset.py's caches, stale after any export


def usable_runs(labels: dict[int, Label], start: int, end: int, statuses=REVIEWED, min_len: int = SEQ_LEN) -> list:
    """Stretches [a, b] (inclusive) of consecutive frames in start..end whose label
    has one of `statuses` ("skip" never counts), at least `min_len` long."""
    runs = []
    a = None
    for f in range(start, end + 2):
        ok = f <= end and f in labels and labels[f].status in statuses and labels[f].status != "skip"
        if ok and a is None:
            a = f
        elif not ok and a is not None:
            if f - a >= min_len:
                runs.append((a, f - 1))
            a = None
    return runs


def split_rallies(n: int, val_frac: float = 0.2) -> list[str]:
    """"train"/"val" for n rallies: about val_frac of them, evenly spread, go to val;
    at least one when there are two or more, none when there is only one."""
    if n < 2 or val_frac <= 0:
        return ["train"] * n
    n_val = min(n - 1, max(1, round(val_frac * n)))
    val = {int((i + 0.5) * n / n_val) for i in range(n_val)}
    return ["val" if i in val else "train" for i in range(n)]


def _label_runs(labels: dict[int, Label], rallies: list | None, statuses) -> list:
    """[(rally_index, [(a, b), ...])] for rallies with at least one usable piece."""
    if not rallies:  # no rally list: every stretch of usable labels is its own rally
        runs = usable_runs(labels, min(labels, default=0), max(labels, default=-1), statuses)
        return [(i, [r]) for i, r in enumerate(runs)]
    out = []
    for i, (a, b) in enumerate(rallies):
        runs = usable_runs(labels, int(a), int(b), statuses)
        if runs:
            out.append((i, runs))
    return out


def _video_median_full(video, samples: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, n // samples)
    imgs = []
    for i in range(n):
        if i % step:
            if not cap.grab():
                break
            continue
        ok, f = cap.read()
        if not ok:
            break
        imgs.append(f[..., ::-1])
    cap.release()
    return np.median(np.stack(imgs), axis=0).astype(np.uint8)


def export_video(video: str | Path, labels_csv: str | Path, out_dir: str | Path, match: int = 1,
                 rallies: list | None = None, val_frac: float = 0.2, include_unreviewed: bool = False,
                 full_size: bool = False, progress=None) -> dict:
    """Export one video's labels as train/match<match> and val/match<match>.
    Returns the manifest entry. An earlier export of the same match is replaced."""
    import cv2
    from PIL import Image

    from .shuttle import HEIGHT, WIDTH, _resize, median_background

    video, out_dir = Path(video), Path(out_dir)
    labels = read_labels(labels_csv)
    statuses = REVIEWED + (("machine",) if include_unreviewed else ())
    units = _label_runs(labels, rallies, statuses)
    if not units:
        raise ValueError(f"{labels_csv}: 没有可导出的标注（至少要有连续 {SEQ_LEN} 帧已确认的标注；"
                         "只有机器预标注时加 --include-unreviewed）")
    splits = split_rallies(len(units), val_frac)

    for split in ("train", "val"):
        shutil.rmtree(out_dir / split / f"match{match}", ignore_errors=True)
    for pattern in CACHE_GLOBS:
        for p in out_dir.glob(pattern):
            p.unlink()

    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"读不出视频画面：{video}")
    src_h, src_w = first.shape[:2]
    out_w, out_h = (src_w, src_h) if full_size else (WIDTH, HEIGHT)
    sx, sy = out_w / src_w, out_h / src_h
    median = _video_median_full(video, FULL_MEDIAN_SAMPLES) if full_size else median_background(video)

    total = sum(b - a + 1 for _, runs in units for a, b in runs)
    done = 0
    t0 = time.perf_counter()
    segments = []
    for (rally_i, runs), split in zip(units, splits):
        match_dir = out_dir / split / f"match{match}"
        (match_dir / "csv").mkdir(parents=True, exist_ok=True)
        np.savez(match_dir / "median.npz", median=median)
        for a, b in runs:
            rally_id = f"r{rally_i:03d}_f{a:06d}"
            rally_dir = match_dir / "frame" / rally_id
            rally_dir.mkdir(parents=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            n = b - a + 1
            keep = set(np.linspace(0, n - 1, min(n, RALLY_MEDIAN_SAMPLES)).astype(int).tolist())
            sample, rows, counts = [], [], {}
            for k in range(n):
                ok, img = cap.read()
                if not ok:
                    raise RuntimeError(f"{video}: 第 {a + k} 帧读不出来")
                rgb = np.ascontiguousarray(img[..., ::-1]) if full_size else _resize(img)
                Image.fromarray(rgb).save(rally_dir / f"{k}.png", compress_level=3)
                if k in keep:
                    sample.append(rgb)
                lab = labels[a + k]
                counts[lab.status] = counts.get(lab.status, 0) + 1
                if lab.visibility:
                    x, y = lab.x * sx, lab.y * sy
                    rows.append((k, 1, int(round(x)) if full_size else round(x, 2), int(round(y)) if full_size else round(y, 2)))
                else:
                    rows.append((k, 0, 0, 0))
                done += 1
                if progress:
                    progress(done, total, time.perf_counter() - t0)
            np.savez(rally_dir / "median.npz", median=np.median(np.stack(sample), axis=0).astype(np.uint8))
            with open(match_dir / "csv" / f"{rally_id}_ball.csv", "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["Frame", "Visibility", "X", "Y"])
                wr.writerows(rows)
            segments.append({"id": rally_id, "split": split, "rally": rally_i, "source_start": a, "source_end": b,
                             "frames": n, "visible": sum(r[1] for r in rows), "status_counts": counts})
    cap.release()
    return {"match": match, "video": str(video), "labels": str(labels_csv), "source_size": [src_w, src_h],
            "frame_size": [out_w, out_h], "label_scale": [sx, sy], "include_unreviewed": include_unreviewed,
            "segments": segments}


def export_dataset(items: list, out_dir: str | Path, val_frac: float = 0.2, include_unreviewed: bool = False,
                   full_size: bool = False, progress=None) -> dict:
    """items: [(video, labels_csv, rallies or None)]. Videos keep their match number
    across exports (looked up by path in manifest.json); new ones get the next free number."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"matches": []}
    known = {m["video"]: m for m in manifest["matches"]}
    for video, labels_csv, rallies in items:
        key = str(Path(video))
        match = known[key]["match"] if key in known else max([m["match"] for m in known.values()], default=0) + 1
        known[key] = export_video(video, labels_csv, out_dir, match, rallies, val_frac, include_unreviewed,
                                  full_size, progress)
    manifest = {"tool": "badminton annotate-export", "tool_version": __version__,
                "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "layout": "TrackNetV3 dataset.py",
                "matches": sorted(known.values(), key=lambda m: m["match"])}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest
