"""Probe: how well does TrackNetV3 (third party, MIT) find the shuttle in a clip?

TrackNetV3 is kept outside this package. Clone it and download its weights
as its README says, give it its own virtualenv, then:

  export TRACKNETV3_DIR=~/codebase/third_party/TrackNetV3
  python scripts/probe_tracknet.py data/singles.mp4 --out outputs/tracknet

Steps (each skipped if its output exists, so the slow prediction runs once):
  1. run TrackNetV3's predict.py (with InpaintNet) -> <name>_ball.csv
  2. detection rate overall and per time range (--ranges "0-8.4,8.4-16.6")
  3. overlay video with a trajectory tail: <name>.shuttle.mp4
  4. audit sheet: N random frames, a crop around the prediction (or the
     whole frame scaled down if nothing was predicted), for checking by eye
  5. plot of shuttle y over time per range, to see whether hits show up as
     sharp turns
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def run_tracknet(video: Path, out: Path, batch: int) -> Path:
    root = Path(os.environ.get("TRACKNETV3_DIR", "")).expanduser()
    if not (root / "predict.py").exists():
        sys.exit("设置 TRACKNETV3_DIR 指向 TrackNetV3 仓库（含 ckpts/TrackNet_best.pt 和 InpaintNet_best.pt）")
    python = os.environ.get("TRACKNETV3_PYTHON", str(root / ".venv" / "bin" / "python"))
    csv = out / f"{video.stem}_ball.csv"
    if csv.exists():
        return csv
    cmd = [python, "predict.py", "--video_file", str(video.resolve()),
           "--tracknet_file", "ckpts/TrackNet_best.pt", "--inpaintnet_file", "ckpts/InpaintNet_best.pt",
           "--save_dir", str(out.resolve()), "--large_video", "--batch_size", str(batch)]
    t = time.perf_counter()
    subprocess.run(cmd, cwd=root, check=True)
    print(f"TrackNetV3 用时 {time.perf_counter() - t:.0f} 秒")
    return csv


def load_csv(path: Path) -> np.ndarray:
    """-> (n, 4) array: frame, visibility, x, y (pixels)."""
    import csv

    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([[int(r["Frame"]), int(r["Visibility"]), float(r["X"]), float(r["Y"])] for r in rows])


def parse_ranges(text: str | None, duration: float) -> list[tuple[float, float]]:
    if not text:
        return [(0.0, duration)]
    return [tuple(float(v) for v in part.split("-")) for part in text.split(",")]


def overlay(video: Path, pred: np.ndarray, out_path: Path, tail: int = 12) -> None:
    from badminton.video import VideoWriter, probe, read_frames

    info = probe(video)
    by_frame = {int(f): (v, x, y) for f, v, x, y in pred}
    hist: list[tuple[int, int] | None] = []
    w = VideoWriter(out_path, info.width, info.height, info.fps or 30)
    try:
        for i, frame in read_frames(video, 0, None):
            v, x, y = by_frame.get(i, (0, 0, 0))
            hist.append((int(x), int(y)) if v else None)
            pts = hist[-tail:]
            for k in range(1, len(pts)):
                if pts[k] and pts[k - 1]:
                    cv2.line(frame, pts[k - 1], pts[k], (0, 255, 255), 2, cv2.LINE_AA)
            if pts and pts[-1]:
                cv2.circle(frame, pts[-1], 9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{i / (info.fps or 30):6.2f}s {'' if v else 'no shuttle'}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            w.write(frame)
    finally:
        w.close()


def audit_sheet(video: Path, pred: np.ndarray, frames: list[int], fps: float, out_path: Path) -> None:
    """Grid of crops for checking by eye. Red circle = prediction."""
    from badminton.video import read_frame

    by_frame = {int(f): (v, x, y) for f, v, x, y in pred}
    tiles = []
    for n, i in enumerate(frames):
        img = read_frame(video, i)
        v, x, y = by_frame.get(i, (0, 0, 0))
        if v:
            cv2.circle(img, (int(x), int(y)), 14, (0, 0, 255), 2, cv2.LINE_AA)
            cx, cy = int(np.clip(x, 200, img.shape[1] - 200)), int(np.clip(y, 150, img.shape[0] - 150))
            tile = img[cy - 150:cy + 150, cx - 200:cx + 200]
        else:
            tile = cv2.resize(img, (400, 225))
            tile = cv2.copyMakeBorder(tile, 37, 38, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        tile = tile.copy()
        cv2.putText(tile, f"#{n + 1} {i / fps:.2f}s {'pred' if v else 'NONE'}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), np.vstack(rows))


def plot_y(pred: np.ndarray, fps: float, ranges, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ranges), 1, figsize=(12, 2.2 * len(ranges)), squeeze=False)
    t = pred[:, 0] / fps
    for ax, (a, b) in zip(axes[:, 0], ranges):
        m = (t >= a) & (t < b)
        vis = m & (pred[:, 1] > 0)
        ax.plot(t[vis], pred[vis, 3], ".", ms=3)
        ax.invert_yaxis()
        ax.set_xlim(a, b)
        ax.set_ylabel("y px")
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from badminton.video import probe

    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", default="outputs/tracknet")
    p.add_argument("--ranges", help="时间段（秒），逗号分隔，如 0-8.4,8.4-16.6；检出率按段统计")
    p.add_argument("--audit", type=int, default=40, help="抽查多少帧")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()

    video, out = Path(a.video), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    fps = info.fps or 30
    pred = load_csv(run_tracknet(video, out, a.batch))
    ranges = parse_ranges(a.ranges, info.frames / fps)

    t = pred[:, 0] / fps
    print(f"总帧数 {len(pred)}，有球的帧 {int(pred[:, 1].sum())}（{100 * pred[:, 1].mean():.1f}%）")
    for lo, hi in ranges:
        m = (t >= lo) & (t < hi)
        print(f"  {lo:6.1f}–{hi:6.1f} 秒：检出率 {100 * pred[m, 1].mean():5.1f}%（{int(m.sum())} 帧）")

    shuttle_video = out / f"{video.stem}.shuttle.mp4"
    if not shuttle_video.exists():
        overlay(video, pred, shuttle_video)
    rng = np.random.default_rng(a.seed)
    in_range = [int(f) for f in pred[:, 0] if any(lo <= f / fps < hi for lo, hi in ranges)]
    sample = sorted(rng.choice(in_range, size=min(a.audit, len(in_range)), replace=False).tolist())
    audit_sheet(video, pred, sample, fps, out / f"{video.stem}.audit.jpg")
    plot_y(pred, fps, ranges, out / f"{video.stem}.y.png")
    print(f"已写入 {shuttle_video}、{out / (video.stem + '.audit.jpg')}、{out / (video.stem + '.y.png')}")
    print("抽查帧（秒）：" + ", ".join(f"#{n + 1} {f / fps:.2f}" for n, f in enumerate(sample)))


if __name__ == "__main__":
    main()
