"""Step 1: skeleton overlay video + per-frame pose data."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .court import CourtCalibration, detection_roi
from .players import Players, select_players
from .pose import PoseResult
from .render import draw_court, draw_players, draw_roi, draw_text
from .video import VideoWriter, probe, read_frames


class Estimator(Protocol):
    def __call__(self, frame: np.ndarray, roi=None) -> PoseResult: ...


@dataclass
class SkeletonOptions:
    stride: int = 2  # run the models on every Nth frame (2 = 30 fps for 60 fps video)
    start_s: float = 0.0
    duration_s: float | None = None
    players_per_side: int = 2  # 2 for doubles, 1 for singles
    draw_court: bool = True
    draw_roi: bool = False
    use_roi: bool = True  # detect inside the court area only
    kpt_thr: float = 0.3
    encoder: str = "libx264"
    max_persons: int = 4  # rows reserved per frame in the saved arrays


def run_skeleton(
    video: str | Path,
    estimator: Estimator,
    out_video: str | Path | None,
    out_data: str | Path | None,
    calib: CourtCalibration | None = None,
    opts: SkeletonOptions | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> dict:
    """Run detection + pose over a video, write the overlay video and an .npz of poses.

    The .npz holds, for each processed frame f and person slot p:
      frame[f], time_s[f], valid[f, p], bbox[f, p, 4], keypoints[f, p, 17, 2],
      scores[f, p, 17], court_xy[f, p, 2] (metres, NaN without calibration), side[f, p]
    """
    opts = opts or SkeletonOptions()
    info = probe(video)
    fps = info.fps or 30.0
    if calib is not None and tuple(calib.frame_size) != (info.width, info.height):
        raise ValueError(f"球场标定的画面尺寸 {calib.frame_size} 和视频 {info.width}x{info.height} 不一致")
    start = int(round(opts.start_s * fps))
    max_frames = int(round(opts.duration_s * fps)) if opts.duration_s else None
    total = max_frames or max(0, info.frames - start)
    roi = detection_roi(calib) if (calib is not None and opts.use_roi) else None
    P = max(opts.max_persons, 2 * opts.players_per_side)

    rec = {k: [] for k in ("frame", "valid", "bbox", "keypoints", "scores", "court_xy", "side")}
    writer = None
    if out_video:
        writer = VideoWriter(out_video, info.width, info.height, fps,
                             audio_from=video if info.has_audio else None,
                             audio_start_s=start / fps, encoder=opts.encoder)
    last: Players | None = None
    t0 = time.perf_counter()
    n = 0
    try:
        for i, frame in read_frames(video, start, max_frames):
            if (i - start) % opts.stride == 0:
                last = select_players(estimator(frame, roi), calib, opts.players_per_side)
                _record(rec, i, last, P)
            if writer:
                if opts.draw_court and calib is not None:
                    draw_court(frame, calib, (255, 255, 255), 1)
                if opts.draw_roi and roi is not None:
                    draw_roi(frame, roi)
                if last is not None:
                    draw_players(frame, last, opts.kpt_thr, scale=info.height / 1080)
                draw_text(frame, f"{i / fps:7.2f}s")
                writer.write(frame)
            n += 1
            if progress:
                progress(n, total, time.perf_counter() - t0)
    finally:
        if writer:
            writer.close()

    data = {k: np.asarray(v) for k, v in rec.items()}
    data["time_s"] = data["frame"] / fps
    if out_data:
        np.savez_compressed(out_data, fps=fps, frame_size=np.array([info.width, info.height]), **data)
    return {"frames_read": n, "frames_processed": len(rec["frame"]), "fps": fps, "roi": roi,
            "elapsed_s": time.perf_counter() - t0, "data": data}


def _record(rec: dict, frame_idx: int, players: Players, P: int) -> None:
    m = min(P, len(players.pose))
    valid = np.zeros(P, bool)
    valid[:m] = True
    bbox = np.full((P, 4), np.nan)
    kp = np.full((P, 17, 2), np.nan)
    sc = np.zeros((P, 17))
    xy = np.full((P, 2), np.nan)
    side = np.full(P, -1)
    bbox[:m] = players.pose.bboxes[:m]
    kp[:m] = players.pose.keypoints[:m]
    sc[:m] = players.pose.scores[:m]
    xy[:m] = players.court_xy[:m]
    side[:m] = players.side[:m]
    for k, v in zip(("frame", "valid", "bbox", "keypoints", "scores", "court_xy", "side"),
                    (frame_idx, valid, bbox, kp, sc, xy, side)):
        rec[k].append(v)


def print_progress(n: int, total: int, elapsed: float) -> None:
    if n % 30 and n != total:
        return
    rate = n / max(elapsed, 1e-6)
    eta = (total - n) / rate if total and rate else 0
    pct = f"{100 * n / total:5.1f}%" if total else ""
    sys.stderr.write(f"\r  {n}/{total or '?'} 帧 {pct}  {rate:5.1f} 帧/秒  剩余约 {eta / 60:4.1f} 分钟 ")
    sys.stderr.flush()
    if n == total:
        sys.stderr.write("\n")
