"""Pictures of the strokes the result page comments on (逐拍分析).

For every stroke in technique.json that got a note (e.g. 击球点偏低), cut the player out of the
video around the hit:
  * <stem>.shot.<frame>.jpg: three frames side by side, backswing / hit / follow-through;
  * <stem>.shot.<frame>.mp4: the stroke as a short silent clip (the page plays it slowed down).
<frame> is the hit's frame in the video, the same number as the stroke's "frame" in technique.json,
so the page finds the pictures without an index file.

Frames are read with OpenCV, like the skeleton, so frame numbers and pixel coordinates (the player's
box, written by `badminton technique`) match the analysis.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

STILL_OFFSETS_S = (-0.3, 0.0, 0.2)  # backswing, hit, follow-through
CLIP_S = (-0.8, 0.5)                # the clip, from the hit
PANEL_HEIGHT = 360                  # pixels of each picture on the page (shown at about half on a phone)
CLIP_HEIGHT = 480
GAP = 4
MAX_SHOTS = 150                     # a long session: enough for the page, keeps the upload small


def crop_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    """The part of the frame to keep around a player's joints: the racket reaches about half a body
    above and beside the joints. Even sizes (H.264) and inside the frame."""
    x0, y0, x1, y1 = (float(v) for v in box)
    bh = max(y1 - y0, 20.0)
    top, bottom = y0 - 0.45 * bh, y1 + 0.1 * bh
    ch = bottom - top
    cw = max((x1 - x0) + 0.9 * bh, 0.8 * ch)
    cx = (x0 + x1) / 2
    ch, cw = min(ch, height), min(cw, width)
    left = min(max(cx - cw / 2, 0), width - cw)
    top = min(max(top, 0), height - ch)
    w, h = int(cw) // 2 * 2, int(ch) // 2 * 2
    return int(left), int(top), max(w, 2), max(h, 2)


def _scaled(img: np.ndarray, height: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    if h == height:
        return img
    nw = max(2, int(round(w * height / h)) // 2 * 2)
    return cv2.resize(img, (nw, height), interpolation=cv2.INTER_AREA if h > height else cv2.INTER_LINEAR)


def picked(doc: dict) -> list[dict]:
    """The strokes the page shows in 逐拍分析 (ok, with a note) that have a player box."""
    rows = [r for r in doc.get("strokes", []) if r.get("ok") and r.get("notes") and r.get("box")]
    return sorted(rows, key=lambda r: int(r["frame"]))[:MAX_SHOTS]


def needs_boxes(doc: dict) -> bool:
    """technique.json from before the boxes were written: strokes with notes but no box."""
    return any(r.get("ok") and r.get("notes") and not r.get("box") for r in doc.get("strokes", []))


def make_shots(video: str | Path, technique_json: str | Path, out_dir: str | Path, clips: bool = True) -> list[Path]:
    """Write the pictures (and clips) into out_dir/shots/ (emptied first) and return their paths."""
    import cv2

    from .video import VideoWriter

    video, out_dir = Path(video), Path(out_dir)
    doc = json.loads(Path(technique_json).read_text())
    rows = picked(doc)
    shots = out_dir / "shots"
    if shots.exists():
        shutil.rmtree(shots)
    if not rows:
        return []
    shots.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = float(doc.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1 << 31
    out: list[Path] = []
    pos = 0  # the frame cap.read() returns next
    try:
        for r in rows:
            hit = int(r["frame"])
            first = max(0, hit + round(CLIP_S[0] * fps))
            last = min(n_frames - 1, hit + round(CLIP_S[1] * fps))
            if first < pos or first - pos > 2 * fps:  # far ahead or behind: seek; close by: read on
                cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                pos = first
            x, y, w, h = crop_box(r["box"], width, height)
            frames: dict[int, np.ndarray] = {}
            while pos <= last:
                ok, img = cap.read()
                if not ok:
                    break
                if pos >= first:
                    frames[pos] = img[y:y + h, x:x + w]
                pos += 1
            if hit not in frames:
                continue
            stem = f"{video.stem}.shot.{hit}"
            panels = []
            for off in STILL_OFFSETS_S:
                f = min(frames, key=lambda k: abs(k - (hit + round(off * fps))))
                panels.append(_scaled(frames[f], PANEL_HEIGHT))
            gap = np.full((PANEL_HEIGHT, GAP, 3), 255, np.uint8)
            strip = np.hstack([p if i == 0 else np.hstack([gap, p]) for i, p in enumerate(panels)])
            still = shots / f"{stem}.jpg"
            cv2.imwrite(str(still), strip, [cv2.IMWRITE_JPEG_QUALITY, 82])
            out.append(still)
            if clips:
                clip = shots / f"{stem}.mp4"
                first_img = _scaled(next(iter(frames.values())), min(CLIP_HEIGHT, h // 2 * 2))
                ch, cw = first_img.shape[:2]
                try:
                    with VideoWriter(clip, cw, ch, fps, crf=26) as wr:
                        for k in sorted(frames):
                            wr.write(_scaled(frames[k], ch))
                    out.append(clip)
                except (RuntimeError, ValueError, OSError):
                    clip.unlink(missing_ok=True)
    finally:
        cap.release()
    return out
