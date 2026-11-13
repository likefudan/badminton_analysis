"""Shuttle labelling on the phone: the frames and predictions the web page needs.

`badminton annotate-pack VIDEO -o DIR` picks the rallies TrackNetV3 is least sure
about (the same ranking as `badminton annotate`), cuts their frames into JPEGs
and writes them to DIR:

  rally-<k>.bin   the JPEGs of rally k one after another; state.json says where each starts
  state.json      video size and fps, the chosen rallies, and per frame the prediction
                  (chosen path and top-1), the moving candidate blobs and the uncertainty

The node uploads DIR to the site; the page fetches a few frames at a time with
HTTP Range requests and saves labels in the site's database, which the node
then writes back to `<video>.labels.csv` (see node.py), so `annotate-export`
works as after a desktop session.

Coordinates in state.json are video pixels, as in the label CSV. The JPEGs are
scaled down to `width` pixels; the page scales back.

When DIR already holds a state.json, the same rallies are packed again (only the
predictions change): labels made earlier then still fall on the new frames, and
the accuracy before and after fine-tuning is measured on the same frames.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import __version__
from . import annotate as an

PACK_VERSION = 1
WIDTH = 1280          # JPEG width on the phone; the shuttle is 3-6 px wide here in a 4K court video
QUALITY = 85
RALLIES = 3           # how many rallies to offer
MAX_RALLY_S = 20.0    # longer rallies only when there are not enough short ones (a phone labels ~1 frame/s)
MAX_FRAMES = 3600     # all chosen rallies together
PREDICT_VERSION = "2"  # raise when the path selection changes, so cached predictions are redone


def model_stamp(tracknet_dir=None) -> str:
    """Which TrackNetV3 weights made a prediction: size and mtime of the checkpoint, hashed.
    A fine-tuned checkpoint gives a new stamp, so cached predictions are redone."""
    try:
        from .shuttle import _resolve_dir

        ckpt = _resolve_dir(tracknet_dir) / "ckpts" / "TrackNet_best.pt"
        st = ckpt.stat()
        return hashlib.sha1(f"{PREDICT_VERSION}|{ckpt.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]
    except (SystemExit, OSError, ImportError):
        return "v" + PREDICT_VERSION


def load_or_predict(video: Path, court: Path | None, fresh: bool = False, tracknet_dir=None, progress=None,
                    predict=an.predict):
    """(path, raw, candidates, stamp, predicted_at): the cached prediction of `badminton annotate`
    (<video>.annotate.npz) unless asked for a fresh one or the model changed since."""
    cache = video.with_suffix(".annotate.npz")
    stamp_file = video.with_suffix(".annotate.model.json")
    stamp = model_stamp(tracknet_dir)
    saved = json.loads(stamp_file.read_text()) if stamp_file.exists() else {}
    if cache.exists() and not fresh and saved.get("model") == stamp:
        path, raw, cands = an.load_predictions(cache)
        print(f"沿用缓存的预测 {cache.name}")
        return path, raw, cands, saved.get("model", stamp), saved.get("predicted_at", "")
    print("用 TrackNetV3 逐帧找球（结果缓存下来，下次直接用）")
    path, raw, cands = predict(video, court, tracknet_dir, progress)
    an.save_predictions(cache, path, raw, cands)
    when = an._now()
    stamp_file.write_text(json.dumps({"model": stamp, "predicted_at": when}))
    return path, raw, cands, stamp, when


def choose_rallies(rallies: list, uncertainty: np.ndarray, fps: float, count: int = RALLIES,
                   max_s: float = MAX_RALLY_S, max_frames: int = MAX_FRAMES) -> list[int]:
    """Indices of the rallies to label: least sure first, short ones preferred, within the frame budget."""
    order = an.rank_rallies(rallies, uncertainty)
    length = {i: rallies[i][1] - rallies[i][0] + 1 for i in order}
    short = [i for i in order if length[i] <= max_s * fps]
    long_ = sorted((i for i in order if length[i] > max_s * fps), key=lambda i: length[i])
    out, total = [], 0
    for i in short + long_:
        if len(out) == count:
            break
        if out and total + length[i] > max_frames:
            continue
        out.append(i)
        total += length[i]
    return out


def write_rally(video: Path, start: int, end: int, out: Path, width: int = WIDTH, quality: int = QUALITY) -> list:
    """JPEGs of frames start..end appended to `out`; [(offset, length)] per frame."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    offsets = []
    with open(out, "wb") as f:
        for _ in range(start, end + 1):
            ok, img = cap.read()
            if not ok:
                break
            h, w = img.shape[:2]
            if w > width:
                img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            data = buf.tobytes()
            offsets.append((f.tell(), len(data)))
            f.write(data)
    cap.release()
    return offsets


def pack(video: str | Path, out_dir: str | Path, court: str | Path | None = None, movement: str | Path | None = None,
         fresh: bool = False, count: int = RALLIES, tracknet_dir=None, progress=None, predict=an.predict,
         width: int = WIDTH) -> dict:
    from .video import probe

    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    fps = info.fps or 30.0
    size = (info.width, info.height)
    path, raw, cands, stamp, predicted_at = load_or_predict(video, Path(court) if court else None, fresh,
                                                            tracknet_dir, progress, predict)
    n = len(path)
    if movement and Path(movement).exists():
        rallies = an.load_rallies(movement, fps, n)
    else:
        rallies = an.rallies_from_predictions(path, fps) or [(0, n - 1)]
        print(f"没有回合列表，按球出现的时段切成 {len(rallies)} 段")
    raw_weight = None
    moving = None
    if cands is not None:
        moving, grid = an.moving_candidates(cands, size)
        if raw is not None:
            raw_weight = np.where(an.on_static(grid, raw[:, 1:3], size) & (raw[:, 0] > 0), an.RAW_ON_STATIC, 1.0)
    unc = an.frame_uncertainty(path, raw, moving, size[1], raw_weight)

    # the same frames as last time, so earlier labels and before/after comparisons still line up
    previous = out_dir / "state.json"
    chosen: list[tuple[int, int]] = []
    if previous.exists():
        try:
            chosen = [(int(r["start"]), int(r["end"])) for r in json.loads(previous.read_text())["rallies"]]
        except (ValueError, KeyError, TypeError):
            chosen = []
        chosen = [(a, b) for a, b in chosen if 0 <= a <= b < n]
    if not chosen:
        chosen = [rallies[i] for i in choose_rallies(rallies, unc, fps, count)]

    r1 = lambda v: round(float(v), 1)  # noqa: E731
    out_rallies = []
    image_size = None
    for k, (a, b) in enumerate(chosen):
        name = f"rally-{k}.bin"
        offsets = write_rally(video, a, b, out_dir / name, width)
        b = a + len(offsets) - 1  # the video may end early
        if image_size is None:
            image_size = [min(width, size[0]), round(size[1] * min(width, size[0]) / size[0])]
        frames = range(a, b + 1)
        out_rallies.append({
            "start": a, "end": b, "pack": name, "offsets": [list(o) for o in offsets],
            "uncertainty": round(float(unc[a:b + 1].mean()), 3) if b >= a else 0,
            "pred": [[int(path[f, 0] > 0), r1(path[f, 1]), r1(path[f, 2])] for f in frames],
            "raw": [[int(raw[f, 0] > 0), r1(raw[f, 1]), r1(raw[f, 2])] for f in frames] if raw is not None else None,
            "cand": [[[r1(x), r1(y), round(float(p), 2)] for x, y, p in np.asarray(moving[f])[:, :3]] for f in frames]
            if moving is not None else None,
            "unc": [round(float(unc[f]), 3) for f in frames],
        })
        print(f"回合 {k + 1}：第 {a}–{b} 帧，{len(offsets)} 张图，{(out_dir / name).stat().st_size / 1e6:.1f} MB")
    state = {
        "version": PACK_VERSION, "tool": "badminton annotate-pack", "tool_version": __version__,
        "video": {"file": video.name, "frames": n, "fps": fps, "size": list(size)},
        "image": image_size or [width, round(size[1] * width / max(size[0], 1))],
        "model": {"stamp": stamp, "predicted_at": predicted_at},
        "confident": an.CONFIDENT, "doubtful": an.DOUBTFUL,
        "all_rallies": [[int(a), int(b)] for a, b in rallies],
        "rallies": out_rallies, "packed_at": an._now(),
    }
    tmp = out_dir / "state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, separators=(",", ":")))
    os.replace(tmp, previous)
    return state


def write_labels_from_cloud(video: str | Path, state: dict, labels: dict, annotator: str = "网页") -> dict:
    """Write what the web page labelled into <video>.labels.csv (+ .json), the files `annotate-export` reads.
    The site is the truth for frames of the rallies offered on the page; labels of other frames
    (a desktop `badminton annotate` session) are kept. `labels`: {frame: [visibility, x, y, status]}."""
    video = Path(video)
    store = an.LabelStore(video.with_suffix(".labels.csv"),
                          {"file": state["video"]["file"], "frames": state["video"]["frames"],
                           "fps": state["video"]["fps"], "size": state["video"]["size"]},
                          annotator, state.get("all_rallies"))
    web = set()
    for r in state["rallies"]:
        web.update(range(int(r["start"]), int(r["end"]) + 1))
    before = {f: lab for f, lab in store.labels.items()}
    for f in list(store.labels):
        if f in web:
            del store.labels[f]
    for key, (v, x, y, status) in labels.items():
        f = int(key)
        if status in an.STATUSES and status != "machine":
            store.labels[f] = an.Label(int(bool(v)) if status != "skip" else 0, float(x) if v else 0.0,
                                       float(y) if v else 0.0, status)
    changed = sum(1 for f in set(before) | set(store.labels) if before.get(f) != store.labels.get(f))
    store.session["frames_changed"] = changed
    store.session["source"] = "ba.llmat.dev"
    store.save()
    return {"frames": len(store.labels), "changed": changed, "csv": str(store.csv_path), "at": time.time()}
