"""Shuttle tracking with TrackNetV3, run as native Core ML (device "coremlnative").

TrackNetV3 (Chen & Wang, MIT licence, https://github.com/qaz812345/TrackNetV3)
finds the shuttle in every frame:

  * TrackNet: 8 consecutive frames plus a median "background" image, all
    resized to 512x288, give 8 heatmaps; overlapping windows (step 1) are
    averaged with weights 1,2,3,4,4,3,2,1 and the largest blob above 0.5 is
    the shuttle.
  * InpaintNet: fills short gaps in the trajectory from 16-frame windows of
    coordinates, again averaged over overlapping windows.

This module reproduces the reference predict.py (eval_mode "weight") step by
step, with three differences that only change speed:
  * every frame is decoded and resized once (the reference re-resizes each
    frame in all 8 windows that contain it);
  * the median background comes from up to MEDIAN_SAMPLES frames resized to
    512x288 (the reference takes up to 1800 full-size frames, ~11 GB for
    1080p; the median of resized frames is the same picture up to rounding);
  * both networks run as fp16 Core ML models. Compute units default to ALL:
    unlike the pose models, TrackNet's 512x288 U-Net runs faster on the
    Neural Engine (M5: 25 ms per window vs 40 ms on the GPU, 63 ms in
    PyTorch on the GPU).

The TrackNetV3 code and weights are NOT part of this package. They are needed
once, to convert the models: clone the repository, put its checkpoints in
ckpts/ (see its README), and pass the folder with --tracknet-dir or the
TRACKNETV3_DIR environment variable. The converted models are cached in
~/.cache/badminton/ and are derived from the MIT-licensed weights.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .coreml_native import CACHE_DIR

HEIGHT, WIDTH = 288, 512
SEQ_LEN = 8  # TrackNet window
INPAINT_LEN = 16  # InpaintNet window
MEDIAN_SAMPLES = 300
SEQUENTIAL_MAX = 20000  # frames; longer videos are sampled by seeking (~5.5 min at 60 fps)
COOR_TH = 50 / math.sqrt(HEIGHT**2 + WIDTH**2)  # normalised coordinates below this count as "no shuttle"
INPAINT_BATCH = 64
TRACKNET_BATCH = 2  # windows per Core ML call: M5, ALL: 25 ms/window at batch 1, 11 ms at 2 (no gain at 4)

# Which TrackNet `badminton shuttle` uses by default. "released": converted from TrackNetV3's own
# checkpoint (needs --tracknet-dir once). Any other key: a fine-tuned model published as a GitHub
# Release asset (a zipped .mlpackage), downloaded once into CACHE_DIR/<key>/ and checked against
# its sha256 (docs/tracknet_finetune.md). BADMINTON_TRACKNET=<key> picks another one; --model-dir
# still overrides both. InpaintNet is always the released one.
TRACKNET_DEFAULT = "tracknet-ft-v1"
TRACKNET_RELEASES = {
    "tracknet-ft-v1": {  # IMG_0971 (60 fps 4K phone video, doubles), 1247 labelled frames, epoch 8
        "file": "tracknetv3_tracknet_fp16_b2.mlpackage.zip",
        "sha256": "0822dc0b7ae0a7cd91e9153e99582063913acf40e982ec0bcc2d51fb2ebae548",
        "size": 21057799,
    },
}
RELEASE_URL = "https://github.com/likefudan/badminton_analysis/releases/download/{tag}/{file}"


def ensemble_weight(seq_len: int) -> np.ndarray:
    """1, 2, ..., n/2, n/2, ..., 2, 1 normalised (reference get_ensemble_weight 'weight')."""
    w = np.ones(seq_len)
    for i in range(math.ceil(seq_len / 2)):
        w[i] = i + 1
        w[seq_len - i - 1] = i + 1
    return w / w.sum()


def largest_blob_center(heatmap: np.ndarray, thr: float = 0.5) -> tuple[int, int] | None:
    """Centre of the bounding box with the largest area among blobs above `thr`
    (reference predict_location; first one wins a tie). None if nothing."""
    binary = (heatmap > thr).astype(np.uint8) * 255
    if not binary.any():
        return None
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in cnts]
    x, y, w, h = max(rects, key=lambda r: r[2] * r[3])  # max() keeps the first maximum
    return int(x + w / 2), int(y + h / 2)


def blob_candidates(heatmap: np.ndarray, thr: float = 0.3) -> np.ndarray:
    """All blobs above `thr` in one (merged) heatmap: rows of
    (x, y, peak, area, bbox_area) in heatmap pixels, x/y = bbox centre like the
    reference, largest bbox first. Used to look beyond the top-1 point."""
    binary = (heatmap > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        out.append((x + w / 2, y + h / 2, float(heatmap[lab == k].max()), float(area), float(w * h)))
    out.sort(key=lambda r: -r[4])
    return np.array(out, dtype=np.float32).reshape(-1, 5)


def inpaint_mask(y: np.ndarray, vis: np.ndarray, th_h: float) -> np.ndarray:
    """Frames InpaintNet should fill: gaps with the shuttle seen (below `th_h`
    from the top) on both sides; a gap after the shuttle left the top of the
    picture is left alone. Port of the reference generate_inpaint_mask."""
    n = len(vis)
    mask = np.zeros(n)
    i = j = 0
    while j < n:
        while i < n - 1 and vis[i] == 1:
            i += 1
        j = i
        while j < n - 1 and vis[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > th_h:
            mask[:j] = 1
        elif (i > 1 and y[i - 1] > th_h) and (j < n and y[j] > th_h):
            mask[i:j] = 1
        i = j
    return mask


class _WindowEnsemble:
    """Averages per-window predictions of a sliding window (step 1) exactly as the
    reference does: weighted where all `n` windows cover a frame, a plain mean
    over the available windows for the first and last n-1 frames."""

    def __init__(self, n: int):
        self.n = n
        self.w = ensemble_weight(n)
        self.buf: deque = deque(maxlen=n)  # last n window outputs, oldest first
        self.count = 0  # windows seen

    def push(self, out: np.ndarray) -> np.ndarray:
        """Add window k's output (n, ...); return the merged value for frame k."""
        self.buf.append(out)
        self.count += 1
        k = len(self.buf)
        # frame index of the newest window's first frame sits at position 0 of
        # the newest window, position 1 of the one before, ...
        parts = [self.buf[k - 1 - j][j] for j in range(k)]  # newest first
        if self.count < self.n:
            return sum(parts) / self.count
        return sum(p * self.w[self.n - 1 - j] for j, p in enumerate(parts))

    def flush(self) -> list[np.ndarray]:
        """Merged values for the last n-1 frames once the input has ended."""
        res = []
        k = len(self.buf)
        for f in range(1, self.n):
            parts = [self.buf[k - 1 - j][f + j] for j in range(k) if f + j < self.n]
            res.append(sum(parts) / (self.n - f))  # reference divides by n-f (missing windows count as 0)
        return res


class _StepEnsemble:
    """Sliding-window average for windows `step` frames apart (step > 1 runs the
    model on fewer windows: each frame is covered by n/step of them instead of n).
    A frame's value is the weighted mean over the windows that contain it, with the
    same position weights as the reference. With step 1 this differs from the
    reference only in the first and last n-1 frames, so _WindowEnsemble stays in
    use there."""

    def __init__(self, n: int, step: int):
        self.n, self.step = n, step
        self.w = ensemble_weight(n)
        self.buf: deque = deque()  # (first frame, output) of the windows that still matter
        self.next = 0  # first frame not returned yet

    def push(self, start: int, out: np.ndarray) -> list[np.ndarray]:
        """Add the window that starts at frame `start`; return the frames that no later
        window (the next one starts at start + step) will touch."""
        self.buf.append((start, out))
        return self.take(start + self.step)

    def take(self, upto: int) -> list[np.ndarray]:
        """Merged values for frames self.next .. upto-1 (those covered by a window)."""
        res = []
        for f in range(self.next, upto):
            parts = [(o[f - s], self.w[f - s]) for s, o in self.buf if 0 <= f - s < self.n]
            if not parts:
                break
            res.append(sum(p * w for p, w in parts) / sum(w for _, w in parts))
        self.next += len(res)
        while self.buf and self.buf[0][0] + self.n <= self.next:
            self.buf.popleft()
        return res


# --- models ------------------------------------------------------------------------


def _import_tracknet(tracknet_dir: Path):
    spec = importlib.util.spec_from_file_location("tracknetv3_model", tracknet_dir / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_dir(tracknet_dir: str | os.PathLike | None) -> Path:
    d = tracknet_dir or os.environ.get("TRACKNETV3_DIR")
    if not d:
        raise SystemExit("第一次使用需要 TrackNetV3 的代码和权重来转换模型：用 --tracknet-dir 或环境变量 "
                         "TRACKNETV3_DIR 指向克隆好的 TrackNetV3 文件夹（ckpts/ 里有 TrackNet_best.pt 和 "
                         "InpaintNet_best.pt），见 README")
    d = Path(d).expanduser()
    for f in ("model.py", "ckpts/TrackNet_best.pt", "ckpts/InpaintNet_best.pt"):
        if not (d / f).exists():
            raise SystemExit(f"{d} 里没有 {f}")
    return d


def _load_checkpoint(path: Path) -> dict:
    """torch.load with weights_only (the safe default), plus the numpy scalar that
    TrackNetV3's train.py stores in a fine-tuned checkpoint (its best validation accuracy)."""
    import torch

    core = getattr(np, "_core", None) or np.core  # numpy 2 / numpy 1
    with torch.serialization.safe_globals([core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]):
        return torch.load(path, map_location="cpu")


def convert_models(tracknet_dir: str | os.PathLike | None, out_dir: Path = CACHE_DIR) -> tuple[Path, Path]:
    """TrackNetV3 checkpoints -> fp16 Core ML packages (once; needs torch + coremltools)."""
    import coremltools as ct
    import torch

    d = _resolve_dir(tracknet_dir)
    mod = _import_tracknet(d)
    paths = []
    for name, shape_fn in (("TrackNet", None), ("InpaintNet", None)):
        ckpt = _load_checkpoint(d / "ckpts" / f"{name}_best.pt")
        p = ckpt["param_dict"]
        if name == "TrackNet":
            assert p["bg_mode"] == "concat" and p["seq_len"] == SEQ_LEN, p
            model = mod.TrackNet(in_dim=(SEQ_LEN + 1) * 3, out_dim=SEQ_LEN)
            example = (torch.zeros(TRACKNET_BATCH, (SEQ_LEN + 1) * 3, HEIGHT, WIDTH),)
            inputs = [ct.TensorType(name="frames", shape=example[0].shape)]
            outputs = [ct.TensorType(name="heatmaps")]
        else:
            assert p["seq_len"] == INPAINT_LEN, p
            model = mod.InpaintNet()
            example = (torch.zeros(INPAINT_BATCH, INPAINT_LEN, 2), torch.zeros(INPAINT_BATCH, INPAINT_LEN, 1))
            inputs = [ct.TensorType(name="coords", shape=example[0].shape),
                      ct.TensorType(name="mask", shape=example[1].shape)]
            outputs = [ct.TensorType(name="inpainted")]
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
        ml = ct.convert(traced, inputs=inputs, outputs=outputs, compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.macOS13)
        out = _model_path(name, out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ml.save(str(out))
        paths.append(out)
    return paths[0], paths[1]


def _model_path(name: str, out_dir: Path = CACHE_DIR) -> Path:
    batch = TRACKNET_BATCH if name == "TrackNet" else INPAINT_BATCH
    return out_dir / f"tracknetv3_{name.lower()}_fp16_b{batch}.mlpackage"


def tracknet_choice() -> str:
    """The TrackNet to use: BADMINTON_TRACKNET if set, else TRACKNET_DEFAULT."""
    key = os.environ.get("BADMINTON_TRACKNET", "").strip() or TRACKNET_DEFAULT
    if key != "released" and key not in TRACKNET_RELEASES:
        raise SystemExit(f"BADMINTON_TRACKNET={key}：没有这个模型，可选 released、{'、'.join(TRACKNET_RELEASES)}")
    return key


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def release_model(key: str, cache_dir: Path = CACHE_DIR, url: str = RELEASE_URL) -> Path:
    """Path of the fine-tuned TrackNet `key`, downloaded and unpacked on first use.

    The zip is checked against the sha256 in TRACKNET_RELEASES before anything is
    unpacked; a mismatch stops with an error and leaves nothing behind. Unpacking
    goes to a temporary folder that is renamed into place, so an interrupted
    download never leaves a half model in the cache."""
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    rel = TRACKNET_RELEASES[key]
    dest = cache_dir / key
    model = _model_path("TrackNet", dest)
    if model.exists():
        return model
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = url.format(tag=key, file=rel["file"])
    print(f"第一次使用 {key}：下载微调的 TrackNet（{rel['size'] / 1e6:.0f} MB）{src}")
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=f".{key}.") as tmp:
        zpath = Path(tmp) / rel["file"]
        with urllib.request.urlopen(src, timeout=60) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        got = _sha256(zpath)
        if got != rel["sha256"]:
            raise SystemExit(f"下载的 {rel['file']} 校验不对（sha256 {got}，应为 {rel['sha256']}），没有使用")
        unpacked = Path(tmp) / "unpacked"
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():  # only the one .mlpackage folder, nothing outside it
                if not name.startswith(model.name + "/") or ".." in Path(name).parts:
                    raise SystemExit(f"{rel['file']} 里有意外的文件 {name}，没有使用")
            z.extractall(unpacked)
        if not (unpacked / model.name).is_dir():
            raise SystemExit(f"{rel['file']} 里没有 {model.name}")
        if dest.exists():
            shutil.rmtree(dest)
        (unpacked).rename(dest)
    return model


def load_models(tracknet_dir=None, compute_units: str = "ALL", model_dir: str | os.PathLike | None = None):
    """The two Core ML models: from `model_dir` (a folder written by convert_models,
    e.g. a fine-tuned TrackNet: docs/tracknet_finetune.md), else the TrackNet chosen by
    tracknet_choice() (a published fine-tuned one, downloaded once, or the released one
    converted into the cache) with the released InpaintNet."""
    import coremltools as ct

    if model_dir:
        out_dir = Path(model_dir).expanduser()
        tn, ip = _model_path("TrackNet", out_dir), _model_path("InpaintNet", out_dir)
        if not (tn.exists() and ip.exists()):
            raise SystemExit(f"{out_dir} 里没有 {tn.name} 和 {ip.name}")
    else:
        tn, ip = _model_path("TrackNet"), _model_path("InpaintNet")
        if not (tn.exists() and ip.exists()):
            print("第一次使用：把 TrackNetV3 转换成 Core ML 模型（约半分钟）")
            convert_models(tracknet_dir)
        key = tracknet_choice()
        if key != "released":
            tn = release_model(key)
    cu = getattr(ct.ComputeUnit, compute_units)
    return ct.models.MLModel(str(tn), compute_units=cu), ct.models.MLModel(str(ip), compute_units=cu)


# --- pipeline ----------------------------------------------------------------------


def _resize(frame_bgr: np.ndarray, method: str = "pil", rect: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """BGR frame (or its part `rect` = x0, y0, x1, y1) -> RGB uint8 (288, 512).

    "pil": Pillow bicubic, exactly what the reference does (~10 ms for 1080p).
    "area": OpenCV area averaging (~1 ms) but positions drift: on the test clip
    95.9 % of frames within 5 px of the reference vs 99.4 % with "pil".

    A picture more than 4x the model's size (a full 4K frame) is first shrunk by a
    whole factor with area averaging: Pillow alone needs 44 ms for 4K, this 12 ms,
    and Pillow's bicubic averages over the same pixels anyway when it shrinks."""
    if rect is not None:
        x0, y0, x1, y1 = rect
        frame_bgr = frame_bgr[y0:y1, x0:x1]
    h, w = frame_bgr.shape[:2]
    k = min(w // (2 * WIDTH), h // (2 * HEIGHT))
    if k >= 2:
        frame_bgr = cv2.resize(frame_bgr, (w // k, h // k), interpolation=cv2.INTER_AREA)
    if method == "pil":
        from PIL import Image

        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        if rgb.shape[:2] == (HEIGHT, WIDTH):
            return rgb
        return np.asarray(Image.fromarray(rgb).resize((WIDTH, HEIGHT), Image.BICUBIC))
    small = cv2.resize(frame_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small[..., ::-1])


def _resize_views(frame_bgr: np.ndarray, method: str, views, pool=None) -> np.ndarray:
    """All views of one frame, (V, 288, 512, 3). Pillow and OpenCV release the GIL
    while they resize, so a thread pool runs the views side by side."""
    if len(views) == 1:
        return _resize(frame_bgr, method, views[0])[None]
    fn = lambda r: _resize(frame_bgr, method, r)  # noqa: E731
    return np.stack(list(pool.map(fn, views)) if pool else [fn(r) for r in views])


def tile_views(size: tuple[int, int], zoom: float, roi: tuple[float, float, float, float] | None = None,
               overlap: float = 0.125) -> list[tuple[int, int, int, int]]:
    """Views (x0, y0, x1, y1 in video pixels) for track_shuttle: the full frame first,
    then tiles `zoom` times finer (each frame_size / zoom) that cover `roi` (default
    the whole frame), neighbours overlapping by at least `overlap` of a tile.

    Why: the model sees 512x288. A full 4K frame is shrunk 7.5x, and a shuttle at the
    far end of the court (6 px in 4K at 0.5x) becomes less than one pixel. In a tile
    it keeps `zoom` times more pixels; the full frame stays in for the near shuttle,
    which a tile would show larger and more blurred than anything in the training data."""
    w, h = size
    views = [(0, 0, w, h)]
    if zoom <= 1:
        return views
    tw, th = int(round(w / zoom)), int(round(h / zoom))
    x0, y0, x1, y1 = roi if roi is not None else (0, 0, w, h)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)

    def starts(a: float, b: float, t: int, limit: int) -> list[int]:
        if b - a <= t:  # one tile is enough: centre it on the region
            return [int(min(max(0, (a + b - t) / 2), limit - t))]
        n = math.ceil((b - a - t) / (t * (1 - overlap))) + 1
        return [int(round(v)) for v in np.linspace(a, b - t, n)]

    for y in starts(y0, y1, th, h):
        for x in starts(x0, x1, tw, w):
            views.append((x, y, x + tw, y + th))
    return views


EDGE_MARGIN = 8  # model pixels: a blob this close to a tile's inner border may be cut off; the neighbour has it whole
MERGE_RADIUS = 5.0  # full-frame heatmap pixels: candidates of two views this close are the same object
# A tile's candidate must peak at least this high (the full frame's: 0.3, blob_candidates). Magnified, the
# floor and the walls are full of small bright specks that light up weakly. Checked by hand on 64 frames
# each of two 1080p phone videos, wrong picks with full frame only / tiles at 0.3 / at 0.7: 24 / 34 / 24
# and 14 / 21 / 15. But a far shuttle that only a tile can see is not always a strong blob: on a video
# shrunk to half size (so that the full frame loses the shuttle) correct picks were 24 without tiles and
# 28 / 30 / 29 / 28 / 25 with tiles at 0.3 / 0.4 / 0.5 / 0.6 / 0.7. Hence the middle.
TILE_MIN_PEAK = 0.6


def merge_view_candidates(per_view: list, views: list, size: tuple[int, int]) -> np.ndarray:
    """Candidates of all views of one frame (blob_candidates rows, in each view's own
    heatmap pixels) -> one array in full-frame heatmap pixels (fractional), best peak
    first. Blobs at a tile's inner border are dropped, so are a tile's weak blobs
    (TILE_MIN_PEAK), and duplicates of one object (seen by overlapping views) keep
    the highest peak."""
    w, h = size
    fx, fy = w / WIDTH, h / HEIGHT
    rows = []
    for c, (x0, y0, x1, y1) in zip(per_view, views):
        sx, sy = (x1 - x0) / WIDTH, (y1 - y0) / HEIGHT
        is_tile = (x0, y0, x1, y1) != (0, 0, w, h)
        for x, y, peak, area, bbox in c:
            if is_tile and peak < TILE_MIN_PEAK:
                continue
            if ((x < EDGE_MARGIN and x0 > 0) or (x > WIDTH - EDGE_MARGIN and x1 < w)
                    or (y < EDGE_MARGIN and y0 > 0) or (y > HEIGHT - EDGE_MARGIN and y1 < h)):
                continue
            a = (sx * sy) / (fx * fy)
            rows.append(((x0 + x * sx) / fx, (y0 + y * sy) / fy, peak, area * a, bbox * a))
    rows.sort(key=lambda r: -r[2])
    keep: list = []
    for r in rows:
        if all((r[0] - k[0]) ** 2 + (r[1] - k[1]) ** 2 >= MERGE_RADIUS**2 for k in keep):
            keep.append(r)
    return np.array(keep, dtype=np.float32).reshape(-1, 5)


def median_background(video: str | Path, samples: int = MEDIAN_SAMPLES, method: str = "pil", views=None) -> np.ndarray:
    """Median of up to `samples` evenly spread frames, resized to 512x288
    (with `views`: one background per view, (V, 288, 512, 3)).

    Seeking costs a decode from the previous keyframe (63 ms per seek on a clip
    with a keyframe every 8 s), so videos up to SEQUENTIAL_MAX frames are read
    straight through instead (grab() without converting the skipped frames)."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if views is not None:
        samples = max(60, min(samples, MEDIAN_SAMPLES * 4 // len(views)))  # 300 samples of 20 views would be 2.6 GB
    step = max(1, n // samples)
    imgs = []
    small = (lambda f: _resize(f, method)) if views is None else (lambda f: _resize_views(f, method, views))
    if n <= SEQUENTIAL_MAX:
        for i in range(n):
            if i % step:
                if not cap.grab():
                    break
                continue
            ok, f = cap.read()
            if not ok:
                break
            imgs.append(small(f))
    else:
        for i in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                break
            imgs.append(small(f))
    cap.release()
    return np.median(np.stack(imgs), axis=0).astype(np.uint8)


@dataclass
class ShuttleTiming:
    seconds: dict = field(default_factory=lambda: {"decode+resize（并行）": 0.0, "tracknet": 0.0, "ensemble": 0.0,
                                                   "locate": 0.0, "inpaint": 0.0})
    median_s: float = 0.0  # one-off: background image before the first frame
    wall_s: float = 0.0  # frame loop + inpaint, wall clock
    frames: int = 0
    measured: np.ndarray | None = None  # per returned row: TrackNet itself saw the shuttle in that frame

    def summary(self) -> str:
        parts = "，".join(f"{k} {1000 * v / max(1, self.frames):.1f}" for k, v in self.seconds.items())
        return (f"{self.frames} 帧：逐帧部分 {self.wall_s:.1f} 秒（{self.frames / max(self.wall_s, 1e-9):.1f} 帧/秒），"
                f"另加背景图 {self.median_s:.1f} 秒；每帧 ms：{parts}")


def _reader(video, max_frames, method, q, busy, step=1, start=0, views=None):
    """Decode + resize in a separate process. Core ML's predict() keeps Python's
    GIL while it runs, so a thread could not overlap with it; a process can.
    Frames travel as uint8 (V views x 440 kB each); `busy` accumulates seconds spent.
    Only every `step`-th frame is sent. An exception is sent as ("error", text)
    so the caller fails instead of silently getting a shorter result."""
    from concurrent.futures import ThreadPoolExecutor

    cap = cv2.VideoCapture(str(video))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = 0
    pool = ThreadPoolExecutor(min(len(views), 6)) if views is not None and len(views) > 1 else None
    try:
        while max_frames is None or i < max_frames:
            t0 = time.perf_counter()
            if i % step:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, f = cap.read()
            if not ok:
                break
            if views is None:
                views = [(0, 0, f.shape[1], f.shape[0])]
            small = _resize_views(f, method, views, pool)
            with busy.get_lock():
                busy.value += time.perf_counter() - t0
            q.put(small)
            i += 1
    except Exception as e:  # noqa: BLE001 - reported to the parent
        q.put(("error", f"{type(e).__name__}: {e}"))
    finally:
        cap.release()
        if pool:
            pool.shutdown()
        q.put(None)


def track_shuttle(video: str | Path, tracknet_dir=None, compute_units: str = "ALL", inpaint: bool = True,
                  max_frames: int | None = None, progress=None, resize: str = "pil", frame_step: int = 1,
                  start_frame: int = 0, debug: dict | None = None,
                  model_dir: str | os.PathLike | None = None, views: list | None = None, window_step: int = 1):
    """Returns (rows, timing); rows = list of (frame, visibility, x, y) in video pixels, one per video frame.

    start_frame: begin there (row frame numbers stay video frame numbers).
    debug: if a dict is given, it receives per processed frame the blob candidates
    of the merged heatmap ("candidates") and, for processed frames listed in
    debug["keep_heatmaps"] (a set), the merged heatmaps ("heatmaps", fp16).

    frame_step=2 feeds every second frame to the models: TrackNetV3 was trained
    on 30 fps footage, so a 60 fps video looks like it does in training (the
    shuttle moves twice as far between the frames the model sees). The skipped
    frames are filled by linear interpolation between their neighbours when
    both have the shuttle.

    views: parts of the frame (x0, y0, x1, y1 in video pixels, see tile_views) that
    each go through TrackNet at 512x288; the first must be the full frame. The rows
    still come from the full frame alone (the reference method); the candidates in
    `debug` are those of all views together, in full-frame heatmap pixels, for
    shuttle_path.rows_from_candidates.

    window_step: run TrackNet on every N-th window only (1 = like the reference,
    8 windows per frame; 4 = 2 windows per frame, 4x less model time)."""
    import multiprocessing as mp
    import queue

    if not 1 <= window_step <= SEQ_LEN:
        raise ValueError(f"window_step must be 1..{SEQ_LEN}")
    tm = ShuttleTiming()
    tracknet, inpaintnet = load_models(tracknet_dir, compute_units, model_dir)
    cap = cv2.VideoCapture(str(video))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()  # size of the decoded frame: a phone video's rotation flag is already applied
    cap.release()
    if not ok:
        raise RuntimeError(f"读不出视频画面：{video}")
    h, w = first.shape[:2]
    sx, sy = w / WIDTH, h / HEIGHT
    views = [tuple(int(v) for v in r) for r in views] if views else [(0, 0, w, h)]
    if views[0] != (0, 0, w, h):
        raise ValueError("the first view must be the full frame")
    V = len(views)

    def chw(img):
        # CHW, 0..1 as the model expects. Must be C-contiguous: astype() keeps the
        # transposed layout, and Core ML converts a non-contiguous input 2.4x slower.
        return np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255

    t = time.perf_counter()
    medians = [chw(m) for m in median_background(video, method=resize, views=views)]
    tm.median_s = time.perf_counter() - t

    frames = [deque(maxlen=SEQ_LEN) for _ in range(V)]
    exact = window_step == 1  # the reference's ensemble, bit for bit
    ens = [_WindowEnsemble(SEQ_LEN) if exact else _StepEnsemble(SEQ_LEN, window_step) for _ in range(V)]
    ready = [deque() for _ in range(V)]  # merged heatmaps per view, waiting for the other views
    xs, ys, vis = [], [], []

    if debug is not None:
        debug.setdefault("candidates", [])
        debug.setdefault("heatmaps", {})
        debug["views"] = views
        keep_hm = debug.get("keep_heatmaps", set())

    def locate(heats):
        """One frame: `heats` = the merged heatmap of every view."""
        heat = heats[0]
        if debug is not None:
            k = len(xs)
            cands = [blob_candidates(hm) for hm in heats]
            debug["candidates"].append(cands[0] if V == 1 else merge_view_candidates(cands, views, (w, h)))
            if k in keep_hm:
                debug["heatmaps"][k] = heat.astype(np.float16)
        t0 = time.perf_counter()
        c = largest_blob_center(heat)
        if c is None:
            xs.append(0), ys.append(0), vis.append(0)
        else:
            x, y = int(c[0] * sx), int(c[1] * sy)
            xs.append(x), ys.append(y), vis.append(0 if x == 0 and y == 0 else 1)
        tm.seconds["locate"] += time.perf_counter() - t0

    def emit():
        while all(ready):
            locate([r.popleft() for r in ready])

    pending: list = []  # (view, first frame of the window, window)

    def run_window(start: int | None, flush: bool = False):
        """Queue the 8-frame window of every view; run the model on TRACKNET_BATCH at a time."""
        if start is not None:
            pending.extend((v, start, np.concatenate([medians[v], *frames[v]])) for v in range(V))
        while len(pending) >= TRACKNET_BATCH or (flush and pending):
            chunk = pending[:TRACKNET_BATCH]
            del pending[:TRACKNET_BATCH]
            t0 = time.perf_counter()
            k = len(chunk)
            batch = np.ascontiguousarray(np.stack([c[2] for c in chunk] + [chunk[-1][2]] * (TRACKNET_BATCH - k)))
            heat = tracknet.predict({"frames": batch})["heatmaps"][:k]
            tm.seconds["tracknet"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            for (v, s, _), hm in zip(chunk, heat):
                hm = hm.astype(np.float32)
                ready[v].extend([ens[v].push(hm)] if exact else ens[v].push(s, hm))
            tm.seconds["ensemble"] += time.perf_counter() - t0
            emit()

    wall = time.perf_counter()
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=max(4, 32 // V))
    busy = ctx.Value("d", 0.0)
    reader = ctx.Process(target=_reader, args=(str(video), max_frames, resize, q, busy, frame_step, start_frame, views),
                         daemon=True)
    reader.start()
    n = 0
    last_start = -1
    while True:
        try:
            item = q.get(timeout=10)
        except queue.Empty:
            if not reader.is_alive():
                raise RuntimeError(f"读视频的子进程意外退出（exit code {reader.exitcode}）：{video}") from None
            continue
        if item is None:
            break
        if isinstance(item, tuple):
            reader.join()
            raise RuntimeError(f"读视频时出错（第 {n * frame_step} 帧附近）：{item[1]}")
        for v in range(V):
            frames[v].append(chw(item[v]))
        n += 1
        if n >= SEQ_LEN and (n - SEQ_LEN) % window_step == 0:
            last_start = n - SEQ_LEN
            run_window(last_start)
        if progress:
            progress(n, (max_frames or n_total) // frame_step, time.perf_counter() - wall)
    reader.join()
    tm.seconds["decode+resize（并行）"] = busy.value
    if 0 < n < SEQ_LEN:  # shorter than one window: pad with the last frame, like the reference
        for v in range(V):
            while len(frames[v]) < SEQ_LEN:
                frames[v].append(frames[v][-1])
        run_window(0)
    elif n and last_start != n - SEQ_LEN:  # window_step > 1: the last frames need a window of their own
        run_window(n - SEQ_LEN)
    run_window(None, flush=True)
    if n:
        for v in range(V):
            if exact:
                ready[v].extend(ens[v].flush()[: n - 1] if n < SEQ_LEN else ens[v].flush())
            else:
                ready[v].extend(ens[v].take(n))
        emit()
    tm.frames = n
    if debug is not None:
        del debug["candidates"][n:]

    xs, ys, vis = np.array(xs[:n], float), np.array(ys[:n], float), np.array(vis[:n], int)
    seen = vis.astype(bool)  # before InpaintNet fills gaps: what TrackNet itself saw
    if inpaint and n >= INPAINT_LEN:
        t0 = time.perf_counter()
        xs, ys, vis = _inpaint(inpaintnet, xs, ys, vis, w, h)
        tm.seconds["inpaint"] += time.perf_counter() - t0
    tm.wall_s = time.perf_counter() - wall
    if frame_step > 1:
        rows = expand_rows(xs, ys, vis, frame_step, max_frames or (n_total - start_frame))
        tm.measured = np.array([f % frame_step == 0 and bool(seen[f // frame_step]) for f, *_ in rows], bool)
    else:
        rows = [(i, int(vis[i]), int(xs[i]), int(ys[i])) for i in range(n)]
        tm.measured = seen[:n].copy()
    return [(f + start_frame, v, x, y) for f, v, x, y in rows], tm


def row_sources(rows, measured) -> list[str]:
    """Source tag per row (see provenance.py): "measured" where the model saw the
    shuttle in that frame, "interpolated" where the position was filled in
    (InpaintNet, skipped frames, short gaps of the chosen path), "" where there is none."""
    from .provenance import INTERPOLATED, MEASURED

    return [(MEASURED if m else INTERPOLATED) if r[1] else "" for r, m in zip(rows, measured)]


def expand_rows(xs, ys, vis, step: int, n_frames: int) -> list:
    """Results for every `step`-th frame -> one row per video frame; skipped frames
    interpolated when the shuttle is seen on both sides, otherwise not seen."""
    rows = []
    last = (len(xs) - 1) * step
    for f in range(min(n_frames, last + 1)):
        k, r = divmod(f, step)
        if r == 0:
            rows.append((f, int(vis[k]), int(xs[k]), int(ys[k])))
        elif vis[k] and vis[k + 1]:
            a = r / step
            rows.append((f, 1, int(round((1 - a) * xs[k] + a * xs[k + 1])), int(round((1 - a) * ys[k] + a * ys[k + 1]))))
        else:
            rows.append((f, 0, 0, 0))
    return rows


def _inpaint(model, xs, ys, vis, w, h):
    n = len(xs)
    mask = inpaint_mask(ys, vis, th_h=h * 0.05)
    coor = np.stack([xs / w, ys / h], 1).astype(np.float32)
    windows = n - INPAINT_LEN + 1
    ens = _WindowEnsemble(INPAINT_LEN)
    merged = []
    for b0 in range(0, windows, INPAINT_BATCH):
        idx = np.arange(b0, min(windows, b0 + INPAINT_BATCH))
        c = np.zeros((INPAINT_BATCH, INPAINT_LEN, 2), np.float32)
        m = np.zeros((INPAINT_BATCH, INPAINT_LEN, 1), np.float32)
        for k, s in enumerate(idx):
            c[k] = coor[s:s + INPAINT_LEN]
            m[k, :, 0] = mask[s:s + INPAINT_LEN]
        out = model.predict({"coords": c, "mask": m})["inpainted"]
        out = out * m + c * (1 - m)
        out[(out[..., 0] < COOR_TH) & (out[..., 1] < COOR_TH)] = 0
        for k in range(len(idx)):
            merged.append(ens.push(out[k]))
    merged += ens.flush()
    merged = np.array(merged)
    merged[(merged[:, 0] < COOR_TH) & (merged[:, 1] < COOR_TH)] = 0
    x = (merged[:, 0] * w).astype(int)
    y = (merged[:, 1] * h).astype(int)
    v = ((x != 0) | (y != 0)).astype(int)
    return x, y, v


def write_csv(rows, path: str | Path, sources: list[str] | None = None) -> None:
    """Frame, Visibility, X, Y and, when `sources` is given (row_sources), a fifth
    column Source saying whether the position was measured or interpolated."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        if sources is None:
            wr.writerow(["Frame", "Visibility", "X", "Y"])
            wr.writerows(rows)
        else:
            wr.writerow(["Frame", "Visibility", "X", "Y", "Source"])
            wr.writerows([*r[:4], s] for r, s in zip(rows, sources))
