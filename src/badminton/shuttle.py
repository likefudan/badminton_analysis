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


def convert_models(tracknet_dir: str | os.PathLike | None, out_dir: Path = CACHE_DIR) -> tuple[Path, Path]:
    """TrackNetV3 checkpoints -> fp16 Core ML packages (once; needs torch + coremltools)."""
    import coremltools as ct
    import torch

    d = _resolve_dir(tracknet_dir)
    mod = _import_tracknet(d)
    paths = []
    for name, shape_fn in (("TrackNet", None), ("InpaintNet", None)):
        ckpt = torch.load(d / "ckpts" / f"{name}_best.pt", map_location="cpu")  # weights_only (safe) by default
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


def load_models(tracknet_dir=None, compute_units: str = "ALL"):
    import coremltools as ct

    tn, ip = _model_path("TrackNet"), _model_path("InpaintNet")
    if not (tn.exists() and ip.exists()):
        print("第一次使用：把 TrackNetV3 转换成 Core ML 模型（约半分钟）")
        convert_models(tracknet_dir)
    cu = getattr(ct.ComputeUnit, compute_units)
    return ct.models.MLModel(str(tn), compute_units=cu), ct.models.MLModel(str(ip), compute_units=cu)


# --- pipeline ----------------------------------------------------------------------


def _resize(frame_bgr: np.ndarray, method: str = "pil") -> np.ndarray:
    """BGR frame -> RGB uint8 (288, 512).

    "pil": Pillow bicubic, exactly what the reference does (~10 ms for 1080p).
    "area": OpenCV area averaging (~1 ms) but positions drift: on the test clip
    95.9 % of frames within 5 px of the reference vs 99.4 % with "pil"."""
    if method == "pil":
        from PIL import Image

        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        return np.asarray(Image.fromarray(rgb).resize((WIDTH, HEIGHT), Image.BICUBIC))
    small = cv2.resize(frame_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small[..., ::-1])


def median_background(video: str | Path, samples: int = MEDIAN_SAMPLES, method: str = "pil") -> np.ndarray:
    """Median of up to `samples` evenly spread frames, resized to 512x288.

    Seeking costs a decode from the previous keyframe (63 ms per seek on a clip
    with a keyframe every 8 s), so videos up to SEQUENTIAL_MAX frames are read
    straight through instead (grab() without converting the skipped frames)."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, n // samples)
    imgs = []
    if n <= SEQUENTIAL_MAX:
        for i in range(n):
            if i % step:
                if not cap.grab():
                    break
                continue
            ok, f = cap.read()
            if not ok:
                break
            imgs.append(_resize(f, method))
    else:
        for i in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                break
            imgs.append(_resize(f, method))
    cap.release()
    return np.median(np.stack(imgs), axis=0).astype(np.uint8)


@dataclass
class ShuttleTiming:
    seconds: dict = field(default_factory=lambda: {"decode+resize（并行）": 0.0, "tracknet": 0.0, "ensemble": 0.0,
                                                   "locate": 0.0, "inpaint": 0.0})
    median_s: float = 0.0  # one-off: background image before the first frame
    wall_s: float = 0.0  # frame loop + inpaint, wall clock
    frames: int = 0

    def summary(self) -> str:
        parts = "，".join(f"{k} {1000 * v / max(1, self.frames):.1f}" for k, v in self.seconds.items())
        return (f"{self.frames} 帧：逐帧部分 {self.wall_s:.1f} 秒（{self.frames / max(self.wall_s, 1e-9):.1f} 帧/秒），"
                f"另加背景图 {self.median_s:.1f} 秒；每帧 ms：{parts}")


def _reader(video, max_frames, method, q, busy, step=1):
    """Decode + resize in a separate process. Core ML's predict() keeps Python's
    GIL while it runs, so a thread could not overlap with it; a process can.
    Frames travel as uint8 (440 kB each); `busy` accumulates seconds spent.
    Only every `step`-th frame is sent. An exception is sent as ("error", text)
    so the caller fails instead of silently getting a shorter result."""
    cap = cv2.VideoCapture(str(video))
    i = 0
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
            small = _resize(f, method)
            with busy.get_lock():
                busy.value += time.perf_counter() - t0
            q.put(small)
            i += 1
    except Exception as e:  # noqa: BLE001 - reported to the parent
        q.put(("error", f"{type(e).__name__}: {e}"))
    finally:
        cap.release()
        q.put(None)


def track_shuttle(video: str | Path, tracknet_dir=None, compute_units: str = "ALL", inpaint: bool = True,
                  max_frames: int | None = None, progress=None, resize: str = "pil", frame_step: int = 1):
    """Returns (rows, timing); rows = list of (frame, visibility, x, y) in video pixels, one per video frame.

    frame_step=2 feeds every second frame to the models: TrackNetV3 was trained
    on 30 fps footage, so a 60 fps video looks like it does in training (the
    shuttle moves twice as far between the frames the model sees). The skipped
    frames are filled by linear interpolation between their neighbours when
    both have the shuttle."""
    import multiprocessing as mp
    import queue

    tm = ShuttleTiming()
    tracknet, inpaintnet = load_models(tracknet_dir, compute_units)
    cap = cv2.VideoCapture(str(video))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()  # size of the decoded frame: a phone video's rotation flag is already applied
    cap.release()
    if not ok:
        raise RuntimeError(f"读不出视频画面：{video}")
    h, w = first.shape[:2]
    sx, sy = w / WIDTH, h / HEIGHT

    t = time.perf_counter()
    median = np.ascontiguousarray(median_background(video, method=resize).transpose(2, 0, 1)).astype(np.float32) / 255
    tm.median_s = time.perf_counter() - t

    frames: deque = deque(maxlen=SEQ_LEN)
    ens = _WindowEnsemble(SEQ_LEN)
    xs, ys, vis = [], [], []

    def locate(heat):
        t0 = time.perf_counter()
        c = largest_blob_center(heat)
        if c is None:
            xs.append(0), ys.append(0), vis.append(0)
        else:
            x, y = int(c[0] * sx), int(c[1] * sy)
            xs.append(x), ys.append(y), vis.append(0 if x == 0 and y == 0 else 1)
        tm.seconds["locate"] += time.perf_counter() - t0

    pending: list = []

    def run_window(flush: bool = False):
        """Queue the current 8-frame window; run the model once TRACKNET_BATCH are queued."""
        if not flush:
            pending.append(np.concatenate([median, *frames]))
        if not pending or (len(pending) < TRACKNET_BATCH and not flush):
            return
        t0 = time.perf_counter()
        k = len(pending)
        batch = np.ascontiguousarray(np.stack(pending + [pending[-1]] * (TRACKNET_BATCH - k)))
        heat = tracknet.predict({"frames": batch})["heatmaps"][:k]
        pending.clear()
        tm.seconds["tracknet"] += time.perf_counter() - t0
        for hm in heat:
            t0 = time.perf_counter()
            merged = ens.push(hm.astype(np.float32))
            tm.seconds["ensemble"] += time.perf_counter() - t0
            locate(merged)

    wall = time.perf_counter()
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=32)
    busy = ctx.Value("d", 0.0)
    reader = ctx.Process(target=_reader, args=(str(video), max_frames, resize, q, busy, frame_step), daemon=True)
    reader.start()
    n = 0
    last = None
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
        # CHW, 0..1 as the model expects. Must be C-contiguous: astype() keeps the
        # transposed layout, and Core ML converts a non-contiguous input 2.4x slower.
        last = np.ascontiguousarray(item.transpose(2, 0, 1)).astype(np.float32) / 255
        frames.append(last)
        n += 1
        if len(frames) == SEQ_LEN:
            run_window()
        if progress:
            progress(n, (max_frames or n_total) // frame_step, time.perf_counter() - wall)
    reader.join()
    tm.seconds["decode+resize（并行）"] = busy.value
    if 0 < n < SEQ_LEN:  # shorter than one window: pad with the last frame, like the reference
        while len(frames) < SEQ_LEN:
            frames.append(last)
        run_window()
        run_window(flush=True)
        tail = ens.flush()[: n - 1]
    else:
        run_window(flush=True)
        tail = ens.flush() if n else []
    for heat in tail:
        locate(heat)
    tm.frames = n

    xs, ys, vis = np.array(xs[:n], float), np.array(ys[:n], float), np.array(vis[:n], int)
    if inpaint and n >= INPAINT_LEN:
        t0 = time.perf_counter()
        xs, ys, vis = _inpaint(inpaintnet, xs, ys, vis, w, h)
        tm.seconds["inpaint"] += time.perf_counter() - t0
    tm.wall_s = time.perf_counter() - wall
    if frame_step > 1:
        return expand_rows(xs, ys, vis, frame_step, max_frames or n_total), tm
    rows = [(i, int(vis[i]), int(xs[i]), int(ys[i])) for i in range(n)]
    return rows, tm


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


def write_csv(rows, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Frame", "Visibility", "X", "Y"])
        wr.writerows(rows)
