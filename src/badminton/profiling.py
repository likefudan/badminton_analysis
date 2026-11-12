"""Where does the time go? Per-stage timing and raw model benchmarks.

Two views:
  * models(): each ONNX model alone, fed a fixed tensor, under several
    execution-provider settings (CPU, Core ML with different compute units and
    model formats). Also reports how many graph nodes Core ML accepted: nodes
    it rejects run on the CPU, and every CPU/Core ML boundary costs a copy.
  * pipeline(): the real step-1 loop on video frames, split into decode,
    detector pre-processing / inference / post-processing, pose pre / infer /
    post, player selection, drawing and encoding.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StageTimer:
    total_s: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.total_s[name] += time.perf_counter() - t
            self.calls[name] += 1

    def wrap(self, obj, method: str, name: str) -> None:
        """Time every call of obj.method under `name` (instance-level patch)."""
        fn = getattr(obj, method)

        def timed(*a, **k):
            with self.stage(name):
                return fn(*a, **k)

        setattr(obj, method, timed)

    def table(self, frames: int) -> list[tuple[str, float, float, int]]:
        """(stage, ms per frame, ms per call, calls)."""
        rows = []
        for name, s in self.total_s.items():
            c = self.calls[name]
            rows.append((name, 1000 * s / max(1, frames), 1000 * s / max(1, c), c))
        return rows


# --- raw model benchmark -------------------------------------------------------

_PARTITION_RE = re.compile(
    r"number of partitions supported by CoreML:\s*(\d+).*?number of nodes in the graph:\s*(\d+)"
    r".*?number of nodes supported by CoreML:\s*(\d+)",
    re.S,
)


@contextmanager
def _capture_fd2():
    """Capture C-level stderr (ONNX Runtime logs there, not to sys.stderr)."""
    buf = tempfile.TemporaryFile(mode="w+b")
    saved = os.dup(2)
    os.dup2(buf.fileno(), 2)
    out = {"text": ""}
    try:
        yield out
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        buf.seek(0)
        out["text"] = buf.read().decode("utf-8", "replace")
        buf.close()


@dataclass
class ModelBench:
    model: str
    device: str
    load_s: float
    ms_per_run: float
    providers: list[str]
    coreml_nodes: int | None = None
    graph_nodes: int | None = None
    partitions: int | None = None
    error: str = ""


def bench_model(path: str, device: str, runs: int = 100, fixed_batch: bool = True,
                x: np.ndarray | None = None) -> ModelBench:
    """Time one ONNX model alone. `x` is the input tensor; pass a real preprocessed
    frame for a detector: on random noise it finds nobody, and the Core ML EP
    cannot run the post-NMS nodes on an empty tensor."""
    import onnxruntime as ort

    from .pose import provider_config

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 1  # INFO: Core ML prints its node coverage at this level
    if fixed_batch:
        opts.add_free_dimension_override_by_name("batch", 1)
    t = time.perf_counter()
    try:
        with _capture_fd2() as log:
            sess = ort.InferenceSession(path, sess_options=opts, providers=provider_config(device))
    except Exception as e:  # noqa: BLE001 - report and keep going with other settings
        return ModelBench(os.path.basename(path), device, 0.0, float("nan"), [], error=str(e).splitlines()[0])
    load_s = time.perf_counter() - t
    m = _PARTITION_RE.search(log["text"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    if x is None:
        x = np.random.default_rng(0).random(shape, dtype=np.float32) * 255
    feed = {inp.name: np.ascontiguousarray(x, dtype=np.float32)}
    try:
        for _ in range(5):  # warm-up (Core ML compiles / specialises on first runs)
            sess.run(None, feed)
        t = time.perf_counter()
        for _ in range(runs):
            sess.run(None, feed)
    except Exception as e:  # noqa: BLE001 - report and keep going with other settings
        return ModelBench(os.path.basename(path), device, load_s, float("nan"), sess.get_providers(),
                          error=str(e).splitlines()[0])
    ms = 1000 * (time.perf_counter() - t) / runs
    return ModelBench(
        os.path.basename(path), device, load_s, ms, sess.get_providers(),
        int(m.group(3)) if m else None, int(m.group(2)) if m else None, int(m.group(1)) if m else None,
    )


def default_device_matrix() -> list[str]:
    from .pose import available_providers

    devs = ["cpu"]
    provs = available_providers()
    if "CoreMLExecutionProvider" in provs:
        devs += [
            "coreml:ALL:MLProgram",
            "coreml:CPUAndGPU:MLProgram",
            "coreml:CPUAndNeuralEngine:MLProgram",
            "coreml:ALL:NeuralNetwork",
            "coreml:CPUAndNeuralEngine:NeuralNetwork",
        ]
    if "CUDAExecutionProvider" in provs:
        devs.append("cuda")
    if "TensorrtExecutionProvider" in provs:
        devs.append("tensorrt")
    return devs


# --- pipeline breakdown --------------------------------------------------------


def pipeline(video, estimator, calib, frames: int = 120, players_per_side: int = 2,
             encode: bool = True, start_frame: int = 0) -> tuple[StageTimer, int, float]:
    """Run the step-1 loop with every stage timed. Returns (timer, frames, wall seconds)."""
    from .court import detection_roi
    from .players import court_box_filter, select_players
    from .render import draw_court, draw_players, draw_text
    from .video import VideoWriter, probe, read_frames

    info = probe(video)
    timer = StageTimer()
    for attr, prefix in (("det", "检测"), ("pose", "姿态")):
        tool = getattr(estimator, attr, None)
        if tool is None:
            continue
        timer.wrap(tool, "preprocess", f"{prefix}·前处理")
        timer.wrap(tool, "inference", f"{prefix}·模型推理")
        timer.wrap(tool, "postprocess", f"{prefix}·后处理")
    roi = detection_roi(calib) if calib is not None else None
    box_filter = court_box_filter(calib) if calib is not None else None

    it = read_frames(video, start_frame, frames + 1)
    _, first = next(it)
    estimator(first, roi, box_filter)  # warm-up outside the measurement
    timer.total_s.clear()
    timer.calls.clear()

    writer = None
    if encode:
        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        writer = VideoWriter(out, info.width, info.height, info.fps or 30)
    n = 0
    t0 = time.perf_counter()
    try:
        while True:
            with timer.stage("读帧解码"):
                try:
                    _, frame = next(it)
                except StopIteration:
                    break
            res = estimator(frame, roi, box_filter)
            with timer.stage("挑选球员"):
                players = select_players(res, calib, players_per_side)
            if writer:
                with timer.stage("画骨架和球场线"):
                    if calib is not None:
                        draw_court(frame, calib)
                    draw_players(frame, players, scale=info.height / 1080)
                    draw_text(frame, "00.00s")
                with timer.stage("写入编码"):
                    writer.write(frame)
            n += 1
    finally:
        if writer:
            writer.close()
            os.unlink(out)
    wall = time.perf_counter() - t0
    return timer, n, wall
