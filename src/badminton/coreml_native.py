"""Native Core ML backend (device "coremlnative"), optional extra `[coreml]`.

ONNX Runtime's Core ML provider hands Core ML an fp32 model. That runs on the
GPU in fp32 and can never use the Neural Engine. Converting the same weights
to a native fp16 Core ML model with coremltools roughly halves the GPU time
(yolox-m 640x640: 22 -> 11 ms on an M5).

On first use each ONNX model is converted and cached under
~/.cache/badminton/ (ONNX -> PyTorch via onnx2torch -> coremltools, fp16).
Later runs only need coremltools. The detector is the graph cut before NMS
(see detector.py); the pose model is converted for fixed batches of 1, 2
and 4 people so that every player in a frame goes through one call (M5,
rtmpose-m: 2.0 / 2.7 / 4.0 ms per call).

coremltools has no ONNX front end any more, hence the PyTorch detour; torch
and onnx2torch are needed only for the one-off conversion.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

CACHE_DIR = Path(os.environ.get("BADMINTON_CACHE", Path.home() / ".cache" / "badminton"))
POSE_BATCHES = (1, 2, 4)  # one model per size; the smallest that fits is used (4 = doubles)
COMPUTE_UNITS = "CPU_AND_GPU"  # measured fastest on M5; the Neural Engine was slower for these models


def available() -> bool:
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


def can_convert() -> bool:
    try:
        import onnx2torch  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return available()


def _cache_path(onnx_path: str, tag: str) -> Path:
    digest = hashlib.sha1(Path(onnx_path).read_bytes()).hexdigest()[:10]
    return CACHE_DIR / f"{Path(onnx_path).stem}_{tag}_{digest}.mlpackage"


def _prepare_onnx(src: str, dst: str, batch: int) -> None:
    """Pin the batch dimension, fold constants (ONNX Runtime basic optimisation)
    and tidy Clip nodes so onnx2torch can read the graph."""
    import onnx
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = dst
    for name in ("batch",):
        opts.add_free_dimension_override_by_name(name, batch)
    ort.InferenceSession(src, opts, providers=["CPUExecutionProvider"])
    m = onnx.load(dst)
    for inp in m.graph.input:
        d = inp.type.tensor_type.shape.dim[0]
        d.ClearField("dim_param")
        d.dim_value = batch
    for n in m.graph.node:
        if n.op_type == "Clip":  # onnx2torch rejects an empty trailing "max" input
            while n.input and n.input[-1] == "":
                n.input.pop()
    onnx.save(m, dst)


def convert(onnx_path: str, out: Path, input_hw: tuple[int, int], batch: int, output_names: list[str]) -> Path:
    import coremltools as ct
    import onnx
    import torch
    from onnx2torch import convert as to_torch

    with tempfile.TemporaryDirectory() as tmp:
        prepared = os.path.join(tmp, "model.onnx")
        _prepare_onnx(onnx_path, prepared, batch)
        net = to_torch(onnx.load(prepared)).eval()
        x = torch.zeros(batch, 3, *input_hw)
        with torch.no_grad():
            traced = torch.jit.trace(net, x)
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=x.shape)],
            outputs=[ct.TensorType(name=n) for n in output_names],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS13,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tmp) / out.name
        ml.save(str(staging))
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(staging), out)
    return out


def load(onnx_path: str, tag: str, input_hw, batch: int, output_names: list[str]):
    import coremltools as ct

    path = _cache_path(onnx_path, tag)
    if not path.exists():
        if not can_convert():
            raise RuntimeError("第一次使用 coremlnative 需要转换模型：pip install -e \".[coreml]\"")
        print(f"转换成 Core ML 模型（只在第一次）：{path.name}")
        convert(onnx_path, path, input_hw, batch, output_names)
    return ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS))


class _Session:
    """Just enough of an onnxruntime session for reporting."""

    def __init__(self, label: str) -> None:
        self.label = label

    def get_providers(self) -> list[str]:
        return [self.label]


class NativeDetector:
    """YOLOX without NMS on native Core ML; same interface as detector.NoNmsDetector."""

    def __init__(self, yolox, nonms_path: str, score_thr: float) -> None:
        h, w = yolox.model_input_size
        self.yolox = yolox
        self.onnx_model = nonms_path
        self.model_input_size = yolox.model_input_size
        self.score_thr = score_thr
        self.model = load(nonms_path, "det_fp16", (h, w), 1, ["boxes", "scores"])
        self.session = _Session(f"CoreML native ({COMPUTE_UNITS}, fp16)")

    def preprocess(self, img):
        return self.yolox.preprocess(img)

    def inference(self, img):
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)
        out = self.model.predict({"input": x})
        return out["boxes"], out["scores"]

    def postprocess(self, outputs, ratio):
        from .detector import person_boxes

        boxes, scores = outputs
        return person_boxes(boxes[0], scores[0][:, 0], ratio, self.score_thr)

    def __call__(self, img):
        x, ratio = self.preprocess(img)
        return self.postprocess(self.inference(x), ratio)


class NativePose:
    """RTMPose on native Core ML; all people of a frame in one call (up to 4)."""

    def __init__(self, rtmpose) -> None:
        self.rtmpose = rtmpose
        self.onnx_model = rtmpose.onnx_model
        self.model_input_size = rtmpose.model_input_size
        self._models: dict = {}
        self.model = self._model(2)  # the common case (singles); others load on first need
        self.session = _Session(f"CoreML native ({COMPUTE_UNITS}, fp16)")

    def _model(self, batch: int):
        if batch not in self._models:
            w, h = self.model_input_size
            self._models[batch] = load(self.onnx_model, f"pose_fp16_b{batch}", (h, w), batch,
                                       ["simcc_x", "simcc_y"])
        return self._models[batch]

    def preprocess(self, image, bbox):
        return self.rtmpose.preprocess(image, bbox)

    def inference(self, imgs: np.ndarray):
        """imgs: (n, H, W, 3) preprocessed crops -> (simcc_x, simcc_y) for those n."""
        n = len(imgs)
        xs, ys = [], []
        biggest = POSE_BATCHES[-1]
        for i in range(0, n, biggest):
            chunk = imgs[i:i + biggest]
            batch = next(b for b in POSE_BATCHES if b >= len(chunk))
            x = np.zeros((batch, 3, *chunk.shape[1:3]), np.float32)
            x[:len(chunk)] = chunk.transpose(0, 3, 1, 2)
            out = self._model(batch).predict({"input": x})
            xs.append(out["simcc_x"][:len(chunk)])
            ys.append(out["simcc_y"][:len(chunk)])
        return np.concatenate(xs), np.concatenate(ys)

    def postprocess(self, outputs, center, scale):
        return self.rtmpose.postprocess(outputs, center, scale)

    def __call__(self, image: np.ndarray, bboxes: list):
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]
        prepped = [self.preprocess(image, b) for b in bboxes]
        sx, sy = self.inference(np.stack([p[0] for p in prepped]))
        kpts, scores = [], []
        for i, (_, center, scale) in enumerate(prepped):
            k, s = self.postprocess([sx[i:i + 1], sy[i:i + 1]], center, scale)
            kpts.append(k)
            scores.append(s)
        return np.concatenate(kpts), np.concatenate(scores)
