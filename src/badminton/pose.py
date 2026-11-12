"""Person detection + 17-keypoint pose, run through ONNX Runtime.

The models come from rtmlib (Apache-2.0): a YOLOX person detector trained on
COCO + HumanArt and RTMPose trained on 7 body datasets, all exported to ONNX.
rtmlib downloads them to ~/.cache/rtmlib on first use.

ONNX Runtime is not CPU-only: it hands the model to an "execution provider".
On Apple silicon that is CoreML (GPU + Neural Engine), on NVIDIA it is CUDA or
TensorRT. Layers a provider cannot run fall back to the CPU, so the provider
list that was actually applied is reported for every model.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# rtmlib's Body() presets: (detector url, detector input, pose url, pose input (w, h))
_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
MODES = {
    "lightweight": (
        _BASE + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        (416, 416),
        _BASE + "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
        (192, 256),
    ),
    "balanced": (
        _BASE + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        (640, 640),
        _BASE + "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        (192, 256),
    ),
    "performance": (
        _BASE + "yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        (640, 640),
        _BASE + "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
        (288, 384),
    ),
}


def available_providers() -> list[str]:
    import onnxruntime as ort

    return ort.get_available_providers()


def resolve_device(device: str) -> str:
    """'auto' -> 'cuda' | 'coremlnative' | 'coreml' | 'cpu' depending on what is installed.

    'coremlnative' (native fp16 Core ML models, extra `[coreml]`) is preferred on
    a Mac when coremltools and the converter are installed.

    Core ML can be tuned with 'coreml:<compute units>[:<model format>]', e.g.
    'coreml:CPUAndNeuralEngine' or 'coreml:CPUAndGPU:NeuralNetwork'.
    """
    if device != "auto":
        return device
    providers = available_providers()
    if "CUDAExecutionProvider" in providers:
        return "cuda"
    if "CoreMLExecutionProvider" in providers:
        from . import coreml_native

        return "coremlnative" if coreml_native.can_convert() else "coreml"
    return "cpu"


COREML_UNITS = ("ALL", "CPUAndGPU", "CPUAndNeuralEngine", "CPUOnly")
COREML_FORMATS = ("MLProgram", "NeuralNetwork")


def provider_config(device: str) -> list:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "tensorrt":
        return [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True}),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    if device == "coreml" or device.startswith("coreml:"):
        parts = device.split(":")
        # MLProgram is the newer Core ML format and covers more operators than
        # NeuralNetwork; ALL lets Core ML choose between GPU and Neural Engine.
        units = parts[1] if len(parts) > 1 and parts[1] else "ALL"
        fmt = parts[2] if len(parts) > 2 and parts[2] else "MLProgram"
        if units not in COREML_UNITS or fmt not in COREML_FORMATS:
            raise ValueError(f"Core ML 设置不对：{device}；计算单元可选 {COREML_UNITS}，格式可选 {COREML_FORMATS}")
        return [
            ("CoreMLExecutionProvider", {"ModelFormat": fmt, "MLComputeUnits": units}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


@dataclass
class PoseResult:
    bboxes: np.ndarray  # (N, 4) x0, y0, x1, y1 in frame pixels
    keypoints: np.ndarray  # (N, 17, 2) COCO order, frame pixels
    scores: np.ndarray  # (N, 17) keypoint confidence

    @classmethod
    def empty(cls) -> "PoseResult":
        return cls(np.zeros((0, 4)), np.zeros((0, 17, 2)), np.zeros((0, 17)))

    def __len__(self) -> int:
        return len(self.bboxes)

    def select(self, idx) -> "PoseResult":
        idx = np.asarray(idx, dtype=int)
        return PoseResult(self.bboxes[idx], self.keypoints[idx], self.scores[idx])


@dataclass
class Timing:
    det_s: float = 0.0
    pose_s: float = 0.0
    frames: int = 0
    persons: int = 0

    def summary(self) -> str:
        if not self.frames:
            return "没有处理任何帧"
        det = 1000 * self.det_s / self.frames
        pose = 1000 * self.pose_s / self.frames
        per_person = 1000 * self.pose_s / max(1, self.persons)
        return (
            f"{self.frames} 帧：检测 {det:.1f} ms/帧，姿态 {pose:.1f} ms/帧"
            f"（{per_person:.1f} ms/人），合计约 {1000 / max(1e-6, det + pose):.1f} 帧/秒"
        )


class PoseEstimator:
    """Detector + top-down pose model. Call with a BGR frame."""

    def __init__(
        self,
        mode: str = "balanced",
        device: str = "auto",
        det_model: str | None = None,
        pose_model: str | None = None,
        det_score_thr: float = 0.3,
        cut_nms: bool | None = None,
    ) -> None:
        """`det_score_thr`: rtmlib applies a fixed 0.3 to the NMS models' output
        (its own score_thr is ignored there), so 0.3 is what the full graph gives.
        `cut_nms`: run the detector without its NMS tail (see detector.py);
        default on for Core ML, where the tail is slow and fails on empty frames."""
        from rtmlib import YOLOX, RTMPose

        if mode not in MODES:
            raise ValueError(f"mode 只能是 {list(MODES)}")
        det_url, det_size, pose_url, pose_size = MODES[mode]
        self.device = resolve_device(device)
        # rtmlib builds CPU sessions; they are swapped for sessions with the
        # chosen provider (and its options) right after.
        self.det = YOLOX(det_model or det_url, model_input_size=det_size, score_thr=det_score_thr, device="cpu")
        self.pose = RTMPose(pose_model or pose_url, model_input_size=pose_size, device="cpu")
        native = self.device == "coremlnative"
        if cut_nms is None:
            cut_nms = self.device.startswith("coreml")
        self.det_nms = "graph"
        if cut_nms:
            from .detector import NoNmsDetector, cut_before_nms

            path = cut_before_nms(self.det.onnx_model)
            if path is not None:
                self.det = NoNmsDetector(self.det, path, None, det_score_thr)
                self.det_nms = "numpy"
            elif native:
                raise RuntimeError("coremlnative 需要切掉 NMS 的检测模型，但切图失败")
        if native:
            from .coreml_native import NativeDetector, NativePose

            self.det = NativeDetector(self.det.yolox, self.det.onnx_model, det_score_thr)
            self.pose = NativePose(self.pose)
        self.providers = {}
        for name, tool in (("det", self.det), ("pose", self.pose)):
            if not native:
                # rtmlib feeds the pose model one person at a time; pinning the
                # dynamic batch dimension to 1 lets Core ML compile the whole graph.
                tool.session = _make_session(tool.onnx_model, self.device, fixed_dims={"batch": 1})
            self.providers[name] = tool.session.get_providers()
        self.timing = Timing()

    def _detect(self, img: np.ndarray) -> np.ndarray:
        try:
            boxes = self.det(img)
        except Exception as e:  # onnxruntime raises a plain Fail/RuntimeException
            if not _is_empty_nms_error(e):
                raise
            boxes = []
        return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)

    def __call__(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
        box_filter: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> PoseResult:
        """Detect people (inside `roi` if given), then estimate pose.

        `box_filter(boxes) -> bool mask` drops boxes before the pose model runs.
        The pose model costs a few ms per person and a broadcast or club frame
        holds many people (umpire, line judges, crowd), so filtering first is
        much faster than estimating everyone and selecting afterwards.
        """
        t0 = time.perf_counter()
        if roi is not None:
            x0, y0, x1, y1 = roi
            boxes = self._detect(frame[y0:y1, x0:x1])
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
        else:
            boxes = self._detect(frame)
        if box_filter is not None and len(boxes):
            boxes = boxes[np.asarray(box_filter(boxes), dtype=bool)]
        t1 = time.perf_counter()
        if len(boxes) == 0:
            result = PoseResult.empty()
        else:
            kpts, scores = self.pose(frame, bboxes=list(boxes))
            result = PoseResult(boxes, np.asarray(kpts, dtype=np.float64), np.asarray(scores, dtype=np.float64))
        t2 = time.perf_counter()
        self.timing.det_s += t1 - t0
        self.timing.pose_s += t2 - t1
        self.timing.frames += 1
        self.timing.persons += len(result)
        return result


def _is_empty_nms_error(err: Exception) -> bool:
    """Core ML EP failure when the detector's NMS keeps no box.

    The detector ONNX graph ends in NonMaxSuppression followed by Gather/Concat
    nodes whose size is the number of kept boxes. Core ML runs some of those
    nodes, and the Core ML EP refuses a tensor with zero elements, so a frame
    with nobody in it (empty court, close-up, black frame) raises instead of
    returning no boxes.
    """
    msg = str(err)
    return "CoreML" in msg and "zero elements" in msg


def _make_session(path: str, device: str, fixed_dims: dict[str, int] | None = None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for dim, value in (fixed_dims or {}).items():
        opts.add_free_dimension_override_by_name(dim, value)
    return ort.InferenceSession(path, sess_options=opts, providers=provider_config(device))
