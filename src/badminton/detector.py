"""Person detector with the NMS tail cut off the ONNX graph.

rtmlib's YOLOX ONNX files end in TopK -> NonMaxSuppression -> Gather/Concat.
The size of every tensor after NMS is "number of people found", which changes
from frame to frame. Core ML cannot handle that well:

  * each new size makes Core ML re-specialise the post-NMS partitions
    (~10 ms per frame for yolox-m, more than the whole yolox-tiny network);
  * a frame with nobody in it makes an empty tensor, which the Core ML EP
    refuses outright (see pose._is_empty_nms_error).

So the graph is cut where the per-anchor outputs are complete — decoded
boxes (1, A, 4) in model-input pixels and class scores (1, A, C) — and the
cheap tail (score threshold, NMS over a few dozen boxes) runs in numpy. Only
fixed-shape tensors cross into Core ML, the graph becomes one partition, and
the results are the same boxes the full graph gives.

Matching the full graph + rtmlib: the graph's NMS is per class with IoU
threshold 0.65, then keeps the top 100 detections; rtmlib then keeps those
scoring above 0.3, of any class. The HumanArt detectors only ever score the
person class (other classes stay < 1e-5), so person scores alone are used.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

GRAPH_IOU_THR = 0.65  # NonMaxSuppression iou_threshold in the rtmlib YOLOX graphs
GRAPH_MAX_DET = 100  # final TopK in the graphs
RTMLIB_SCORE_THR = 0.3  # rtmlib's filter on the NMS model's output

# Ops between the raw per-anchor tensors and the NMS inputs (pre-NMS TopK gather).
_PASS_THROUGH = {"Gather", "Transpose", "Squeeze", "Unsqueeze", "Reshape"}


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS, IoU as ONNX NonMaxSuppression computes it (no +1 on sizes).

    Returns indices of kept boxes, highest score first.
    """
    order = np.argsort(-scores, kind="stable")
    x0, y0, x1, y1 = boxes.T
    area = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0, np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]))
        h = np.maximum(0, np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]))
        inter = w * h
        union = area[i] + area[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=int)


def person_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    ratio: float,
    score_thr: float = RTMLIB_SCORE_THR,
    iou_thr: float = GRAPH_IOU_THR,
    max_det: int = GRAPH_MAX_DET,
) -> np.ndarray:
    """Raw per-anchor outputs -> (N, 4) person boxes in the input image's pixels.

    boxes: (A, 4) x0, y0, x1, y1 in model-input pixels; scores: (A,) person score.
    Thresholding before NMS gives the same result as NMS-then-threshold: a box
    below the threshold can never suppress one above it.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = scores > score_thr
    boxes, scores = boxes[m], scores[m]
    keep = nms(boxes, scores, iou_thr)[:max_det]
    return boxes[keep] / ratio


def raw_output_names(model) -> tuple[str, str] | None:
    """Names of the (boxes, scores) tensors feeding NonMaxSuppression, before any gather."""
    producer = {o: n for n in model.graph.node for o in n.output}
    nms_nodes = [n for n in model.graph.node if n.op_type == "NonMaxSuppression"]
    if len(nms_nodes) != 1:
        return None

    def walk_back(name: str) -> str:
        while name in producer and producer[name].op_type in _PASS_THROUGH:
            name = producer[name].input[0]
        return name

    return walk_back(nms_nodes[0].input[0]), walk_back(nms_nodes[0].input[1])


def cut_before_nms(path: str | os.PathLike) -> str | None:
    """Write `<model>_nonms.onnx` next to `path` (once) and return its path.

    Returns None if the graph does not look like an rtmlib YOLOX export or
    `onnx` is missing; the caller then keeps the full model.
    """
    src = Path(path)
    dst = src.with_name(src.stem + "_nonms.onnx")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return str(dst)
    try:
        import onnx
        from onnx.utils import extract_model

        model = onnx.load(str(src), load_external_data=False)
        names = raw_output_names(model)
        if names is None:
            return None
        tmp = dst.with_suffix(".tmp.onnx")
        extract_model(str(src), str(tmp), [model.graph.input[0].name], list(names))
        os.replace(tmp, dst)  # atomic: a half-written file is never picked up
        return str(dst)
    except Exception:  # noqa: BLE001 - any failure means "use the full model"
        return None


class NoNmsDetector:
    """Drop-in for rtmlib's YOLOX (same call, preprocess, inference, postprocess)
    running the graph cut before NMS and finishing in numpy."""

    def __init__(self, yolox, nonms_path: str, session=None, score_thr: float = RTMLIB_SCORE_THR) -> None:
        self.yolox = yolox
        self.onnx_model = nonms_path
        self.session = session or yolox.session  # PoseEstimator sets the real one
        self.model_input_size = yolox.model_input_size
        self.score_thr = score_thr

    def preprocess(self, img: np.ndarray):
        return self.yolox.preprocess(img)

    def inference(self, img: np.ndarray):
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)
        # outputs come back as (boxes, scores): extract_model keeps the order given
        return self.session.run(None, {self.session.get_inputs()[0].name: x})

    def postprocess(self, outputs, ratio: float) -> np.ndarray:
        boxes, scores = outputs
        return person_boxes(boxes[0], scores[0][:, 0], ratio, self.score_thr)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x, ratio = self.preprocess(img)
        return self.postprocess(self.inference(x), ratio)
