"""Experiment: W8A8 (int8 weights + int8 activations) Core ML models vs fp16.

Not used by the tool; it measures whether 8-bit models let the Neural Engine
beat the GPU on this Mac, and what that costs in accuracy.

  python scripts/quantize_coreml.py data/singles.mp4 --court data/singles.court.json --mode balanced

Calibrates on real frames of the video (detector: the court crop; pose: the
players' crops), evaluates on other frames, and prints ms per call for each
compute unit plus box / keypoint differences against the fp16 model.
Needs the `[coreml]` extra. Outputs go to ~/.cache/badminton/.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def collect(est, video, calib, frame_ids):
    from badminton.court import detection_roi
    from badminton.players import court_box_filter
    from badminton.video import read_frame

    x0, y0, x1, y1 = detection_roi(calib)
    keep = court_box_filter(calib)
    det_in, ratios, pose_in = [], [], []
    for i in frame_ids:
        frame = read_frame(video, i)
        img, ratio = est.det.preprocess(frame[y0:y1, x0:x1])
        det_in.append(np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32))
        ratios.append(ratio)
        boxes = est._detect(frame[y0:y1, x0:x1]) + [x0, y0, x0, y0]
        for b in boxes[keep(boxes)]:
            crop, _, _ = est.pose.preprocess(frame, b)
            pose_in.append(np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32))
    return det_in, ratios, pose_in


def _calibrate_on_gpu() -> None:
    """coremltools runs calibration with a CPU_ONLY copy of the model that
    exposes every intermediate tensor. For yolox that crashes inside Apple's
    BNNS (SIGTRAP, macOS 26 / coremltools 9.0), so run it on the GPU instead."""
    import coremltools as ct
    from coremltools.optimize.coreml.experimental import _model_debugger as dbg

    orig = dbg.ModelDebugger.predict_intermediate_outputs

    def on_gpu(self, inputs, names, compute_units=ct.ComputeUnit.CPU_AND_GPU):
        return orig(self, inputs, names, ct.ComputeUnit.CPU_AND_GPU)

    dbg.ModelDebugger.predict_intermediate_outputs = on_gpu


def quantize(model, samples):
    import coremltools.optimize.coreml as cto

    _calibrate_on_gpu()
    from coremltools.optimize.coreml.experimental import OpActivationLinearQuantizerConfig, linear_quantize_activations

    # Some 1x1 convs have output channels whose weights are all zero; per-channel
    # int8 then divides by a zero scale and fills the layer with NaN. Keep those
    # weights in fp16.
    meta = cto.get_weights_metadata(model, weight_threshold=2048)
    zero_ch = {name: None for name, w in meta.items()
               if (np.abs(np.asarray(w.val, np.float32).reshape(w.val.shape[0], -1)).max(1) == 0).any()}
    act_cfg = cto.OptimizationConfig(global_config=OpActivationLinearQuantizerConfig(mode="linear_symmetric"))
    a8 = linear_quantize_activations(model, act_cfg, [{"input": x} for x in samples])
    w_cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"), op_name_configs=zero_ch)
    return cto.linear_quantize_weights(a8, w_cfg)


def timed(path, unit, x, runs=50):
    import coremltools as ct

    m = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, unit))
    for _ in range(5):
        m.predict({"input": x})
    t = time.perf_counter()
    for _ in range(runs):
        m.predict({"input": x})
    return 1000 * (time.perf_counter() - t) / runs, m


def main() -> None:
    import coremltools as ct

    from badminton import coreml_native as cn
    from badminton.court import CourtCalibration
    from badminton.detector import person_boxes
    from badminton.pose import PoseEstimator
    from badminton.video import probe

    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--court", required=True)
    p.add_argument("--mode", default="balanced")
    p.add_argument("--calib-frames", type=int, default=64)
    p.add_argument("--eval-frames", type=int, default=20)
    a = p.parse_args()

    est = PoseEstimator(mode=a.mode, device="coremlnative")
    calib = CourtCalibration.load(a.court)
    n = probe(a.video).frames
    cal_ids = np.linspace(0, n - 1, a.calib_frames, endpoint=False).astype(int)
    eval_ids = (np.linspace(0, n - 1, a.eval_frames, endpoint=False) + n / a.calib_frames / 2).astype(int)
    eval_ids = np.setdiff1d(eval_ids, cal_ids)  # evaluate on frames not used for calibration
    cal_det, _, cal_pose = collect(est, a.video, calib, cal_ids)
    ev_det, ev_ratio, ev_pose = collect(est, a.video, calib, eval_ids)
    print(f"标定样本：检测 {len(cal_det)} 帧，姿态 {len(cal_pose)} 个人；评估：{len(ev_det)} 帧，{len(ev_pose)} 个人")

    h, w = est.det.model_input_size
    pw, ph = est.pose.model_input_size
    models = {
        "检测": (est.det.onnx_model, "det", (h, w), ["boxes", "scores"], cal_det, ev_det),
        "姿态": (est.pose.onnx_model, "pose", (ph, pw), ["simcc_x", "simcc_y"], cal_pose, ev_pose),
    }
    for label, (onnx_path, tag, hw, outs, cal, ev) in models.items():
        fp16 = cn._cache_path(onnx_path, f"{tag}_fp16" if tag == "det" else f"{tag}_fp16_b1")
        if not fp16.exists():
            cn.convert(onnx_path, fp16, hw, 1, outs)
        w8a8 = cn._cache_path(onnx_path, f"{tag}_w8a8")
        if not w8a8.exists():
            t = time.perf_counter()
            quantize(ct.models.MLModel(str(fp16), compute_units=ct.ComputeUnit.CPU_ONLY), cal).save(str(w8a8))
            print(f"{label}：量化用时 {time.perf_counter() - t:.0f} 秒")
        print(f"\n{label}（{fp16.name.split('_')[0]}）ms/次")
        ref = {}
        for unit in ("CPU_AND_GPU", "CPU_AND_NE", "ALL", "CPU_ONLY"):
            row = []
            for name, path in (("fp16", fp16), ("W8A8", w8a8)):
                ms, m = timed(path, unit, ev[0])
                row.append(f"{name} {ms:6.2f}")
                if unit == "CPU_AND_GPU":
                    ref[name] = [m.predict({"input": x}) for x in ev]
            print(f"  {unit:12s} " + "   ".join(row))
        # accuracy of W8A8 against fp16 (both on the GPU)
        if tag == "det":
            worst_px, worst_iou, same = 0.0, 1.0, 0
            for o16, o8, r in zip(ref["fp16"], ref["W8A8"], ev_ratio):
                b16 = person_boxes(o16["boxes"][0], o16["scores"][0][:, 0], 1.0)
                b8 = person_boxes(o8["boxes"][0], o8["scores"][0][:, 0], 1.0)
                same += len(b16) == len(b8)
                if len(b16) and len(b8):
                    d = np.abs(b16[:, None] - b8[None]).max(-1)
                    worst_px = max(worst_px, d.min(1).max())
            print(f"  W8A8 对 fp16：框数相同 {same}/{len(ev)} 帧，框角点最大偏差 {worst_px:.2f} 像素（模型输入 {w}×{h} 坐标）")
        else:
            diffs, conf_diffs = [], []
            for o16, o8 in zip(ref["fp16"], ref["W8A8"]):
                k16 = np.stack([o16["simcc_x"][0].argmax(-1), o16["simcc_y"][0].argmax(-1)], -1) / 2
                k8 = np.stack([o8["simcc_x"][0].argmax(-1), o8["simcc_y"][0].argmax(-1)], -1) / 2
                d = np.hypot(*(k16 - k8).T)
                conf = np.minimum(o16["simcc_x"][0].max(-1), o16["simcc_y"][0].max(-1)) >= 0.3
                diffs.append(d.max())
                if conf.any():
                    conf_diffs.append(d[conf].max())
            print(f"  W8A8 对 fp16：关节点偏差（{pw}×{ph} 裁剪图像素）中位数 {np.median(diffs):.2f}，"
                  f"最大 {max(diffs):.2f}；置信度 ≥0.3 的点最大 {max(conf_diffs) if conf_diffs else float('nan'):.2f}")


if __name__ == "__main__":
    main()
