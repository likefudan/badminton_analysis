"""Command line: `badminton <command> ...`  (run `badminton -h`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="badminton", description="羽毛球视频分析工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="查看视频参数（分辨率、帧率、是否 HDR）和可用的推理后端")
    s.add_argument("video")

    s = sub.add_parser("sdr", help="把 iPhone 的 HDR 视频转成 SDR（颜色不发灰）")
    s.add_argument("video")
    s.add_argument("output")
    s.add_argument("--height", type=int, help="同时缩放到这个高度，例如 4K 转 1080")

    s = sub.add_parser("calibrate", help="在画面上点球场标记点，保存成 court.json")
    s.add_argument("video")
    s.add_argument("-o", "--output", default=None, help="默认是视频同名的 .court.json")
    s.add_argument("--time", type=float, default=1.0, help="用第几秒的画面来点（秒）")
    s.add_argument("--points", help="不开窗口，直接给点：name=x,y;name=x,y;...")

    s = sub.add_parser("skeleton", help="第 1 步：输出叠加骨架的视频和每帧的姿态数据")
    s.add_argument("video")
    s.add_argument("-o", "--output", help="输出视频，默认是 <视频名>.skeleton.mp4")
    s.add_argument("--court", help="calibrate 生成的 court.json；不给就不滤掉场外的人")
    s.add_argument("--singles", action="store_true", help="单打：每边只保留 1 人")
    s.add_argument("--start", type=float, default=0.0, help="从第几秒开始")
    s.add_argument("--duration", type=float, help="只处理这么多秒")
    s.add_argument("--stride", type=int, default=2, help="每 N 帧跑一次模型（默认 2，60fps 视频即 30 次/秒）")
    _model_args(s)
    s.add_argument("--no-court", action="store_true", help="不画球场线")
    s.add_argument("--show-roi", action="store_true", help="画出检测区域")
    s.add_argument("--full-frame", action="store_true", help="在整幅画面上检测，不裁到球场区域")
    s.add_argument("--encoder", default="libx264", help="ffmpeg 编码器，Mac 上可用 h264_videotoolbox")

    s = sub.add_parser("bench", help="测速：比较不同推理后端的速度")
    s.add_argument("video")
    s.add_argument("--court")
    s.add_argument("--frames", type=int, default=120)
    s.add_argument("--devices", default="auto,cpu", help="逗号分隔：auto, coreml, cuda, tensorrt, cpu")
    s.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])

    s = sub.add_parser("profile", help="分段计时：时间花在模型推理还是前后处理、解码编码上")
    s.add_argument("video")
    s.add_argument("--court")
    s.add_argument("--frames", type=int, default=120)
    s.add_argument("--start", type=float, default=0.0, help="从第几秒开始取帧")
    s.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    s.add_argument("--device", default="auto", help="分段计时用的后端")
    s.add_argument("--devices", help="单模型测试要比较的后端，逗号分隔；默认 CPU 加几种 Core ML 设置")
    s.add_argument("--runs", type=int, default=100, help="单模型测试每种设置跑几次")
    s.add_argument("--singles", action="store_true")
    s.add_argument("--json", help="把结果另存成 JSON")

    a = p.parse_args(argv)
    return {"info": cmd_info, "sdr": cmd_sdr, "calibrate": cmd_calibrate,
            "skeleton": cmd_skeleton, "bench": cmd_bench, "profile": cmd_profile}[a.cmd](a)


def _model_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"],
                   help="模型大小；远端球员小、姿态不准时用 performance")
    s.add_argument("--device", default="auto", help="auto（默认）、coreml、cuda、tensorrt、cpu")
    s.add_argument("--det-model", help="自定义检测模型 .onnx 或 .zip")
    s.add_argument("--pose-model", help="自定义姿态模型 .onnx 或 .zip")


def cmd_info(a) -> int:
    from .pose import available_providers, resolve_device
    from .video import probe

    info = probe(a.video)
    print(f"分辨率 {info.width}x{info.height}，{info.fps:.2f} fps，{info.frames} 帧，编码 {info.codec}")
    print(f"色彩传输 {info.color_transfer or '未知'}，{'HDR：建议先运行 badminton sdr' if info.is_hdr else 'SDR'}")
    print(f"ONNX Runtime 后端：{', '.join(available_providers())}；auto 会选 {resolve_device('auto')}")
    return 0


def cmd_sdr(a) -> int:
    from .video import to_sdr

    to_sdr(a.video, a.output, a.height)
    print(f"已写入 {a.output}")
    return 0


def cmd_calibrate(a) -> int:
    import cv2

    from .calibrate import click_points, parse_points
    from .court import CourtCalibration
    from .render import draw_court, draw_markers
    from .video import probe, read_frame

    info = probe(a.video)
    idx = int(a.time * (info.fps or 30))
    if info.frames:
        idx = min(idx, info.frames - 1)
    frame = read_frame(a.video, idx)
    points = parse_points(a.points) if a.points else click_points(frame)
    if not points:
        print("已取消")
        return 1
    calib = CourtCalibration(points, (frame.shape[1], frame.shape[0]))
    out = Path(a.output or Path(a.video).with_suffix(".court.json"))
    calib.save(out)
    preview = frame.copy()
    draw_court(preview, calib, (0, 255, 0), 2)
    draw_markers(preview, calib)
    cv2.imwrite(str(out.with_suffix(".png")), preview)
    print(f"已保存 {out}（预览图 {out.with_suffix('.png')}）")
    print(f"平均重投影误差 {calib.reprojection_error_px:.1f} 像素")
    suspect = calib.suspect_point()
    if suspect and calib.reprojection_error_px > 3 and suspect[1] < calib.reprojection_error_px / 2:
        print(f"  {suspect[0]} 可能点偏了：去掉它后其余点的误差降到 {suspect[1]:.1f} 像素，建议重点")
    if len(points) < 6:
        print("提示：只点了 4–5 个点，远端精度会差一些，建议再加发球线交点")
    return 0


def _estimator(a):
    from .pose import PoseEstimator

    est = PoseEstimator(mode=a.mode, device=a.device, det_model=getattr(a, "det_model", None),
                        pose_model=getattr(a, "pose_model", None))
    print(f"推理后端：{est.device}；检测 {est.providers['det'][0]}，姿态 {est.providers['pose'][0]}")
    return est


def cmd_skeleton(a) -> int:
    from .court import CourtCalibration
    from .pipeline import SkeletonOptions, print_progress, run_skeleton

    calib = CourtCalibration.load(a.court) if a.court else None
    if calib is None:
        print("提示：没有给 --court，场外的人也会画上骨架。先运行 badminton calibrate。")
    est = _estimator(a)
    out = Path(a.output or Path(a.video).with_suffix(".skeleton.mp4"))
    opts = SkeletonOptions(
        stride=a.stride, start_s=a.start, duration_s=a.duration,
        players_per_side=1 if a.singles else 2, draw_court=not a.no_court,
        draw_roi=a.show_roi, use_roi=not a.full_frame, encoder=a.encoder,
    )
    res = run_skeleton(a.video, est, out, out.with_suffix(".npz"), calib, opts, print_progress)
    print(f"已写入 {out} 和 {out.with_suffix('.npz')}")
    print(f"总耗时 {res['elapsed_s']:.1f} 秒；{est.timing.summary()}")
    return 0


def cmd_bench(a) -> int:
    from .court import CourtCalibration, detection_roi
    from .players import court_box_filter
    from .pose import PoseEstimator
    from .video import read_frames

    calib = CourtCalibration.load(a.court) if a.court else None
    roi = detection_roi(calib) if calib else None
    keep = court_box_filter(calib) if calib else None
    frames = [f for _, f in read_frames(a.video, 0, a.frames)]
    results = {}
    for dev in a.devices.split(","):
        est = PoseEstimator(mode=a.mode, device=dev.strip())
        est(frames[0], roi, keep)  # warm-up: Core ML compiles the model on first run
        est.timing = type(est.timing)()
        for f in frames:
            est(f, roi, keep)
        results[f"{dev}->{est.device}"] = est.timing.summary()
        print(f"{dev}（{est.device}，{est.providers['det'][0]}）：{est.timing.summary()}")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _model_inputs(est, video: str, frame_idx: int, calib) -> dict:
    """Real input tensors for the single-model benchmark, made with rtmlib's own
    preprocessing from one video frame (detector: the court crop; pose: the
    first person found, or the whole frame if nobody is)."""
    import numpy as np

    from .court import detection_roi
    from .video import read_frame

    frame = read_frame(video, frame_idx)
    x0, y0, x1, y1 = detection_roi(calib) if calib is not None else (0, 0, frame.shape[1], frame.shape[0])
    det_img, _ = est.det.preprocess(frame[y0:y1, x0:x1])
    res = est(frame, (x0, y0, x1, y1))
    bbox = res.bboxes[0] if len(res) else [0, 0, frame.shape[1], frame.shape[0]]
    pose_img, _, _ = est.pose.preprocess(frame, bbox)
    to_nchw = lambda im: np.ascontiguousarray(im.transpose(2, 0, 1)[None], dtype=np.float32)  # noqa: E731
    return {"检测": to_nchw(det_img), "姿态": to_nchw(pose_img)}


def cmd_profile(a) -> int:
    from .court import CourtCalibration
    from .pose import PoseEstimator
    from .profiling import bench_model, default_device_matrix, pipeline
    from .video import probe

    calib = CourtCalibration.load(a.court) if a.court else None
    est = PoseEstimator(mode=a.mode, device=a.device)
    report: dict = {"mode": a.mode, "device": est.device, "providers": est.providers}

    print(f"== 1. 整条流程分段计时（{est.device}，{a.mode}，{a.frames} 帧）==")
    fps = probe(a.video).fps or 30
    timer, n, wall = pipeline(a.video, est, calib, a.frames, 1 if a.singles else 2,
                              start_frame=int(a.start * fps))
    rows = sorted(timer.table(n), key=lambda r: -r[1])
    measured = sum(r[1] for r in rows)
    print(f"{'阶段':<14}{'ms/帧':>8}{'占比':>8}{'ms/次':>8}{'次数':>7}")
    for name, per_frame, per_call, calls in rows:
        print(f"{name:<14}{per_frame:8.1f}{100 * per_frame / max(1e-9, 1000 * wall / max(1, n)):7.0f}%{per_call:8.2f}{calls:7d}")
    wall_ms = 1000 * wall / max(1, n)
    print(f"{'其他（未单独计时）':<14}{wall_ms - measured:8.1f}{100 * (wall_ms - measured) / max(1e-9, wall_ms):7.0f}%")
    print(f"{'合计（实测墙钟）':<14}{wall_ms:8.1f}      -> {1000 / max(1e-9, wall_ms):.1f} 帧/秒（每帧都跑模型）")
    model_ms = sum(r[1] for r in rows if "模型推理" in r[0])
    print(f"其中模型推理 {model_ms:.1f} ms/帧，占 {100 * model_ms / max(1e-9, wall_ms):.0f}%；其余是 Python/numpy/解码/编码")
    report["pipeline"] = {"frames": n, "wall_ms_per_frame": wall_ms,
                          "stages": [dict(zip(("stage", "ms_per_frame", "ms_per_call", "calls"), r)) for r in rows]}

    devices = a.devices.split(",") if a.devices else default_device_matrix()
    inputs = _model_inputs(est, a.video, int(a.start * fps), calib)
    print(f"\n== 2. 单独测模型（固定输入，每种设置 {a.runs} 次）==")
    print(f"{'模型':<10}{'后端设置':<42}{'ms/次':>8}{'Core ML 接管节点':>18}{'分段':>6}")
    report["models"] = []
    for label, tool in (("检测", est.det), ("姿态", est.pose)):
        for dev in devices:
            r = bench_model(tool.onnx_model, dev.strip(), a.runs, x=inputs[label])
            cov = f"{r.coreml_nodes}/{r.graph_nodes}" if r.graph_nodes else "-"
            part = str(r.partitions) if r.partitions is not None else "-"
            ms = f"{r.ms_per_run:8.2f}" if not r.error else "    失败"
            print(f"{label:<10}{dev:<42}{ms}{cov:>18}{part:>6}" + (f"  {r.error[:80]}" if r.error else ""))
            report["models"].append({"model": label, **r.__dict__})
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print(f"\n已保存 {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
