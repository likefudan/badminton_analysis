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

    a = p.parse_args(argv)
    return {"info": cmd_info, "sdr": cmd_sdr, "calibrate": cmd_calibrate,
            "skeleton": cmd_skeleton, "bench": cmd_bench}[a.cmd](a)


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
    from .pose import PoseEstimator
    from .video import read_frames

    calib = CourtCalibration.load(a.court) if a.court else None
    roi = detection_roi(calib) if calib else None
    frames = [f for _, f in read_frames(a.video, 0, a.frames)]
    results = {}
    for dev in a.devices.split(","):
        est = PoseEstimator(mode=a.mode, device=dev.strip())
        est(frames[0], roi)  # warm-up: Core ML compiles the model on first run
        est.timing = type(est.timing)()
        for f in frames:
            est(f, roi)
        results[f"{dev}->{est.device}"] = est.timing.summary()
        print(f"{dev}（{est.device}，{est.providers['det'][0]}）：{est.timing.summary()}")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
