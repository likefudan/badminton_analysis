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
    s.add_argument("--auto", action="store_true", help="自动标定：检测白色球场线并拟合标准球场，不用手点")
    s.add_argument("--hint", help="自动标定时指定要哪片场地：场地内任意一点的像素坐标 x,y（画面里有多片场地时用）")
    s.add_argument("--frames", type=int, default=5, help="自动标定取几帧做中值（去掉走动的人），默认 5；1 = 只用一帧")
    s.add_argument("--every", type=float, help="自动标定后每隔这么多秒再对一次线，报告机位漂移")

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

    for name, helptext in (("label", "第 2 步：给跟踪到的球员起名字（每局一次）"),
                           ("movement", "第 2 步：跑动卡片和统计 JSON；双打同时出第 3 步的站位报告"),
                           ("positioning", "第 3 步：双打站位报告（前后/左右站位、轮转、搭档间距、空当）")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("video")
        s.add_argument("--court", required=True, help="calibrate 生成的 court.json")
        s.add_argument("--data", help="skeleton 生成的 .npz，默认 <视频名>.skeleton.npz")
        s.add_argument("--players", help="球员名字文件，默认 <视频名>.players.json")
        s.add_argument("--singles", action="store_true", help="单打：每边 1 人（默认从数据里判断）")
        s.add_argument("--no-appearance", action="store_true", help="不读视频比对球衣颜色（更快，双打交叉换位时容易换错编号）")
        s.add_argument("--cuts", help="剪辑过的视频：剪辑点的时间（秒），逗号分隔；跑动不跨这些点计算。换镜头会自动识别，同一机位拼接要手动给")
        s.add_argument("--start", type=float, default=0.0, help="从第几秒开始统计（跳过开头还在摆手机的部分）")
        if name == "label":
            s.add_argument("--names", help='直接给名字："1=小王,2=小李"；不给就在终端里逐个输入')
            s.add_argument("--time", type=float, help="用第几秒的画面标编号，默认找所有人都在场的第一帧")
        else:
            s.add_argument("-o", "--out-dir", help="输出目录，默认和视频同目录")
            s.add_argument("--shuttle", help="可选：TrackNetV3 的 _ball.csv，只用来对照回合切分")

    s = sub.add_parser("shuttle", help="羽毛球轨迹（TrackNetV3，原生 Core ML），输出每帧球的位置 CSV")
    s.add_argument("video")
    s.add_argument("-o", "--output", help="输出 CSV，默认 <视频名>.shuttle.csv")
    s.add_argument("--tracknet-dir", help="TrackNetV3 文件夹（只在第一次转换模型时需要），也可用环境变量 TRACKNETV3_DIR")
    s.add_argument("--compute-units", default="ALL", choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"])
    s.add_argument("--no-inpaint", action="store_true", help="不用 InpaintNet 补短暂丢失")
    s.add_argument("--max-frames", type=int, help="只处理前 N 帧（试跑）")
    s.add_argument("--overlay", action="store_true", help="同时输出画了轨迹的视频 <视频名>.shuttle.mp4")
    s.add_argument("--court", help="球场标定：给了就用改进的选点（去掉静止干扰点、场地上方空间、按轨迹连续性挑点）")
    s.add_argument("--raw-output", help="另存 TrackNetV3 原版逐帧选点的结果（用来对比）")
    s.add_argument("--frame-step", type=int, default=1,
                   help="每 N 帧送一帧给模型；60fps 视频建议 2（TrackNetV3 是用 30fps 训练的），跳过的帧用插值补")
    s.add_argument("--resize", default="pil", choices=["pil", "area"],
                   help="缩小画面的方法：pil 和 TrackNetV3 原版一样（默认）；area 稍快但位置稍有偏差")

    s = sub.add_parser("stabilize", help="手持或晃动的视频：把每一帧对齐到参考帧，之后才能用一次标定")
    s.add_argument("video")
    s.add_argument("output")
    s.add_argument("--ref-time", type=float, required=True, help="参考帧在第几秒（选一帧球场线清楚、人少挡线的）")
    s.add_argument("--start", type=float, default=0.0)
    s.add_argument("--duration", type=float)
    s.add_argument("--floor-from", type=float, help="只用画面这个高度比例以下的特征点（地面），例如 0.45")

    s = sub.add_parser("analyze", help="一键处理：标定（已有就沿用）→ 骨架 → 跑动卡片 → 双打站位报告")
    s.add_argument("video")
    s.add_argument("--court", help="court.json，默认 <视频名>.court.json；没有就先打开窗口点标记点")
    s.add_argument("-o", "--out-dir", help="输出目录，默认 <视频名>_analysis/")
    s.add_argument("--singles", action="store_true")
    s.add_argument("--names", help='球员名字："1=小王,2=小李,3=老张,4=老刘"（编号见 players.png）')
    s.add_argument("--force", action="store_true", help="骨架数据已存在也重新跑")
    s.add_argument("--no-skeleton-video", action="store_true", help="不输出骨架视频（快一些）")
    s.add_argument("--stride", type=int, default=2)
    s.add_argument("--duration", type=float, help="只处理前多少秒（试跑）")
    s.add_argument("--cuts")
    _model_args(s)

    a = p.parse_args(argv)
    return {"info": cmd_info, "sdr": cmd_sdr, "calibrate": cmd_calibrate,
            "skeleton": cmd_skeleton, "bench": cmd_bench, "profile": cmd_profile,
            "label": cmd_label, "movement": cmd_movement, "shuttle": cmd_shuttle,
            "stabilize": cmd_stabilize, "positioning": cmd_positioning, "analyze": cmd_analyze}[a.cmd](a)


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
    if a.auto:
        return _calibrate_auto(a, info, idx)
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


def _calibrate_auto(a, info, idx: int) -> int:
    """calibrate --auto: fit the court to the white lines, no clicking."""
    import cv2
    import numpy as np

    from .autocal import AutoCalibrationError, auto_calibrate, intersection_distance_px, median_frame, track
    from .render import draw_court, draw_markers
    from .video import read_frame

    fps = info.fps or 30
    last = info.frames - 1 if info.frames else None
    # a few frames half a second apart: the median keeps the floor and drops the players
    wanted = [idx + int(round(k * 0.5 * fps)) for k in range(max(1, a.frames))]
    frames = [read_frame(a.video, i) for i in wanted if last is None or i <= last]
    frame = median_frame(frames)
    hint = tuple(float(v) for v in a.hint.split(",")) if a.hint else None
    out = Path(a.output or Path(a.video).with_suffix(".court.json"))
    try:
        res = auto_calibrate(frame, hint)
    except AutoCalibrationError as e:
        print(f"自动标定失败：{e}")
        bad = getattr(e, "result", None)
        if bad is not None:  # show what was found, but do not save it as the calibration
            preview = frames[0].copy()
            draw_court(preview, bad.calib, (0, 0, 255), 2)
            rejected = out.with_suffix(".rejected.png")
            cv2.imwrite(str(rejected), preview)
            print(f"  找到的最好结果画在 {rejected}（红线），没有保存标定")
        print("请换一个球场线清楚、少有人挡的时刻（--time），或用 --hint x,y 指定场地，"
              "或者改用手动标定：badminton calibrate 视频（不加 --auto）")
        return 2
    calib = res.calib
    calib.save(out)
    preview = frames[0].copy()
    draw_court(preview, calib, (0, 255, 0), 2)
    draw_markers(preview, calib)
    cv2.imwrite(str(out.with_suffix(".png")), preview)
    print(f"已保存 {out}（预览图 {out.with_suffix('.png')}，请看一眼绿线是否压在白线上）")
    print(f"置信度 {res.confidence:.0%}（投影的球场线落在白线上的比例），"
          f"对上 {res.lines_found}/{res.lines_visible} 条线，离白线平均 {res.mean_distance_px:.1f} 像素")
    cam_side = calib.to_image(np.array([(0.0, 0.0)]))[0]
    print(f"近端左角在画面 ({cam_side[0]:.0f}, {cam_side[1]:.0f})；如果选错了场地，用 --hint x,y 指定")
    if a.every:
        if not info.frames:
            print("不知道视频总帧数，无法检查漂移")
            return 0
        print(f"机位漂移（每 {a.every:g} 秒对一次线，和第 {a.time:g} 秒比，画面内交点的像素偏移）：")
        worst = 0.0
        t = a.time + a.every
        while int(t * fps) <= last:
            cur = track(read_frame(a.video, int(t * fps)), calib)
            if cur is None:
                print(f"  {t:7.1f}s  找不到球场（机位大幅变化或线被挡住）")
            else:
                mean, mx = intersection_distance_px(calib, cur.calib, in_frame=True)
                worst = max(worst, mx)
                print(f"  {t:7.1f}s  平均 {mean:5.1f}  最大 {mx:5.1f}  置信度 {cur.confidence:.0%}")
            t += a.every
        if worst > 5:
            print(f"最大漂移 {worst:.1f} 像素：建议先运行 badminton stabilize 再标定")
        else:
            print(f"最大漂移 {worst:.1f} 像素：机位稳定")
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
    from .profiling import bench_model, bench_native, default_device_matrix, pipeline
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
        if hasattr(tool, "model"):  # native Core ML: time MLModel.predict on the same input
            r = bench_native(tool.model, inputs[label], a.runs)
            dev = f"coremlnative（{r.device}）"
            print(f"{label:<10}{dev:<42}{r.ms_per_run:8.2f}{'整图':>18}{'1':>6}")
            report["models"].append({"model": label, **r.__dict__})
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print(f"\n已保存 {a.json}")
    return 0


def _tracking(a):
    import numpy as np

    from .court import CourtCalibration
    from .tracking import load_npz, track_players, video_features

    video = Path(a.video)
    data_path = Path(a.data) if a.data else video.with_suffix(".skeleton.npz")
    if not data_path.exists():
        raise SystemExit(f"找不到 {data_path}：先运行 badminton skeleton，或用 --data 指定 .npz")
    data = load_npz(data_path)
    if getattr(a, "start", 0.0):
        keep = data["time_s"] >= a.start
        data = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(keep) else v)
                for k, v in data.items()}
    calib = CourtCalibration.load(a.court)
    if a.singles:
        per_side = 1
    else:  # most people seen on one half in a frame
        counts = [max(((data["side"][i] == s) & data["valid"][i]).sum() for s in (0, 1)) for i in range(len(data["valid"]))]
        per_side = int(np.percentile(counts, 90)) if counts else 2
        per_side = min(2, max(1, per_side))
    features, cuts = (None, None) if a.no_appearance else video_features(video, data)
    if a.cuts:
        cuts = np.zeros(len(data["time_s"]), bool) if cuts is None else cuts
        for c in (float(v) for v in a.cuts.split(",") if v.strip()):
            k = int(np.searchsorted(data["time_s"], c))
            if 0 < k < len(cuts):
                cuts[k] = True
    if cuts is not None and cuts.any():
        print(f"视频里有 {int(cuts.sum())} 处剪辑切换（{', '.join(f'{t:.1f}' for t in data['time_s'][cuts])} 秒），跑动不跨切换点计算")
    tracking = track_players(data, calib, per_side, features, cuts=cuts)
    names_path = Path(a.players) if a.players else video.with_suffix(".players.json")
    if names_path.exists():
        names = {int(k): v for k, v in json.loads(names_path.read_text()).items()}
        for t in tracking.tracks:
            t.name = names.get(t.id, "")
    return tracking, data, names_path


def cmd_label(a) -> int:
    import cv2
    import numpy as np

    from .render import draw_text
    from .video import probe, read_frame

    tracking, data, names_path = _tracking(a)
    if not tracking.tracks:
        print("没有跟踪到球员")
        return 1
    seen = np.stack([t.seen for t in tracking.tracks], axis=1)
    if a.time is not None:
        i = int(np.argmin(np.abs(tracking.time_s - a.time)))
    else:
        i = int(np.argmax(seen.sum(1)))
    frame = read_frame(a.video, int(tracking.frame[i]))
    scale = probe(a.video).height / 1080
    for t in tracking.tracks:
        if t.det_index[i] < 0:
            continue
        x0, y0, x1, y1 = data["bbox"][i, t.det_index[i]].astype(int)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), max(1, int(2 * scale)))
        label = f"{t.id}" + (f" {t.name}" if t.name else "")
        cv2.putText(frame, label, (x0, max(30, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * scale, (0, 0, 0),
                    int(8 * scale), cv2.LINE_AA)
        cv2.putText(frame, label, (x0, max(30, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * scale, (0, 255, 255),
                    int(3 * scale), cv2.LINE_AA)
    draw_text(frame, f"{tracking.time_s[i]:.1f}s")
    png = names_path.with_suffix(".png")
    cv2.imwrite(str(png), frame)
    print(f"编号画在 {png}（第 {tracking.time_s[i]:.1f} 秒）")
    ids = [t.id for t in tracking.tracks]
    if a.names:
        names = {}
        for pair in (x.strip() for x in a.names.replace("，", ",").split(",")):
            k, sep, v = pair.partition("=")
            if not pair:
                continue
            if not sep or not k.strip().isdigit() or not v.strip():
                print(f'看不懂 "{pair}"：格式是 --names "1=小王,2=小李"')
                return 1
            names[int(k)] = v.strip()
    elif sys.stdin.isatty():
        names = {}
        for tid in ids:
            n = input(f"{tid} 号是谁（回车跳过）：").strip()
            if n:
                names[tid] = n
    else:
        print('看一眼图片，然后用 --names "1=名字,2=名字" 再运行一次')
        return 0
    unknown = set(names) - set(ids)
    if unknown:
        print(f"没有这些编号：{sorted(unknown)}；现有编号 {ids}")
        return 1
    names_path.write_text(json.dumps({str(k): v for k, v in sorted(names.items())}, ensure_ascii=False, indent=2))
    print(f"已保存 {names_path}：" + "，".join(f"{k}={v}" for k, v in sorted(names.items())))
    return 0


def cmd_movement(a) -> int:
    from .card import render_card
    from .movement import analyse, json_ready, shuttle_agreement

    tracking, _, _ = _tracking(a)
    result = analyse(tracking)
    video = Path(a.video)
    out_dir = Path(a.out_dir) if a.out_dir else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.shuttle:
        result["shuttle_check"] = shuttle_agreement(a.shuttle, tracking.frame, result["_in_play"])
    print(f"视频 {result['duration_s']:.0f} 秒，切出 {len(result['rallies'])} 个回合，回合内共 {result['in_play_s']:.0f} 秒")
    for p, tr, (pos, _) in zip(result["players"], tracking.tracks, result["_smoothed"]):
        png = render_card(p, pos, result["_in_play"], out_dir / f"{video.stem}.movement.{p['id']}.png")
        print(f"  {p['id']} {p['name']}（{'近端' if p['side'] == 'near' else '远端'}）：{p['distance_m']:.0f} 米，"
              f"最高 {p['max_speed_mps']:.1f} 米/秒，平均 {p['mean_speed_mps']:.1f} 米/秒，冲刺 {p['sprints']} 次，"
              f"腾空/异常 {p['airborne_frames']} 帧 -> {png}")
    if "shuttle_check" in result:
        c = result["shuttle_check"]
        print(f"和 TrackNetV3 对照：{c['agreement_pct']:.0f}% 的帧一致（回合内球在飞 {c['in_play_shuttle_moving_pct']:.0f}%，"
              f"死球时间球在飞 {c['dead_shuttle_moving_pct']:.0f}%）")
    out_json = out_dir / f"{video.stem}.movement.json"
    out_json.write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=1))
    print(f"统计已写入 {out_json}")
    _positioning(tracking, result["_in_play"], video, out_dir)
    return 0


def _positioning(tracking, in_play, video: Path, out_dir: Path) -> list:
    from .positioning import analyse_pairs, json_ready, render_pair_card, summary_lines

    reports = analyse_pairs(tracking, in_play)
    if not reports:
        return []
    print("双打站位：")
    for n, rep in enumerate(reports, 1):
        for line in summary_lines(rep):
            print("  " + line)
        png = out_dir / f"{video.stem}.positioning.{rep['side']}.png"
        render_pair_card(rep, png)
        print(f"  -> {png}")
    out_json = out_dir / f"{video.stem}.positioning.json"
    out_json.write_text(json.dumps([json_ready(r) for r in reports], ensure_ascii=False, indent=1))
    print(f"站位统计已写入 {out_json}")
    return reports


def cmd_positioning(a) -> int:
    from .movement import analyse

    tracking, _, _ = _tracking(a)
    video = Path(a.video)
    out_dir = Path(a.out_dir) if a.out_dir else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = _positioning(tracking, analyse(tracking)["_in_play"], video, out_dir)
    if not reports:
        print("没有哪一边正好有两名球员，站位报告只用于双打")
        return 1
    return 0


def cmd_analyze(a) -> int:
    """Chain the steps through the same CLI entry points, reusing what exists."""
    video = Path(a.video)
    out_dir = Path(a.out_dir or video.parent / f"{video.stem}_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    court = Path(a.court or video.with_suffix(".court.json"))
    if not court.exists():
        if not sys.stdin.isatty():
            raise SystemExit(f"没有 {court}：先运行 badminton calibrate {video}（在窗口里点球场标记点）")
        print("== 1/4 标定球场 ==")
        if main(["calibrate", str(video), "-o", str(court)]) != 0:
            return 1
    else:
        print(f"== 1/4 沿用球场标定 {court} ==")
    data = out_dir / f"{video.stem}.skeleton.npz"
    stale = data.exists() and court.stat().st_mtime > data.stat().st_mtime
    if stale:
        print(f"球场标定 {court} 比骨架数据新，骨架重新跑（球场坐标要用新标定）")
    if data.exists() and not a.force and not stale:
        print(f"== 2/4 沿用骨架数据 {data}（--force 重新跑）==")
    else:
        print("== 2/4 骨架 ==")
        args = ["skeleton", str(video), "--court", str(court), "-o", str(data.with_suffix(".mp4")),
                "--stride", str(a.stride), "--mode", a.mode, "--device", a.device]
        if a.singles:
            args.append("--singles")
        if a.duration:
            args += ["--duration", str(a.duration)]
        if a.det_model:
            args += ["--det-model", a.det_model]
        if a.pose_model:
            args += ["--pose-model", a.pose_model]
        if main(args) != 0:
            return 1
        if a.no_skeleton_video:
            data.with_suffix(".mp4").unlink(missing_ok=True)
    players = out_dir / f"{video.stem}.players.json"
    common = [str(video), "--court", str(court), "--data", str(data), "--players", str(players)]
    if a.singles:
        common.append("--singles")
    if a.cuts:
        common += ["--cuts", a.cuts]
    if players.exists() and not a.names:
        print(f"== 3/4 沿用球员名字 {players} ==")
    else:
        print("== 3/4 球员编号 ==")
        if main(["label", *common] + (["--names", a.names] if a.names else [])) != 0:
            return 1
    print("== 4/4 跑动卡片和站位报告 ==")
    if main(["movement", *common, "-o", str(out_dir)]) != 0:
        return 1
    report = {"video": str(video), "court": str(court)}
    for key, name in (("movement", "movement"), ("positioning", "positioning")):
        f = out_dir / f"{video.stem}.{name}.json"
        if f.exists():
            report[key] = json.loads(f.read_text())
    out = out_dir / f"{video.stem}.report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\n全部结果在 {out_dir}/；汇总给 AI 点评用的数据：{out}")
    if not players.exists():
        print(f'看 {players.with_suffix(".png")} 里的编号，用 --names "1=...,2=..." 再运行一次就能带上名字（骨架不会重跑）')
    return 0


def cmd_stabilize(a) -> int:
    import numpy as np

    from .pipeline import print_progress
    from .stabilize import stabilize

    log = stabilize(a.video, a.output, a.ref_time, a.start, a.duration, progress=print_progress,
                    floor_from=a.floor_from)
    segs = log["still_segments"]
    still = sum(b - a + 1 for a, b in segs)
    n = len(log["H"])
    print(f"已写入 {a.output}：{n} 帧，镜头静止的 {len(segs)} 段共 {still} 帧（{100 * still / max(1, n):.0f}%），"
          f"其余 {n - still} 帧是镜头被碰动的过渡")
    return 0


def cmd_shuttle(a) -> int:
    from .pipeline import print_progress
    from .shuttle import track_shuttle, write_csv

    from .video import probe

    video = Path(a.video)
    info = probe(video)
    if info.is_hdr:
        print("注意：这是 HDR 视频，颜色会发灰，球更难找。建议先运行 badminton sdr 转成 SDR")
    if a.frame_step == 1 and info.fps and info.fps > 45:
        print(f"提示：视频是 {info.fps:.0f} fps，TrackNetV3 是用 30 fps 训练的，可以试试 --frame-step 2")
    out = Path(a.output or video.with_suffix(".shuttle.csv"))
    debug = {} if a.court else None
    rows, tm = track_shuttle(video, a.tracknet_dir, a.compute_units, inpaint=not a.no_inpaint,
                             max_frames=a.max_frames, progress=print_progress, resize=a.resize,
                             frame_step=a.frame_step, debug=debug)
    if a.raw_output:
        write_csv(rows, a.raw_output)
    if a.court:
        from .court import CourtCalibration
        from .shuttle_path import rows_from_candidates

        if a.frame_step != 1:
            raise SystemExit("--court 目前只支持 --frame-step 1")
        rows = rows_from_candidates(debug["candidates"], (info.width, info.height), CourtCalibration.load(a.court))
    write_csv(rows, out)
    seen = sum(r[1] for r in rows)
    print(f"已写入 {out}：{len(rows)} 帧，{seen} 帧有球（{100 * seen / max(1, len(rows)):.1f}%）")
    print(tm.summary())
    if a.overlay:
        from .render import draw_shuttle_video

        ov = out.with_suffix(".mp4")
        draw_shuttle_video(video, rows, ov)
        print(f"轨迹视频 {ov}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
