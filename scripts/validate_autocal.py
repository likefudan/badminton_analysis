"""Compare automatic court calibration with a hand-made one.

    python scripts/validate_autocal.py VIDEO REFERENCE.court.json --time 1 -o outputs/autocal

Prints the pixel distance between the two calibrations at the 30 crossings of
the painted court lines, how well each lies on the white lines of the frame,
and writes an overlay (automatic = green, reference = red).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from badminton.autocal import AutoCalibrationError, auto_calibrate, intersection_distance_px, mask_agreement
from badminton.court import CourtCalibration
from badminton.render import draw_court
from badminton.video import probe, read_frame


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video")
    p.add_argument("reference")
    p.add_argument("--time", type=float, default=1.0)
    p.add_argument("--hint", help="x,y")
    p.add_argument("-o", "--out-dir", default="outputs/autocal")
    a = p.parse_args()

    frame = read_frame(a.video, int(a.time * (probe(a.video).fps or 30)))
    ref = CourtCalibration.load(a.reference)
    hint = tuple(float(v) for v in a.hint.split(",")) if a.hint else None
    try:
        res = auto_calibrate(frame, hint)
    except AutoCalibrationError as e:
        print(f"自动标定失败：{e}")
        return 2
    mean, worst = intersection_distance_px(res.calib, ref)
    mean_in, worst_in = intersection_distance_px(res.calib, ref, in_frame=True)
    print(f"{Path(a.video).name} @ {a.time:g}s")
    print(f"  自动：置信度 {res.confidence:.0%}，{res.lines_found}/{res.lines_visible} 条线，离白线 {res.mean_distance_px:.1f} px")
    conf, dist = mask_agreement(ref, frame)
    print(f"  参考：线条重合度 {conf:.0%}，离白线 {dist:.1f} px（手点误差 {ref.reprojection_error_px:.1f} px）")
    print(f"  自动 vs 参考，30 个线交点：平均 {mean:.1f} px，最大 {worst:.1f} px"
          f"；只算画面内的：平均 {mean_in:.1f} px，最大 {worst_in:.1f} px")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = frame.copy()
    draw_court(img, ref, (0, 0, 255), 1)
    draw_court(img, res.calib, (0, 255, 0), 1)
    path = out / f"{Path(a.video).stem}.t{a.time:g}.compare.png"
    cv2.imwrite(str(path), img)
    print(f"  对比图 {path}（绿 = 自动，红 = 参考）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
