"""Click court markers on a video frame to calibrate the court.

Window keys:
  left click   place the point that is being asked for
  n / space    skip this point (not visible, or hard to see)
  u            undo the last point
  enter / q    finish (needs at least 4 points; 6+ recommended)
  esc          cancel

A magnified view of the area around the cursor is shown in the top-left
corner to help put the point exactly on the line intersection.
"""

from __future__ import annotations

import numpy as np

from .court import MARKER_LABELS_ZH, MARKERS, CourtCalibration


def parse_points(spec: str) -> dict[str, tuple[float, float]]:
    """'near_left_corner=100,900;near_right_corner=1800,905;...' -> dict."""
    out = {}
    for item in filter(None, (s.strip() for s in spec.replace("\n", ";").split(";"))):
        name, xy = item.split("=")
        x, y = (float(v) for v in xy.split(","))
        out[name.strip()] = (x, y)
    return out


def click_points(frame: np.ndarray, max_display_width: int = 1600) -> dict[str, tuple[float, float]] | None:
    import cv2

    h, w = frame.shape[:2]
    scale = min(1.0, max_display_width / w)
    base = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame.copy()
    names = list(MARKERS)
    points: dict[str, tuple[float, float]] = {}
    skipped: list[str] = []
    state = {"i": 0, "mouse": (0, 0)}
    win = "court calibration"

    def current() -> str | None:
        return names[state["i"]] if state["i"] < len(names) else None

    def announce() -> None:
        n = current()
        if n:
            print(f"  请点：{MARKER_LABELS_ZH[n]}（{n}），看不到就按 n 跳过")
        else:
            print("  所有点都问完了，按 Enter 完成，或 u 撤销")

    def on_mouse(event, x, y, flags, param):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and current():
            points[current()] = (x / scale, y / scale)
            state["i"] += 1
            announce()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    announce()
    try:
        while True:
            img = base.copy()
            for n, (x, y) in points.items():
                cv2.circle(img, (int(x * scale), int(y * scale)), 5, (0, 0, 255), 2, cv2.LINE_AA)
            if len(points) >= 4:
                try:
                    from .render import draw_court

                    shown = {k: (x * scale, y * scale) for k, (x, y) in points.items()}
                    draw_court(img, CourtCalibration(shown, (img.shape[1], img.shape[0])), (0, 255, 0), 1)
                except ValueError:
                    pass
            _loupe(img, base, state["mouse"])
            n = current()
            label = f"click: {n}" if n else "done? press Enter"
            cv2.putText(img, label, (10, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, label, (10, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(win, img)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                return None
            if key in (13, 10, ord("q")):
                if len(points) >= 4:
                    return points
                print("  至少需要 4 个点")
            elif key in (ord("n"), ord(" ")) and current():
                skipped.append(current())
                state["i"] += 1
                announce()
            elif key == ord("u") and state["i"] > 0:
                state["i"] -= 1
                points.pop(names[state["i"]], None)
                announce()
    finally:
        cv2.destroyWindow(win)


def _loupe(img: np.ndarray, base: np.ndarray, center, radius: int = 24, zoom: int = 5) -> None:
    import cv2

    x, y = center
    h, w = base.shape[:2]
    x0, y0 = max(0, x - radius), max(0, y - radius)
    x1, y1 = min(w, x + radius), min(h, y + radius)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    patch = cv2.resize(base[y0:y1, x0:x1], ((x1 - x0) * zoom, (y1 - y0) * zoom), interpolation=cv2.INTER_NEAREST)
    ph, pw = patch.shape[:2]
    if ph > img.shape[0] or pw > img.shape[1]:
        return
    cx, cy = (x - x0) * zoom, (y - y0) * zoom
    cv2.line(patch, (cx, 0), (cx, ph), (0, 255, 255), 1)
    cv2.line(patch, (0, cy), (pw, cy), (0, 255, 255), 1)
    img[:ph, :pw] = patch
    cv2.rectangle(img, (0, 0), (pw, ph), (255, 255, 255), 1)
