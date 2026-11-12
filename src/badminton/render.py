"""Drawing: court lines and skeletons on video frames."""

from __future__ import annotations

import cv2
import numpy as np

from .court import CourtCalibration, court_lines
from .players import FAR, NEAR, Players

# COCO-17 skeleton
EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # legs, hips
    (5, 11), (6, 12), (5, 6),  # torso
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),  # head
]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT = {2, 4, 6, 8, 10, 12, 14, 16}

# BGR. Each half of the court gets its own colours; left/right limbs differ in shade.
SIDE_COLORS = {
    NEAR: {"left": (255, 170, 60), "right": (255, 90, 20), "mid": (255, 220, 160)},
    FAR: {"left": (60, 200, 255), "right": (20, 120, 255), "mid": (160, 230, 255)},
    -1: {"left": (80, 255, 80), "right": (20, 180, 20), "mid": (200, 255, 200)},
}


def draw_court(frame: np.ndarray, calib: CourtCalibration, color=(255, 255, 255), thickness: int = 1) -> None:
    for a, b in court_lines():
        # Sample along each line so lines stay correct if they leave the image.
        pts = calib.to_image(np.linspace(a, b, 20)).round().astype(np.int32)
        cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)


def draw_markers(frame: np.ndarray, calib: CourtCalibration) -> None:
    """Clicked points (circles) next to where the fitted court puts them (crosses)."""
    from .court import MARKERS

    for name, p in calib.image_points.items():
        q = calib.to_image(np.array([MARKERS[name]]))[0]
        cv2.circle(frame, tuple(int(v) for v in p), 6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, tuple(int(v) for v in q), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)


def draw_players(frame: np.ndarray, players: Players, kpt_thr: float = 0.3, scale: float = 1.0) -> None:
    pose = players.pose
    t = max(1, int(round(2 * scale)))
    r = max(2, int(round(3 * scale)))
    for i in range(len(pose)):
        c = SIDE_COLORS.get(int(players.side[i]), SIDE_COLORS[-1])
        k, s = pose.keypoints[i], pose.scores[i]
        for a, b in EDGES:
            if s[a] < kpt_thr or s[b] < kpt_thr:
                continue
            col = c["left"] if a in LEFT and b in LEFT else c["right"] if a in RIGHT and b in RIGHT else c["mid"]
            cv2.line(frame, _pt(k[a]), _pt(k[b]), col, t, cv2.LINE_AA)
        for j in range(17):
            if s[j] >= kpt_thr:
                cv2.circle(frame, _pt(k[j]), r, (255, 255, 255), -1, cv2.LINE_AA)


def draw_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = roi
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (128, 128, 128), 1)


def draw_text(frame: np.ndarray, text: str, org=(12, 28)) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))
