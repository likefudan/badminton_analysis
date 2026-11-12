"""Badminton court model and image <-> court homography.

Court coordinates are in metres, seen from above:
  x: across the court, 0 = left doubles sideline, 6.10 = right doubles sideline
     ("left" as seen from the camera, which stands behind the near baseline)
  y: along the court, 0 = near baseline (camera side), 13.40 = far baseline
The net is at y = 6.70.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LENGTH = 13.40
WIDTH = 6.10
NET_Y = LENGTH / 2
SINGLES_INSET = 0.46  # singles sideline is 0.46 m inside the doubles sideline
SHORT_SERVICE = 1.98  # short service line distance from the net
LONG_SERVICE_DOUBLES = 0.76  # doubles long service line distance from the baseline

_NEAR_SHORT = NET_Y - SHORT_SERVICE
_FAR_SHORT = NET_Y + SHORT_SERVICE
_NEAR_LONG = LONG_SERVICE_DOUBLES
_FAR_LONG = LENGTH - LONG_SERVICE_DOUBLES
_CX = WIDTH / 2

# Named floor points that are easy to click: line intersections.
# Order = the order the calibration tool asks for them.
MARKERS: dict[str, tuple[float, float]] = {
    "near_left_corner": (0.0, 0.0),
    "near_right_corner": (WIDTH, 0.0),
    "far_right_corner": (WIDTH, LENGTH),
    "far_left_corner": (0.0, LENGTH),
    "near_short_service_left": (0.0, _NEAR_SHORT),
    "near_short_service_right": (WIDTH, _NEAR_SHORT),
    "far_short_service_right": (WIDTH, _FAR_SHORT),
    "far_short_service_left": (0.0, _FAR_SHORT),
    "near_short_service_center": (_CX, _NEAR_SHORT),
    "far_short_service_center": (_CX, _FAR_SHORT),
    "near_long_service_left": (0.0, _NEAR_LONG),
    "near_long_service_right": (WIDTH, _NEAR_LONG),
    "far_long_service_right": (WIDTH, _FAR_LONG),
    "far_long_service_left": (0.0, _FAR_LONG),
    "net_left": (0.0, NET_Y),
    "net_right": (WIDTH, NET_Y),
}

MARKER_LABELS_ZH: dict[str, str] = {
    "near_left_corner": "近端左角",
    "near_right_corner": "近端右角",
    "far_right_corner": "远端右角",
    "far_left_corner": "远端左角",
    "near_short_service_left": "近端前发球线 × 左边线",
    "near_short_service_right": "近端前发球线 × 右边线",
    "far_short_service_right": "远端前发球线 × 右边线",
    "far_short_service_left": "远端前发球线 × 左边线",
    "near_short_service_center": "近端前发球线 × 中线",
    "far_short_service_center": "远端前发球线 × 中线",
    "near_long_service_left": "近端双打后发球线 × 左边线",
    "near_long_service_right": "近端双打后发球线 × 右边线",
    "far_long_service_right": "远端双打后发球线 × 右边线",
    "far_long_service_left": "远端双打后发球线 × 左边线",
    "net_left": "网下 × 左边线",
    "net_right": "网下 × 右边线",
}


def court_lines() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """All painted lines as segments in court coordinates."""
    L, W = LENGTH, WIDTH
    s = SINGLES_INSET
    return [
        ((0, 0), (W, 0)),  # near baseline
        ((0, L), (W, L)),  # far baseline
        ((0, 0), (0, L)),  # doubles sidelines
        ((W, 0), (W, L)),
        ((s, 0), (s, L)),  # singles sidelines
        ((W - s, 0), (W - s, L)),
        ((0, _NEAR_SHORT), (W, _NEAR_SHORT)),  # short service lines
        ((0, _FAR_SHORT), (W, _FAR_SHORT)),
        ((0, _NEAR_LONG), (W, _NEAR_LONG)),  # doubles long service lines
        ((0, _FAR_LONG), (W, _FAR_LONG)),
        ((_CX, 0), (_CX, _NEAR_SHORT)),  # centre lines
        ((_CX, _FAR_SHORT), (_CX, L)),
        ((0, NET_Y), (W, NET_Y)),  # net (on the floor)
    ]


@dataclass
class CourtCalibration:
    """Image points clicked for named court markers, and the fitted homography."""

    image_points: dict[str, tuple[float, float]]
    frame_size: tuple[int, int]  # (width, height) of the video frame
    H: np.ndarray = field(init=False)  # image -> court
    H_inv: np.ndarray = field(init=False)  # court -> image
    reprojection_error_px: float = field(init=False)

    def __post_init__(self) -> None:
        unknown = set(self.image_points) - set(MARKERS)
        if unknown:
            raise ValueError(f"未知的球场标记点: {sorted(unknown)}")
        if len(self.image_points) < 4:
            raise ValueError("至少需要 4 个标记点（建议 6 个以上）")
        names = list(self.image_points)
        img = np.array([self.image_points[n] for n in names], dtype=np.float64)
        court = np.array([MARKERS[n] for n in names], dtype=np.float64)
        # Plain least squares over all points (no RANSAC): a mis-click should show
        # up as a high reprojection error rather than be silently dropped.
        H = _fit(names, self.image_points)
        if H is None:
            raise ValueError("标记点共线或重复，无法计算单应矩阵")
        self.H = H
        self.H_inv = np.linalg.inv(H)
        projected = court_to_image(self.H_inv, court)
        self.reprojection_error_px = float(np.sqrt(((projected - img) ** 2).sum(1)).mean())

    def per_point_error_px(self) -> dict[str, float]:
        """Distance between each clicked point and where the fitted court puts it."""
        return {name: _residuals(self.H_inv, [name], self.image_points)[0] for name in self.image_points}

    def suspect_point(self) -> tuple[str, float] | None:
        """The click most likely to be wrong, and the mean error once it is left out.

        A least-squares fit spreads one bad click over every point, so the
        largest residual does not reliably name the culprit. Instead each point
        is left out in turn; the one whose removal makes the remaining points
        agree best is the suspect. Needs 6+ points (5 left after removal, so
        the rest still over-determine the homography).
        """
        names = list(self.image_points)
        if len(names) < 6:
            return None
        best = None
        for name in names:
            others = [n for n in names if n != name]
            H = _fit(others, self.image_points)
            if H is None:
                continue
            err = float(np.mean(_residuals(np.linalg.inv(H), others, self.image_points)))
            if best is None or err < best[1]:
                best = (name, err)
        return best

    def to_image(self, court_xy: np.ndarray) -> np.ndarray:
        return court_to_image(self.H_inv, court_xy)

    def to_court(self, image_xy: np.ndarray) -> np.ndarray:
        return image_to_court(self.H, image_xy)

    def save(self, path: str | Path) -> None:
        data = {
            "frame_size": list(self.frame_size),
            "image_points": {k: list(v) for k, v in self.image_points.items()},
            "homography_image_to_court": self.H.tolist(),
            "reprojection_error_px": self.reprojection_error_px,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CourtCalibration":
        data = json.loads(Path(path).read_text())
        pts = {k: (float(v[0]), float(v[1])) for k, v in data["image_points"].items()}
        return cls(pts, tuple(data["frame_size"]))


def _fit(names, image_points) -> np.ndarray | None:
    img = np.array([image_points[n] for n in names], dtype=np.float64)
    court = np.array([MARKERS[n] for n in names], dtype=np.float64)
    H, _ = cv2.findHomography(img, court, 0)
    return H


def _residuals(H_inv, names, image_points) -> list[float]:
    proj = court_to_image(H_inv, np.array([MARKERS[n] for n in names]))
    img = np.array([image_points[n] for n in names])
    return np.hypot(*(proj - img).T).tolist()


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    return cv2.perspectiveTransform(pts[None], H)[0]


def image_to_court(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return _apply_h(H, pts)


def court_to_image(H_inv: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return _apply_h(H_inv, pts)


def in_play_area(court_xy: np.ndarray, side_margin: float = 1.0, end_margin: float = 2.0) -> np.ndarray:
    """True for points on the court or close enough that a player could stand there.

    Players often stand outside the sidelines to retrieve wide shots and well
    behind the baseline for clears, so the area is the court plus margins.
    """
    p = np.asarray(court_xy, dtype=np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return (x >= -side_margin) & (x <= WIDTH + side_margin) & (y >= -end_margin) & (y <= LENGTH + end_margin)


def detection_roi(
    calib: CourtCalibration,
    side_margin: float = 1.0,
    end_margin: float = 2.0,
    head_room_m: float = 2.6,
) -> tuple[int, int, int, int]:
    """Image rectangle (x0, y0, x1, y1) that contains every player on this court.

    Running the person detector on this crop instead of the whole frame makes
    far-court players larger in the detector's input and drops most bystanders.
    The top edge is raised by roughly a standing-and-jumping person's height at
    the far end; the floor homography cannot see height, so the far baseline's
    pixel width is used as the scale.
    """
    w, h = calib.frame_size
    corners = np.array(
        [
            (-side_margin, -end_margin),
            (WIDTH + side_margin, -end_margin),
            (WIDTH + side_margin, LENGTH + end_margin),
            (-side_margin, LENGTH + end_margin),
        ]
    )
    img = calib.to_image(corners)
    far = calib.to_image(np.array([(0.0, LENGTH), (WIDTH, LENGTH)]))
    px_per_m_far = float(np.hypot(*(far[1] - far[0]))) / WIDTH
    x0, y0 = img.min(0)
    x1, y1 = img.max(0)
    y0 -= head_room_m * px_per_m_far
    pad = 0.03 * (x1 - x0)
    x0, x1 = x0 - pad, x1 + pad
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return 0, 0, w, h
    return x0, y0, x1, y1
