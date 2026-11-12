"""Pick the players on our court out of everyone the detector found."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .court import LENGTH, NET_Y, WIDTH, CourtCalibration, in_play_area
from .pose import PoseResult

L_ANKLE, R_ANKLE = 15, 16
NEAR, FAR = 0, 1


def foot_points(pose: PoseResult, kpt_thr: float = 0.3) -> np.ndarray:
    """Image point where each person touches the floor.

    Midpoint of the ankles when both are confident, the one confident ankle
    otherwise, and the bottom centre of the box as a last resort. (A jump
    lifts the ankles off the floor; step 2 deals with airborne frames.)
    """
    out = np.zeros((len(pose), 2))
    for i in range(len(pose)):
        k, s = pose.keypoints[i], pose.scores[i]
        good = [j for j in (L_ANKLE, R_ANKLE) if s[j] >= kpt_thr]
        if good:
            out[i] = k[good].mean(0)
        else:
            x0, _, x1, y1 = pose.bboxes[i]
            out[i] = ((x0 + x1) / 2, y1)
    return out


def _distance_outside_court(xy: np.ndarray) -> np.ndarray:
    dx = np.maximum(0, np.maximum(-xy[:, 0], xy[:, 0] - WIDTH))
    dy = np.maximum(0, np.maximum(-xy[:, 1], xy[:, 1] - LENGTH))
    return np.hypot(dx, dy)


@dataclass
class Players:
    pose: PoseResult  # only the kept people
    court_xy: np.ndarray  # (N, 2) foot position on the court, metres
    side: np.ndarray  # (N,) NEAR or FAR


def select_players(
    pose: PoseResult,
    calib: CourtCalibration | None,
    players_per_side: int = 2,
    side_margin: float = 1.0,
    end_margin: float = 2.0,
    min_mean_score: float = 0.3,
) -> Players:
    """Keep at most `players_per_side` people on each half of the court.

    People standing off the court (waiting players, people on the next court,
    spectators) are dropped. If several people stand on one half, the ones
    deepest inside the court and with the most confident pose are kept.
    Without a calibration only the confidence filter is applied.
    """
    if len(pose) == 0:
        return Players(pose, np.zeros((0, 2)), np.zeros(0, dtype=int))
    mean_score = pose.scores.mean(1)
    keep = mean_score >= min_mean_score
    if calib is None:
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(-mean_score[idx])][: 2 * players_per_side]
        return Players(pose.select(idx), np.full((len(idx), 2), np.nan), np.full(len(idx), -1))

    xy = calib.to_court(foot_points(pose))
    keep &= in_play_area(xy, side_margin, end_margin)
    side = np.where(xy[:, 1] < NET_Y, NEAR, FAR)
    outside = _distance_outside_court(xy)
    chosen: list[int] = []
    for s in (NEAR, FAR):
        cand = np.flatnonzero(keep & (side == s))
        # deepest inside first, then most confident
        order = np.lexsort((-mean_score[cand], outside[cand]))
        chosen.extend(cand[order][:players_per_side].tolist())
    idx = np.array(chosen, dtype=int)
    return Players(pose.select(idx), xy[idx], side[idx])


def court_box_filter(
    calib: CourtCalibration, side_margin: float = 1.5, end_margin: float = 2.5
) -> Callable[[np.ndarray], np.ndarray]:
    """Box filter for PoseEstimator: keep boxes whose bottom centre is near the court.

    Applied before the pose model, so it uses the box bottom instead of the
    ankles; the margins are 0.5 m wider than select_players' so a box whose
    bottom edge sits a little off the feet is not dropped by mistake.
    """

    def keep(boxes: np.ndarray) -> np.ndarray:
        bottom = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
        return in_play_area(calib.to_court(bottom), side_margin, end_margin)

    return keep
