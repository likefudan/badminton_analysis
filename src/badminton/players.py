"""Pick the players on our court out of everyone the detector found."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .court import LENGTH, NET_Y, WIDTH, CourtCalibration, in_play_area
from .pose import PoseResult

L_ANKLE, R_ANKLE = 15, 16
_SHOULDERS = (5, 6)
NEAR, FAR = 0, 1
# A player in the air. The floor homography puts the feet of a jumping player further from the
# camera than they are; at the far end of the court, where a few pixels are metres, that is far
# behind the baseline, and the player hitting a jump smash from the back was dropped as
# "someone off the court". The floor under a jumper is lower in the picture than the feet, by up
# to JUMP_FRAC of the box height (~0.8 m for a 1.7 m player)...
JUMP_FRAC = 0.5
JUMP_STEPS = 5
# ...and standing there they must still be as tall as a person: someone further away (the next
# courts behind a low phone camera) is too small for a place on our court. Heights are in
# metres of floor at that place (image height / floor pixels per metre): shoulders to ankles
# of a standing player measure 1.1-1.5 that way, a whole box 1.5-2.
MIN_BODY_M = 1.1
MIN_BOX_M = 1.4
# Even so, from a low camera someone on the next court can pass for "in the air over ours".
# A real jumper stood on the court a moment ago, at about the same place in the picture:
JUMP_MEMORY_S = 1.0  # a jump with its run-up and landing
JUMP_IOU = 0.3  # overlap of the airborne box with a box that stood on the court within JUMP_MEMORY_S


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


def floor_scale(calib: CourtCalibration, xy: np.ndarray) -> np.ndarray:
    """(N,) image pixels per metre of floor (across the court) at court points `xy`."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    a, b = calib.to_image(xy - (0.5, 0.0)), calib.to_image(xy + (0.5, 0.0))
    return np.hypot(*(b - a).T)


def landing_points(calib: CourtCalibration, feet: np.ndarray, boxes: np.ndarray, height_px: np.ndarray,
                   min_height_m: float, side_margin: float, end_margin: float) -> np.ndarray:
    """(N, 2) court point under each person if they are in the air over the play area, NaN otherwise.

    `feet` are image points, `height_px` the image height that must measure at least
    `min_height_m` at the landing point. The smallest lift (up to JUMP_FRAC box heights)
    that brings the point into the play area is taken."""
    out = np.full((len(feet), 2), np.nan)
    box_h = boxes[:, 3] - boxes[:, 1]
    for frac in np.linspace(JUMP_FRAC / JUMP_STEPS, JUMP_FRAC, JUMP_STEPS):
        low = np.asarray(feet, dtype=np.float64).copy()
        low[:, 1] += frac * box_h
        xy = calib.to_court(low)
        with np.errstate(invalid="ignore", divide="ignore"):
            tall = height_px / floor_scale(calib, xy) >= min_height_m
        ok = np.isnan(out[:, 0]) & in_play_area(xy, side_margin, end_margin) & tall
        out[ok] = xy[ok]
    return out


def _body_px(pose: PoseResult, kpt_thr: float = 0.3) -> np.ndarray:
    """(N,) shoulders-to-ankles height in pixels, NaN when either is not seen."""
    out = np.full(len(pose), np.nan)
    for i in range(len(pose)):
        k, s = pose.keypoints[i], pose.scores[i]
        sh = [j for j in _SHOULDERS if s[j] >= kpt_thr]
        an = [j for j in (L_ANKLE, R_ANKLE) if s[j] >= kpt_thr]
        if sh and an:
            out[i] = k[an, 1].mean() - k[sh, 1].mean()
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
    airborne: np.ndarray | None = None  # (N,) kept as "in the air over the court" (feet project off it)


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    w = np.maximum(0, np.minimum(box[2], others[:, 2]) - np.maximum(box[0], others[:, 0]))
    h = np.maximum(0, np.minimum(box[3], others[:, 3]) - np.maximum(box[1], others[:, 1]))
    union = (box[2] - box[0]) * (box[3] - box[1]) + (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]) - w * h
    return np.where(union > 0, w * h / np.maximum(union, 1e-9), 0.0)


def select_players(
    pose: PoseResult,
    calib: CourtCalibration | None,
    players_per_side: int = 2,
    side_margin: float = 1.0,
    end_margin: float = 2.0,
    min_mean_score: float = 0.3,
    stood: np.ndarray | None = None,
) -> Players:
    """Keep at most `players_per_side` people on each half of the court.

    People standing off the court (waiting players, people on the next court,
    spectators) are dropped. If several people stand on one half, the ones
    deepest inside the court and with the most confident pose are kept.
    Without a calibration only the confidence filter is applied.

    `stood` (M, 4): boxes of the players who stood on the court a moment ago
    (PlayerSelector keeps them). With it, someone whose feet project off the
    court because they are in the air over it (landing_points) and whose box
    overlaps one of those is kept too, after everyone standing on the court;
    the recorded position stays the one of the feet (step 2 deals with jumps).
    """
    if len(pose) == 0:
        return Players(pose, np.zeros((0, 2)), np.zeros(0, dtype=int))
    mean_score = pose.scores.mean(1)
    keep = mean_score >= min_mean_score
    if calib is None:
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(-mean_score[idx])][: 2 * players_per_side]
        return Players(pose.select(idx), np.full((len(idx), 2), np.nan), np.full(len(idx), -1))

    feet = foot_points(pose)
    xy = calib.to_court(feet)
    standing = in_play_area(xy, side_margin, end_margin)
    jumping = np.zeros(len(pose), bool)
    landing = xy
    if stood is not None and len(stood) and not standing.all():
        landing = landing_points(calib, feet, pose.bboxes, _body_px(pose), MIN_BODY_M, side_margin, end_margin)
        jumping = ~standing & np.isfinite(landing[:, 0])
        for i in np.flatnonzero(jumping):
            jumping[i] = _iou(pose.bboxes[i], stood).max() >= JUMP_IOU
    keep &= standing | jumping
    side = np.where(xy[:, 1] < NET_Y, NEAR, FAR)
    outside = _distance_outside_court(np.where(jumping[:, None], landing, xy))
    chosen: list[int] = []
    for s in (NEAR, FAR):
        cand = np.flatnonzero(keep & (side == s))
        # standing on the court first; then deepest inside, then most confident
        order = np.lexsort((-mean_score[cand], outside[cand], jumping[cand]))
        chosen.extend(cand[order][:players_per_side].tolist())
    idx = np.array(chosen, dtype=int)
    return Players(pose.select(idx), xy[idx], side[idx], jumping[idx])


class PlayerSelector:
    """select_players frame after frame, remembering where the players stood so that one
    of them leaving the floor (a jump smash) is not taken for someone off the court."""

    def __init__(self, calib: CourtCalibration | None, players_per_side: int = 2):
        self.calib, self.players_per_side = calib, players_per_side
        self._stood: list[tuple[float, np.ndarray]] = []  # (time, box) of players standing on the court

    def box_filter(self, side_margin: float = 1.5, end_margin: float = 2.5) -> Callable[[np.ndarray], np.ndarray]:
        """court_box_filter that also keeps a box in the air over the court where a player just stood."""
        on_court = court_box_filter(self.calib, side_margin, end_margin)

        def keep(boxes: np.ndarray) -> np.ndarray:
            out = on_court(boxes)
            stood = np.array([b for _, b in self._stood]).reshape(-1, 4)
            if len(stood) and not out.all():
                bottom = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
                landing = landing_points(self.calib, bottom, boxes, boxes[:, 3] - boxes[:, 1], MIN_BOX_M,
                                         side_margin, end_margin)
                for i in np.flatnonzero(~out & np.isfinite(landing[:, 0])):
                    out[i] = _iou(boxes[i], stood).max() >= JUMP_IOU
            return out

        return keep

    def __call__(self, pose: PoseResult, time_s: float) -> Players:
        self._stood = [(t, b) for t, b in self._stood if 0 <= time_s - t <= JUMP_MEMORY_S]
        stood = np.array([b for _, b in self._stood]).reshape(-1, 4)
        players = select_players(pose, self.calib, self.players_per_side, stood=stood)
        if players.airborne is not None:
            self._stood += [(time_s, b) for b in players.pose.bboxes[~players.airborne]]
        return players


def court_box_filter(
    calib: CourtCalibration, side_margin: float = 1.5, end_margin: float = 2.5
) -> Callable[[np.ndarray], np.ndarray]:
    """Box filter for PoseEstimator: keep boxes whose bottom centre is near the court.

    Applied before the pose model, so it uses the box bottom instead of the
    ankles; the margins are 0.5 m wider than select_players' so a box whose
    bottom edge sits a little off the feet is not dropped by mistake.
    (PlayerSelector.box_filter also keeps the boxes of players in the air.)
    """

    def keep(boxes: np.ndarray) -> np.ndarray:
        bottom = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
        return in_play_area(calib.to_court(bottom), side_margin, end_margin)

    return keep
