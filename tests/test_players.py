import numpy as np

from badminton.court import CourtCalibration
from badminton.players import FAR, NEAR, foot_points, select_players
from badminton.pose import PoseResult
from conftest import H, W, fake_person, marker_pixels

SIX = ["near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner",
       "near_short_service_left", "near_short_service_right"]


def make_pose(camera, positions, scores=None):
    boxes, kps, scs = [], [], []
    for i, (x, y) in enumerate(positions):
        b, k, s = fake_person(camera, x, y, score=(scores[i] if scores else 0.9))
        boxes.append(b), kps.append(k), scs.append(s)
    return PoseResult(np.array(boxes), np.array(kps), np.array(scs))


def test_foot_point_maps_to_standing_position(camera):
    calib = CourtCalibration(marker_pixels(camera, SIX), (W, H))
    pose = make_pose(camera, [(1.5, 3.0), (4.5, 10.0)])
    np.testing.assert_allclose(calib.to_court(foot_points(pose)), [[1.5, 3.0], [4.5, 10.0]], atol=1e-6)


def test_foot_point_falls_back_to_box_bottom(camera):
    pose = make_pose(camera, [(1.5, 3.0)])
    pose.scores[0, [15, 16]] = 0.0
    fp = foot_points(pose)[0]
    assert fp[1] == pose.bboxes[0, 3]


def test_doubles_drops_bystanders_and_keeps_two_per_side(camera):
    calib = CourtCalibration(marker_pixels(camera, SIX), (W, H))
    positions = [
        (1.5, 2.0), (4.5, 4.5),  # near pair
        (1.5, 9.0), (4.5, 11.5),  # far pair
        (-2.5, 4.0),  # waiting player beside the court
        (3.0, -3.5),  # person behind the camera-side baseline
        (6.6, 3.0),  # someone just outside the sideline (inside margin): a 3rd near person
    ]
    players = select_players(make_pose(camera, positions), calib, players_per_side=2)
    kept = {tuple(np.round(p, 1)) for p in players.court_xy}
    assert kept == {(1.5, 2.0), (4.5, 4.5), (1.5, 9.0), (4.5, 11.5)}
    assert sorted(players.side.tolist()) == [NEAR, NEAR, FAR, FAR]


def test_singles_keeps_one_per_side(camera):
    calib = CourtCalibration(marker_pixels(camera, SIX), (W, H))
    pose = make_pose(camera, [(3.0, 3.0), (3.0, 10.0), (1.0, 5.0)], scores=[0.9, 0.9, 0.5])
    players = select_players(pose, calib, players_per_side=1)
    assert len(players.pose) == 2
    assert {tuple(np.round(p, 1)) for p in players.court_xy} == {(3.0, 3.0), (3.0, 10.0)}


def test_without_calibration_keeps_most_confident(camera):
    pose = make_pose(camera, [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], scores=[0.9, 0.2, 0.8, 0.7, 0.6])
    players = select_players(pose, None, players_per_side=2)
    assert len(players.pose) == 4
    assert np.isnan(players.court_xy).all()


def test_empty(camera):
    assert len(select_players(PoseResult.empty(), None).pose) == 0
