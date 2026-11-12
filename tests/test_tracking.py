import numpy as np

from badminton.court import NET_Y, CourtCalibration
from badminton.movement import player_stats
from badminton.tracking import track_players
from conftest import H, W, fake_person, marker_pixels

SIX = ["near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner",
       "near_short_service_left", "near_short_service_right"]
FPS = 15.0


def calib_for(camera):
    return CourtCalibration(marker_pixels(camera, SIX), (W, H))


def make_data(camera, calib, people_per_frame, seed=0):
    """people_per_frame: list (frames) of lists of (x, y, lift) or None (missed).
    Detections are shuffled per frame, as a detector gives them in no fixed order."""
    rng = np.random.default_rng(seed)
    F, P = len(people_per_frame), 4
    d = {"frame": np.arange(F) * 2, "time_s": np.arange(F) / FPS, "valid": np.zeros((F, P), bool),
         "bbox": np.full((F, P, 4), np.nan), "keypoints": np.full((F, P, 17, 2), np.nan),
         "scores": np.zeros((F, P, 17)), "court_xy": np.full((F, P, 2), np.nan), "side": np.full((F, P), -1)}
    for i, people in enumerate(people_per_frame):
        dets = [p for p in people if p is not None]
        for slot, (x, y, lift) in zip(rng.permutation(len(dets)), dets):
            box, kp, sc = fake_person(camera, x, y, lift=lift)
            d["valid"][i, slot] = True
            d["bbox"][i, slot], d["keypoints"][i, slot], d["scores"][i, slot] = box, kp, sc
            d["court_xy"][i, slot] = calib.to_court(kp[[15, 16]].mean(0, keepdims=True))[0]
            d["side"][i, slot] = 0 if d["court_xy"][i, slot, 1] < NET_Y else 1
    return d


def ids_follow(tracking, truth_xy, tol=0.2):
    """Each true trajectory (F, 2) is followed by one track for all frames it is seen."""
    for true in truth_xy:
        best = min(tracking.tracks, key=lambda t: np.nanmean(np.hypot(*(t.raw_xy - true).T)))
        err = np.hypot(*(best.raw_xy - true).T)
        assert np.nanmax(err) < tol, f"track {best.id} leaves its player (max error {np.nanmax(err):.2f} m)"


def test_doubles_ids_stable_when_partners_cross(camera):
    calib = calib_for(camera)
    F = 45
    t = np.arange(F) / FPS
    a = np.stack([1.0 + 1.3 * t, np.full(F, 3.0)], 1)  # crosses b at t ~ 1.5 s
    b = np.stack([5.0 - 1.3 * t, np.full(F, 3.3)], 1)
    c = np.stack([np.full(F, 1.5), np.full(F, 10.0)], 1)
    e = np.stack([np.full(F, 4.5), 9.0 + 0.5 * t], 1)
    frames = [[(*a[i], 0), (*b[i], 0), (*c[i], 0), (*e[i], 0)] for i in range(F)]
    tr = track_players(make_data(camera, calib, frames), calib, players_per_side=2)
    assert len(tr.tracks) == 4
    assert [t.side for t in tr.tracks] == [0, 0, 1, 1]
    ids_follow(tr, [a, b, c, e])


def test_appearance_keeps_ids_when_players_meet_and_turn_back(camera):
    """Two partners run towards each other, meet and both turn back. Position alone
    cannot tell who is who at the meeting point; shirt colour can."""
    calib = calib_for(camera)
    F = 40
    t = np.arange(F) / FPS
    meet = np.minimum(t, (F - 1) / FPS - t)  # out and back
    a = np.stack([1.2 + 2.0 * meet, np.full(F, 3.0)], 1)
    b = np.stack([4.9 - 2.0 * meet, np.full(F, 3.0)], 1)
    frames = [[(*a[i], 0), (*b[i], 0)] for i in range(F)]
    data = make_data(camera, calib, frames)
    # appearance: player a wears red, b blue (one-hot "histograms")
    feats = np.full((F, 4, 2), np.nan)
    for i in range(F):
        for p in np.flatnonzero(data["valid"][i]):
            is_a = abs(data["court_xy"][i, p, 0] - a[i, 0]) < abs(data["court_xy"][i, p, 0] - b[i, 0])
            feats[i, p] = (1.0, 0.0) if is_a else (0.0, 1.0)
            if abs(a[i, 0] - b[i, 0]) < 0.3:  # at the meeting point the positions coincide
                feats[i, p] = (1.0, 0.0) if p == np.flatnonzero(data["valid"][i])[0] else (0.0, 1.0)
    keep = [i for i in range(F) if abs(a[i, 0] - b[i, 0]) >= 0.3]
    tr = track_players(data, calib, players_per_side=2, features=feats)
    for true in (a, b):
        best = min(tr.tracks, key=lambda tk: np.nanmean(np.hypot(*(tk.raw_xy[keep] - true[keep]).T)))
        assert np.nanmax(np.hypot(*(best.raw_xy[keep] - true[keep]).T)) < 0.2


def test_short_gap_bridged_long_gap_left_open(camera):
    calib = calib_for(camera)
    F = 60
    xs = np.linspace(1.0, 5.0, F)
    frames = []
    for i in range(F):
        missing = 10 <= i < 14 or 30 <= i < 50  # 0.27 s and 1.33 s
        frames.append([None if missing else (xs[i], 3.0, 0.0), (3.0, 10.0, 0.0)])
    tr = track_players(make_data(camera, calib, frames), calib, players_per_side=1)
    near = tr.tracks[0]
    assert near.side == 0 and len(tr.tracks) == 2
    assert np.isfinite(near.xy[10:14]).all()
    np.testing.assert_allclose(near.xy[10:14, 0], xs[10:14], atol=0.05)
    assert np.isnan(near.xy[30:50]).all()
    assert np.isfinite(near.xy[50:]).all()


def test_jump_does_not_add_distance(camera):
    """Running 2 m/s across the court with a 0.4 m jump in the middle: the
    projected foot point flies away from the camera during the jump, but the
    measured distance stays the real one."""
    calib = calib_for(camera)
    F = 45
    t = np.arange(F) / FPS
    x = 1.0 + 2.0 * t
    jump = (t > 1.2) & (t < 1.7)
    lift = np.where(jump, 0.4 * np.sin(np.pi * (t - 1.2) / 0.5), 0.0)
    frames = [[(x[i], 3.0, lift[i]), (3.0, 10.0, 0.0)] for i in range(F)]
    data = make_data(camera, calib, frames)
    raw_y = data["court_xy"][..., 1][data["side"] == 0]
    assert raw_y.max() > 3.4  # without filtering the jump would look like a 0.4+ m excursion

    tr = track_players(data, calib, players_per_side=1)
    near = tr.tracks[0]
    assert near.airborne[jump & (lift > 0.15)].all()
    assert not near.airborne[~jump].any()
    stats = player_stats(near, t, np.ones(F, bool), [(0, F)])
    true_distance = x[-1] - x[0]
    assert abs(stats.distance_m - true_distance) / true_distance < 0.03
    assert stats.max_speed_mps < 2.3


def test_kalman_smoother_tames_far_court_noise():
    """Far from a low camera one pixel spans ~15 cm along the court. With that
    noise in y only, the plain window filter would add metres of fake running;
    the covariance-aware smoother stays close to the true distance."""
    from badminton.movement import smooth_track

    rng = np.random.default_rng(1)
    t = np.arange(0, 20, 1 / FPS)
    true = np.stack([3.0 + 1.5 * np.sin(0.8 * t), 10.0 + 1.2 * np.sin(0.5 * t + 1)], 1)
    noisy = true + np.stack([rng.normal(0, 0.02, len(t)), rng.normal(0, 0.18, len(t))], 1)
    cov = np.tile(np.diag([0.02**2, 0.18**2]), (len(t), 1, 1))
    length = lambda p: np.hypot(*np.diff(p, axis=0).T).sum()  # noqa: E731
    d_true = length(true)
    d_sg = length(smooth_track(noisy, t)[0])
    d_kf = length(smooth_track(noisy, t, cov=cov)[0])
    assert abs(d_kf - d_true) / d_true < 0.1
    assert d_sg > 1.2 * d_true  # what the step-2 filter did with such noise


def test_short_identity_glitch_is_dropped(camera):
    """Step 1 picks a bystander behind the baseline for 0.2 s while the far player
    is missed; the track must not run to the bystander and back."""
    calib = calib_for(camera)
    F = 45
    t = np.arange(F) / FPS
    far_x = 2.0 + 0.5 * t
    frames = []
    for i in range(F):
        far = (-0.3, 15.0, 0.0) if 20 <= i < 23 else (far_x[i], 11.0, 0.0)
        frames.append([(3.0, 3.0, 0.0), far])
    tr = track_players(make_data(camera, calib, frames), calib, players_per_side=1)
    far_track = tr.tracks[1]
    assert not np.isfinite(far_track.raw_xy[20:23]).all() or far_track.airborne[20:23].all()
    assert np.nanmax(np.abs(far_track.xy[:, 1] - 11.0)) < 0.1  # never follows the bystander
    stats = player_stats(far_track, t, np.ones(F, bool), [(0, F)])
    assert stats.max_speed_mps < 1.0


def test_person_parked_behind_the_baseline_is_not_the_player(camera):
    calib = calib_for(camera)
    F = 60
    t = np.arange(F) / FPS
    frames = []
    for i in range(F):
        far = (-0.2, 15.0, 0.0) if 15 <= i < 40 else (2.0 + 0.02 * i, 11.0, 0.0)  # 1.7 s at the same spot
        frames.append([(3.0, 3.0, 0.0), far])
    tr = track_players(make_data(camera, calib, frames), calib, players_per_side=1)
    far_track = tr.tracks[1]
    assert np.isnan(far_track.xy[16:39]).all()
    stats = player_stats(far_track, t, np.ones(F, bool), [(0, F)])
    assert stats.max_speed_mps < 1.0
