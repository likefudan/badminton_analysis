import json

import numpy as np

from badminton.card import render_card
from badminton.court import NET_Y
from badminton.movement import analyse, json_ready, player_stats, savgol_coeffs, segment_rallies, zone_of
from badminton.tracking import Track, Tracking

FPS = 15.0


def make_track(xy, side=0, tid=1):
    F = len(xy)
    return Track(tid, side, xy.copy(), xy.copy(), np.ones(F, bool), np.zeros(F, bool), np.zeros(F, int))


def test_savgol_fits_polynomials_exactly():
    x = np.arange(-3, 4, dtype=float)
    y = 2 + 3 * x - 0.5 * x**2
    assert np.isclose(np.dot(savgol_coeffs(7, 2, 0)[::-1], y), 2.0)
    assert np.isclose(np.dot(savgol_coeffs(7, 2, 1)[::-1], y), 3.0)


def test_stats_on_known_trajectory():
    # 2 s walking 1 m/s across, 1 s sprint 4 m/s back towards the baseline, 2 s standing
    t = np.arange(0, 5, 1 / FPS)
    xy = np.zeros((len(t), 2))
    for i, ti in enumerate(t):
        if ti < 2:
            xy[i] = (1 + ti, 5.0)
        elif ti < 3:
            xy[i] = (3.0, 5.0 - 4 * (ti - 2))
        else:
            xy[i] = (3.0, 1.0)
    s = player_stats(make_track(xy), t, np.ones(len(t), bool), [(0, len(t))])
    assert abs(s.distance_m - 6.0) < 0.2
    assert abs(s.max_speed_mps - 4.0) < 0.3
    assert s.sprints == 1
    assert abs(s.in_play_s - (t[-1] - t[0])) < 0.01
    assert s.zones_pct["back"] > s.zones_pct["front"]


def test_zones_are_from_the_players_view():
    near = np.array([[1.0, NET_Y - 0.5], [5.0, 0.5]])
    zone, lat = zone_of(near, 0)
    assert zone.tolist() == [0, 2] and lat.tolist() == [0, 1]
    far = np.array([[1.0, NET_Y + 0.5], [5.0, 13.0]])
    zone, lat = zone_of(far, 1)
    assert zone.tolist() == [0, 2] and lat.tolist() == [1, 0]  # far player's left is the camera's right


def test_rallies_from_motion():
    t = np.arange(0, 30, 1 / FPS)
    speed = np.full(len(t), 0.5)
    speed[(t > 2) & (t < 10)] = 3.0
    speed[(t > 5) & (t < 6)] = 0.8  # short lull inside a rally
    speed[(t > 15) & (t < 25)] = 2.5
    speed[(t > 28.5) & (t < 29)] = 3.0  # a few quick steps well after the rally
    play = segment_rallies(speed[:, None], t)
    runs = np.flatnonzero(np.diff(play.astype(int)))
    assert len(runs) == 4  # two rallies
    assert play[(t > 3) & (t < 9)].all()
    assert not play[(t > 11) & (t < 14)].any() and not play[t > 28].any()
    cut = np.zeros(len(t), bool)
    cut[np.searchsorted(t, 7.0)] = True
    play = segment_rallies(speed[:, None], t, cut)
    assert len(np.flatnonzero(np.diff(play.astype(int)))) == 6  # the cut splits the first rally


def test_analyse_and_card(tmp_path):
    t = np.arange(0, 12, 1 / FPS)
    x = 3.05 + 2.0 * np.sin(t * 1.5)  # side to side, ~3 m/s peaks
    near = make_track(np.stack([x, np.full(len(t), 3.0)], 1), 0, 1)
    far = make_track(np.stack([np.full(len(t), 3.0), 10.0 + 1.5 * np.sin(t)], 1), 1, 2)
    far.name = "小李"
    res = analyse(Tracking(t, np.arange(len(t)), [near, far]))
    assert len(res["rallies"]) == 1 and res["players"][1]["name"] == "小李"
    json.dumps(json_ready(res), ensure_ascii=False)
    png = render_card(res["players"][0], res["_smoothed"][0][0], res["_in_play"], tmp_path / "card.png")
    assert png.exists() and png.stat().st_size > 10_000
