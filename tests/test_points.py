import numpy as np

from badminton import points as P
from badminton.court import MARKERS, CourtCalibration
from conftest import Camera, marker_pixels

FPS = 30.0


def _calib(camera):
    return CourtCalibration(marker_pixels(camera, list(MARKERS)), camera.size)


def _stroke(t, pid, side, group="挑球", review=False):
    return {"frame": int(round(t * FPS)), "time_s": t, "player_id": pid, "side": side, "type": group,
            "group": group, "needs_review": review}


def _hands(camera, n, bends):
    """Two players standing with a hand at 0.45 body heights; `bends`: (player index, first row,
    last row, court point where the hand touches the floor)."""
    frames = np.arange(n)
    hands = []
    for k, side in enumerate(("near", "far")):
        height = np.full(n, 0.45)
        spot = np.tile(camera.project([(3.0, 3.0 if side == "near" else 10.0, 0.8)])[0], (n, 1))
        for who, a, b, xy in bends:
            if who == k:
                height[a:b + 1] = 0.05
                spot[a:b + 1] = camera.project([xy])[0]
        hands.append({"id": k + 1, "side": side, "height": height, "spot": spot, "down": height < P.PICK_HAND_BH})
    return hands, frames


def test_split_rallies_by_pause():
    s = [_stroke(t, 1, "near") for t in (0.0, 1.0, 2.0, 6.0, 7.0)]
    assert [len(g) for g in P.split_rallies(s, FPS)] == [3, 2]


def test_judge_spot_in_out_and_no_call():
    assert P.judge_spot((3.0, 2.0), False) | {"court_xy": 0} == {"court_xy": 0, "half": "near", "outside_m": 0.0, "in": True}
    assert P.judge_spot((0.2, 10.0), True)["in"] is True        # 0.26 m outside the singles line: still in
    assert P.judge_spot((0.2, 10.0), False)["outside_m"] == 0.0  # inside for doubles
    assert P.judge_spot((3.0, 14.0), False)["in"] is None        # 0.6 m long: too close to call
    out = P.judge_spot((3.0, 15.0), False)
    assert out["in"] is False and out["half"] == "far"


def test_spot_vote():
    assert P.spot_vote(P.judge_spot((3.0, 2.0), False)) == ("far", "界内")   # lay in the near half: far won
    assert P.spot_vote(P.judge_spot((3.0, 15.0), False)) == ("far", "出界")  # out behind the far side: far won
    assert P.spot_vote(P.judge_spot((3.0, 14.0), False)) == (None, None)
    assert P.spot_vote(P.judge_spot((3.0, 30.0), False)) == (None, None)     # not this court
    assert P.spot_vote(None) == (None, None)


def test_find_pickup_first_bend_and_its_court_spot(camera):
    hands, frames = _hands(camera, 300, [(1, 100, 110, (4.0, 9.0)), (0, 200, 210, (2.0, 2.0))])
    pick = P.find_pickup(hands, frames, FPS, 60, 300, _calib(camera), False)
    assert pick["player_id"] == 2 and pick["frame"] == 100 and pick["half"] == "far" and pick["in"] is True
    assert np.allclose(pick["court_xy"], (4.0, 9.0), atol=0.05)
    assert P.find_pickup(hands, frames, FPS, 120, 190, _calib(camera), False) is None
    hands, frames = _hands(camera, 300, [(0, 100, 102, (2.0, 2.0))])  # 0.1 s: a lunge, not a pick-up
    assert P.find_pickup(hands, frames, FPS, 60, 300, _calib(camera), False) is None


def test_find_track_end_needs_a_flight_from_the_hit():
    xy = np.full((200, 2), np.nan)
    assert P.find_track_end(xy, 10, 200, FPS) is None
    xy[12:40] = np.stack([np.linspace(100, 300, 28), np.linspace(100, 400, 28)], 1)
    end = P.find_track_end(xy, 10, 200, FPS)
    assert end["frame"] == 39 and end["descending"] and end["flight_s"] == round(29 / FPS, 2)
    xy[12:40, 1] = np.linspace(400, 100, 28)
    assert not P.find_track_end(xy, 10, 200, FPS)["descending"]
    assert P.find_track_end(xy, 0, 5, FPS) is None  # nothing seen before the next rally


def test_error_kind():
    slow = [_stroke(0.0, 2, "far", "高远/平高"), _stroke(1.2, 1, "near")]
    assert P.error_kind(slow, FPS) == "unforced"
    assert P.error_kind([_stroke(0.0, 2, "far", "杀球"), _stroke(1.2, 1, "near")], FPS) == "forced"
    assert P.error_kind([_stroke(0.0, 2, "far", "杀球", review=True), _stroke(1.2, 1, "near")], FPS) == "unforced"
    assert P.error_kind([_stroke(0.0, 2, "far"), _stroke(0.4, 1, "near")], FPS) == "forced"
    assert P.error_kind([_stroke(0.0, 1, "near")], FPS) == "unforced"


class _Scene:
    """judge_rallies with the skeleton replaced by ready-made hand heights."""

    def __init__(self, camera, n, bends, monkeypatch):
        self.calib = _calib(camera)
        hands, frames = _hands(camera, n, bends)
        monkeypatch.setattr(P, "hand_heights", lambda data, tracking: hands)
        self.data = {"frame": frames}
        self.xy = np.full((n, 2), np.nan)

    def judge(self, strokes, singles=False):
        return P.judge_rallies(strokes, self.xy, self.calib, FPS, singles, self.data, object())


def test_rallies_winner_error_and_dead_ball(camera, monkeypatch):
    strokes = (
        # rally 1: near's lift is the last hit, the shuttle is picked up in the near half: near's error (net)
        [_stroke(0.0, 2, "far", "发球"), _stroke(1.0, 1, "near"), _stroke(2.0, 2, "far", "高远/平高"), _stroke(3.2, 1, "near")]
        # the shuttle knocked back and caught: two hits, nobody bends
        + [_stroke(9.0, 1, "near", "高远/平高"), _stroke(10.0, 2, "far", "抽挡")]
        # rally 2: far's smash is the last hit, picked up in the near half: far's winner
        + [_stroke(15.0, 1, "near", "发球"), _stroke(16.0, 2, "far", "挑球"), _stroke(17.0, 1, "near"), _stroke(18.0, 2, "far", "杀球")]
        # rally 3: near hits last, the shuttle lies 1.6 m behind the far baseline: near's error (out)
        + [_stroke(25.0, 2, "far", "发球"), _stroke(26.0, 1, "near"), _stroke(27.0, 2, "far"), _stroke(28.0, 1, "near", "高远/平高")]
        # rally 4: nobody seen picking it up, no track: not called
        + [_stroke(35.0, 1, "near", "发球"), _stroke(36.0, 2, "far"), _stroke(37.0, 1, "near")]
    )
    bends = [(0, 150, 160, (3.0, 5.5)), (0, 600, 610, (2.0, 1.0)), (1, 900, 910, (3.0, 15.0))]
    rallies = _Scene(camera, 1300, bends, monkeypatch).judge(strokes)
    assert [r["n"] for r in rallies] == [1, 0, 2, 3, 4] and [r["dead_ball"] for r in rallies] == [False, True, False, False, False]
    r1, _, r2, r3, r4 = rallies
    assert (r1["winner_side"], r1["ending"], r1["error"]) == ("far", "error", {"kind": "unforced", "how": "没过网"})
    assert (r2["winner_side"], r2["ending"], r2["confidence"]) == ("far", "winner", "high")
    assert (r3["winner_side"], r3["ending"], r3["error"]) == ("far", "error", {"kind": "unforced", "how": "出界"})
    assert r4["winner_side"] is None and r4["ending"] is None

    doc = P.points_json(strokes, rallies, FPS, names={"1": "小王"})
    assert (doc["rallies_total"], doc["rallies_judged"], doc["dead_ball_runs"]) == (4, 3, 1)
    assert doc["sides"]["far"] == {"points": 3, "winners": 1, "opponent_errors": 2, "opponent_unforced_errors": 2}
    near, far = doc["players"]
    assert near["name"] == "小王" and near["hits"] == 8 and (near["errors"], near["unforced_errors"]) == (2, 2)
    assert near["errors_by_type"] == {"挑球": 1, "高远/平高": 1}
    assert far["winners"] == 1 and far["winners_by_type"] == {"杀球": 1} and far["hits"] == 7  # the knock-back is not counted
    assert far["strokes"]["发球"] == 2
    lines = P.summary_lines(doc)
    assert "4 个回合" in lines[0] and "小王" in "".join(lines)
    text = "\n".join(P.digest_text(P.digest(doc)))
    assert "回合 2" in text and "制胜球" in text and "主动失误（出界）" in text and "回合 4" not in text


def test_track_end_alone_is_a_guess_and_not_counted(camera, monkeypatch):
    strokes = [_stroke(0.0, 2, "far", "发球"), _stroke(1.0, 1, "near"), _stroke(2.0, 2, "far", "吊球")]
    scene = _Scene(camera, 400, [], monkeypatch)
    a, b = camera.project([(3.0, 9.0, 2.0)])[0], camera.project([(3.0, 4.0, 0.0)])[0]
    scene.xy[62:90] = np.linspace(a, b, 28)
    (r,) = scene.judge(strokes)
    assert (r["winner_side"], r["confidence"], r["ending"]) == ("far", "low", "winner")
    doc = P.points_json(strokes, [r], FPS)
    assert doc["rallies_judged"] == 0 and doc["rallies_guessed"] == 1 and doc["sides"]["far"]["points"] == 0
    assert doc["players"][1]["winners"] == 0
    assert P.digest(doc)["rallies"][0]["winner_side"] is None


def test_shuttle_out_in_the_last_hitters_own_half_is_doubtful(camera, monkeypatch):
    # far "hit last" yet the shuttle lies behind far's own baseline: a later hit by near was missed
    strokes = [_stroke(0.0, 1, "near", "发球"), _stroke(1.0, 2, "far"), _stroke(2.0, 1, "near"), _stroke(3.0, 2, "far")]
    (r,) = _Scene(camera, 400, [(1, 150, 160, (3.0, 15.0))], monkeypatch).judge(strokes)
    assert r["winner_side"] == "far" and r["confidence"] == "low"
