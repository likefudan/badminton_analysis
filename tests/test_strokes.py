import numpy as np
import pytest
from test_bst import _calib, _skeleton

from badminton import strokes as ST
from badminton.court import LENGTH, WIDTH


class _Track:
    def __init__(self, tid, side, det, xy):
        self.id, self.side = tid, side
        self.det_index = np.asarray(det)
        self.seen = self.det_index >= 0
        self.xy = np.asarray(xy, float)


class _Tracking:
    def __init__(self, tracks):
        self.tracks = tracks


def test_class_order_and_names():
    # BST's unmerged order: far x 17, near x 17, unknown last
    assert ST.N_CLASSES_FULL == 35 and ST.UNKNOWN == 34
    assert ST.type_names(2) == ("杀球", "杀球", "smash")
    assert ST.type_names(17 + 9) == ("吊球", "吊球", "drop")
    assert ST.type_names(34)[0] == "未知"
    assert set(ST.NAMES) == set(ST.SHUTTLESET_TYPES)
    groups = {g for _, g, _ in ST.NAMES.values()}
    assert groups == {"杀球", "吊球", "放网", "勾对角", "推扑", "挑球", "高远/平高", "抽挡", "发球"}


def test_only_the_hitters_classes_count():
    p = np.zeros((2, 35))
    p[:, 2] = 0.5  # far smash
    p[:, 17 + 9] = 0.3  # near drop
    p[:, 34] = 0.2
    far, near = ST.hitter_probs(p, ["far", "near"])
    assert far.argmax() == 2 and far.sum() == pytest.approx(1)
    assert near.argmax() == 9 and near[9] == pytest.approx(0.6)


def test_sample_frames_30_and_60_fps():
    f30 = ST.sample_frames(100, 30.0, 1000)
    assert len(f30) == 30 and f30[0] == 85 and f30[-1] == 114
    f60 = ST.sample_frames(100, 60.0, 1000)
    assert len(f60) == 30 and f60[0] == 70 and np.all(np.diff(f60) == 2)
    assert ST.sample_frames(5, 30.0, 1000)[0] == 0  # cut at the start of the video


def test_short_gaps_are_bridged_long_ones_not():
    a = np.arange(10, dtype=float)[:, None].repeat(2, 1)
    have = np.ones(10, bool)
    have[[2, 3]] = False
    a[[2, 3]] = 0
    have[6:10] = False
    out = ST._bridge(a, have, max_gap=3)
    assert out[:6].all() and not out[6:].any()  # the tail has no data after it
    np.testing.assert_allclose(a[2], [2, 2])


def test_clip_for_hitter_and_opponent():
    skel = _skeleton(F=40)
    rows = np.arange(30)
    near_det, far_det = np.zeros(30, int), np.ones(30, int)  # slot 0 near, slot 1 far
    far_det[10:12] = -1  # far player lost for two samples: bridged
    shuttle = np.full((40, 2), np.nan)
    clip = ST.build_clip(skel, rows, (far_det, near_det), np.arange(30), shuttle, _calib())
    assert clip.pose.shape == (30, 2, 72) and clip.pos.shape == (30, 2, 2) and clip.length == 30
    assert clip.ok_share == 1.0
    np.testing.assert_allclose(clip.pos[11, 0], [2.0 / WIDTH, 1 - 10.0 / LENGTH], atol=1e-6)


def test_player_just_behind_the_baseline_is_kept_and_clipped():
    skel = _skeleton(F=40, near=(3.0, -1.0))  # a metre behind the near baseline
    clip = ST.build_clip(skel, np.arange(30), (np.ones(30, int), np.zeros(30, int)), np.arange(30),
                         np.full((40, 2), np.nan), _calib())
    assert clip.ok_share == 1.0 and clip.pos[0, 1, 1] == pytest.approx(1.0)
    far_out = ST.build_clip(skel, np.arange(30), (np.ones(30, int), np.zeros(30, int)), np.arange(30),
                            np.full((40, 2), np.nan), _calib(), outside_keep=0.01)
    assert far_out.ok_share == 0.0  # BST's own rule drops these frames


def _rally(fps=30.0):
    xy1 = np.tile([[1.0, 2.0]], (200, 1))  # near player, left side
    xy2 = np.tile([[1.5, 12.0]], (200, 1))  # far player, same side: straight
    xy3 = np.tile([[5.0, 11.0]], (200, 1))  # far partner, other side: cross-court
    tracks = [_Track(1, 0, np.zeros(200, int), xy1), _Track(2, 1, np.ones(200, int), xy2),
              _Track(3, 1, np.full(200, 2), xy3)]
    hits = [{"frame": 10, "player_id": 1, "side": "near", "_row": 10},
            {"frame": 40, "player_id": 2, "side": "far", "_row": 40},
            {"frame": 70, "player_id": 1, "side": "near", "_row": 70},
            {"frame": 100, "player_id": 3, "side": "far", "_row": 100}]
    return hits, tracks


def test_direction_and_flat_clear():
    hits, tracks = _rally()
    by_id = {t.id: t for t in tracks}
    a = ST.attributes(hits, 0, 30.0, by_id, "高远球")
    assert a["direction"]["value"] == "直线"
    # 10.0 m in 1.0 s: flat
    assert a["length"]["value"] == "平高" and a["length"]["ground_speed_mps"] == pytest.approx(10.0, abs=0.1)
    assert ST.attributes(hits, 2, 30.0, by_id, "杀球")["direction"]["value"] == "对角"
    assert ST.attributes(hits, 3, 30.0, by_id, "杀球") == {}  # last hit: nothing to tell where it went
    assert ST.attributes(hits, 0, 30.0, by_id, "发短球") == {}


def test_opponent_is_who_returns_else_most_seen():
    hits, tracks = _rally()
    side_of = {1: "near", 2: "far", 3: "far"}
    assert ST.pick_opponent(hits, 0, 30.0, tracks, side_of) == 2
    assert ST.pick_opponent(hits, 2, 30.0, tracks, side_of) == 3
    assert ST.pick_opponent(hits, 1, 30.0, tracks, side_of) == 1
    tracks[2].det_index[:] = -1
    tracks[2].seen[:] = False
    assert ST.pick_opponent(hits, 3, 30.0, tracks, side_of) == 1


def test_classify_hits_end_to_end():
    pytest.importorskip("torch")
    from badminton import bst

    skel = _skeleton(F=90)
    tracks = [_Track(1, 0, np.zeros(90, int), np.tile([[4.0, 3.0]], (90, 1))),
              _Track(2, 1, np.ones(90, int), np.tile([[2.0, 10.0]], (90, 1)))]
    model = bst.build_model(n_class=ST.N_CLASSES_FULL).eval()
    hits = [{"frame": 20, "player_id": 1}, {"frame": 50, "player_id": 2}]
    res = ST.classify_hits(skel, _Tracking(tracks), hits, np.full((90, 2), np.nan), _calib(), [model], 30.0)
    assert [r["side"] for r in res] == ["near", "far"]
    assert [r["opponent_id"] for r in res] == [2, 1]
    # the first hit (3.7 m from the net, returned from 3.3 m) is a short serve by rule
    assert res[0]["model"] == "rule" and res[0]["type"] == "发短球" and not res[0]["needs_review"]
    for r in res[1:]:
        assert 0 <= r["prob"] <= 1 and r["group"] and len(r["alternatives"]) == 2
        assert r["group"] not in ("发球", "未知") and r["type"] != "未知"  # never a serve mid-rally, never "unknown"


def _serve_hits(first_y, reply_y, gap_frames=30):
    tracks = {1: _Track(1, 0, np.zeros(200, int), np.tile([[4.0, first_y]], (200, 1))),
              2: _Track(2, 1, np.ones(200, int), np.tile([[2.0, reply_y]], (200, 1)))}
    hits = [{"frame": 10, "player_id": 1, "side": "near", "_row": 10},
            {"frame": 10 + gap_frames, "player_id": 2, "side": "far", "_row": 10 + gap_frames}]
    return hits, tracks


def test_serve_rule():
    hits, tracks = _serve_hits(4.5, 10.0)  # served 2.2 m from the net, returned 3.3 m from it: short
    assert ST.serve_rule(hits, 0, 30.0, tracks) == 0 and ST.serve_rule(hits, 1, 30.0, tracks) is None
    hits, tracks = _serve_hits(4.5, 12.5)  # returned from the back: long
    assert ST.serve_rule(hits, 0, 30.0, tracks) == 1
    hits, tracks = _serve_hits(0.5, 10.0)  # first hit from the back of the court: the serve was missed
    assert ST.serve_rule(hits, 0, 30.0, tracks) is None
    hits, tracks = _serve_hits(4.5, 10.0, gap_frames=120)  # 4 s later: a new rally, from the back
    assert ST.serve_rule(hits, 1, 30.0, tracks) == 0
    hits[0]["player_id"] = 9  # first hitter not tracked: up to the models
    assert ST.serve_rule(hits, 0, 30.0, tracks) == -1


def _physics_always(kind):
    """A physics model of one round that is sure of `kind` whatever it is shown."""
    from badminton.hits import HitClassifier

    trees = [{"feature": [0], "threshold": [0.0], "left": [0], "right": [0], "value": [10.0 if c == kind else 0.0],
              "missing_left": [1], "leaf": [1]} for c in ST.PHYSICS_CLASSES]
    return HitClassifier({"features": list(ST.PHYSICS_FEATURES), "classes": list(ST.PHYSICS_CLASSES),
                          "baseline": [0.0] * len(ST.PHYSICS_CLASSES), "trees": [trees]}, ST.PHYSICS_FEATURES)


def test_physics_features_measure_the_flight():
    hits, tracks = _rally()
    by_id = {t.id: t for t in tracks}
    data = {"bbox": np.tile([[100.0, 300.0, 180.0, 500.0]], (200, 3, 1))}  # feet at y 500, box 200 px high
    upright = {t.id: np.full(200, 200.0) for t in tracks}
    shuttle = np.full((200, 2), np.nan)
    shuttle[10] = (140, 250)  # 1.25 body heights above the feet: overhead
    shuttle[40] = (140, 400)  # 0.5: underhand
    f = ST.physics_features(hits, 0, 30.0, by_id, upright, data, shuttle)
    # hit 4.7 m from the net, reply hit 5.3 m from it one second later, half a metre across
    assert f == pytest.approx([4.7, 5.3, 1.0, 1.0, 0.5])
    assert ST.physics_features(hits, 1, 30.0, by_id, upright, data, shuttle)[3] == 0.0
    last = ST.physics_features(hits, 3, 30.0, by_id, upright, data, shuttle)
    assert last[0] == pytest.approx(4.3) and np.isnan(last[[1, 2, 4]]).all()  # no reply: only where it was hit from
    # a crouching player's short box does not turn a low shuttle into an overhead one
    data["bbox"][40, 1] = (100, 420, 180, 500)
    assert ST.physics_features(hits, 1, 30.0, by_id, upright, data, shuttle)[3] == 0.0


def test_physics_decides_the_kind_and_bst_the_type_inside_it():
    pytest.importorskip("torch")
    from badminton import bst

    skel = _skeleton(F=120)
    tracks = [_Track(1, 0, np.zeros(120, int), np.tile([[4.0, 3.0]], (120, 1))),
              _Track(2, 1, np.ones(120, int), np.tile([[2.0, 10.0]], (120, 1)))]
    model = bst.build_model(n_class=ST.N_CLASSES_FULL).eval()
    hits = [{"frame": 20, "player_id": 1}, {"frame": 50, "player_id": 2}, {"frame": 80, "player_id": 1}]
    args = (skel, _Tracking(tracks), hits, np.full((120, 2), np.nan), _calib(), [model], 30.0)
    plain = ST.classify_hits(*args)
    lifted = ST.classify_hits(*args, physics=_physics_always("lift"))
    assert lifted[0]["model"] == "rule" and lifted[0]["physics"] is None  # the serve rule comes first
    for a, b in zip(plain[1:], lifted[1:]):
        assert a["model"] == "bst" and b["model"] == "bst+physics"
        assert b["group"] == "挑球" and b["type"] in ("挑球", "防守挑球") and b["physics"]["kind"] == "挑球"
        assert b["prob"] > 0.5 or b["alternatives"][0]["type"] in ("挑球", "防守挑球")
    assert ST.classify_hits(*args, physics=_physics_always("lift"), w_physics=0.0)[1]["model"] == "bst"


def test_physics_file_is_optional(tmp_path):
    import json

    assert ST.load_physics(tmp_path / "none.json") is None
    other = {"features": list(ST.PHYSICS_FEATURES), "classes": ["hit", "bounce"], "baseline": [0.0], "trees": []}
    (tmp_path / "other.json").write_text(json.dumps(other))
    assert ST.load_physics(tmp_path / "other.json") is None  # some other model's file
