"""Hit detection on a synthetic rally: two players, a shuttle flying between their rackets."""

import csv
import json

import numpy as np
import pytest

from badminton import hits as H
from badminton import provenance as P

FPS, FRAME_H, N = 30.0, 1080.0, 300
NEAR, FAR = np.array([960.0, 800.0]), np.array([960.0, 380.0])  # racket-hand positions
BH = {0: 300.0, 1: 150.0}  # body heights in pixels: the far player is half the size
HIT_FRAMES = [20, 50, 80, 110, 140, 170]  # near, far, near, ...


def rally(hit_frames=HIT_FRAMES, n=N):
    """Shuttle path: straight flights (with an arc) from one hand to the other, landing after the last hit."""
    xy = np.full((n, 2), np.nan)
    ends = [NEAR if k % 2 == 0 else FAR for k in range(len(hit_frames))]
    for k, (a, b) in enumerate(zip(hit_frames[:-1], hit_frames[1:])):
        t = np.linspace(0, 1, b - a + 1)
        p = ends[k][None] * (1 - t[:, None]) + ends[k + 1][None] * t[:, None]
        p[:, 1] -= 180 * np.sin(np.pi * t)  # arc
        p[:, 0] += 120 * np.sin(np.pi * t)  # out to the side and back: the direction flips at each hit
        xy[a:b + 1] = p
    # before the first hit the shuttle falls towards the near hand; after the last it flies off and lands
    a = hit_frames[0]
    xy[a - 12:a + 1] = NEAR[None] + np.linspace(1, 0, 13)[:, None] * np.array([150.0, -350.0])
    b = hit_frames[-1]
    t = np.linspace(0, 1, 25)
    xy[b:b + 25] = ends[-1][None] + t[:, None] * np.array([-300.0, 420.0 if len(hit_frames) % 2 == 0 else -300.0])
    return xy


def players(hit_frames=HIT_FRAMES, n=N, swing=True):
    out = []
    for side, hand in ((0, NEAR), (1, FAR)):
        w = np.tile(hand, (n, 2, 1)).astype(float)
        w[:, 1, 0] -= 0.25 * BH[side]  # the other wrist rests beside the body
        if swing:
            for k, f in enumerate(hit_frames):
                if k % 2 == side:  # a swing: the racket hand sweeps through its position around the hit
                    for d in range(-3, 4):
                        w[f + d, 0] = hand + np.array([0.0, d * 0.12 * BH[side]])
        out.append(H.PlayerSignals(side + 1, side, w, np.full(n, BH[side]), np.ones(n, bool)))
    return out


def track(xy):
    return H.ShuttleTrack(xy, np.where(np.isfinite(xy[:, 0]), P.MEASURED, "").astype(object))


def frames(hits):
    return [h["frame"] for h in hits]


def test_finds_every_hit_with_the_right_hitter():
    hits = H.detect_hits(track(rally()), players(), FPS, FRAME_H)
    assert len(hits) == len(HIT_FRAMES)
    assert all(abs(a - b) <= 1 for a, b in zip(frames(hits), HIT_FRAMES))
    assert [h["player_id"] for h in hits] == [1, 2, 1, 2, 1, 2]
    assert [h["side"] for h in hits] == ["near", "far"] * 3


def test_landing_is_not_a_hit():
    # the shuttle stops on the floor after the last hit: a turn with nobody near it
    xy = rally()
    last = HIT_FRAMES[-1] + 24
    xy[last:last + 40] = xy[last]
    hits = H.detect_hits(track(xy), players(), FPS, FRAME_H)
    assert max(frames(hits)) <= HIT_FRAMES[-1] + 1


def test_far_side_kink_counts_like_a_near_side_one():
    # the same stroke in half the pixels: scaled by the nearest player's body height both score alike
    sh, pl = track(rally()), players()
    xy = H._interp_gaps(sh.xy, 6)
    score, _ = H.kink_series(sh, FPS, H.local_scale(pl, xy, FRAME_H))
    assert score[HIT_FRAMES[1]] > 1.0 and score[HIT_FRAMES[2]] > 1.0


def test_fast_wrist_turns_a_weak_kink_into_a_hit():
    # a soft far-side shot: the kink alone is below the threshold
    c = {"kink": 0.6, "dist_bh": [2.0, 0.2], "wrist": [0.5, 5.0]}
    assert H.evidence(c, 1, H.Options(use_wrist=False)) < 0 < H.evidence(c, 1, H.Options())
    assert H.evidence(c, 0, H.Options()) == -np.inf  # the other player cannot reach it
    c["wrist"] = [0.5, 0.5]  # nobody swung: no hit
    assert H.evidence(c, 1, H.Options()) < 0


def test_follow_through_of_the_same_player_is_dropped():
    # an extra fast swing of the near player 8 frames after their real hit, shuttle still close
    pl = players()
    f = HIT_FRAMES[2] + 9
    for d in range(-3, 4):
        pl[0].wrists[f + d, 0] = NEAR + np.array([d * 40.0, -d * 30.0])
    hits = H.detect_hits(track(rally()), pl, FPS, FRAME_H)
    assert len(hits) == len(HIT_FRAMES)
    assert [h["player_id"] for h in hits] == [1, 2, 1, 2, 1, 2]


def test_serve_is_found_where_the_track_starts_in_the_hand():
    xy = rally()
    xy[:HIT_FRAMES[0]] = np.nan  # nothing before the first contact
    hits = H.detect_hits(track(xy), players(), FPS, FRAME_H)
    assert abs(hits[0]["frame"] - HIT_FRAMES[0]) <= 1 and hits[0]["player_id"] == 1
    assert hits[0]["by"].startswith("track-start")


def test_short_serve_start_is_not_hidden_by_the_kink_at_its_top():
    # a short serve: the track starts in the server's hand, rises slowly and bends at the top
    # of the serve 6 frames later. That bend used to block the start candidate, and the serve
    # was then found 6 frames late (DShuttleNet, docs/hits.md).
    xy = rally()
    a = HIT_FRAMES[0]
    xy[:a] = np.nan
    xy[a:a + 7] = NEAR[None] + np.linspace(0, 1, 7)[:, None] * np.array([0.0, -0.4 * BH[0]])
    xy[a + 7:a + 10] = xy[a + 6] + np.arange(1, 4)[:, None] * np.array([30.0, -20.0])
    hits = H.detect_hits(track(xy), players(), FPS, FRAME_H)
    assert abs(hits[0]["frame"] - a) <= 1 and hits[0]["player_id"] == 1
    assert hits[0]["by"].startswith("track-start")


def test_drop_static_removes_a_phantom_point():
    xy = rally()
    xy[200:260] = (915.0, 270.0)  # TrackNetV3's "nothing found" position, frame after frame
    sh = H.drop_static(track(xy))
    assert not sh.seen[200:260].any() and sh.seen[HIT_FRAMES[0]]


def test_without_players_only_strong_kinks_count():
    hits = H.detect_hits(track(rally()), [], FPS, FRAME_H)
    assert set(HIT_FRAMES[1:-1]) <= {f + d for f in frames(hits) for d in (-1, 0, 1)}
    assert all("player_id" not in h for h in hits)


def test_lone_hits_between_rallies_are_dropped():
    gap = int(H.RALLY_GAP_S * FPS)
    rally_a = [{"frame": f} for f in (100, 130, 160)]
    pickup = {"frame": 160 + gap + 30}  # the shuttle picked up and tossed back: nobody hits it back
    rally_b = [{"frame": f} for f in (pickup["frame"] + gap + 30, pickup["frame"] + gap + 60)]
    kept = H.drop_lone_hits(rally_a + [pickup] + rally_b, FPS)
    assert frames(kept) == frames(rally_a + rally_b)


def test_two_hits_within_the_rally_gap_are_kept():
    gap = int(H.RALLY_GAP_S * FPS)
    hits = [{"frame": 10}, {"frame": 10 + gap}]  # exactly the gap apart: still one rally
    assert frames(H.drop_lone_hits(hits, FPS)) == [10, 10 + gap]
    assert H.drop_lone_hits([{"frame": 10}], FPS) == []
    assert H.drop_lone_hits([], FPS) == []


def test_score_hits_and_ignore_labels():
    hits = [{"frame": f, "player_id": 1 + k % 2} for k, f in enumerate([10, 41, 70, 100, 131])]
    labels = [{"frame": 10, "player_id": 1, "note": ""}, {"frame": 40, "player_id": 2, "note": ""},
              {"frame": 70, "player_id": 2, "note": ""}, {"frame": 100, "player_id": None, "note": "ignore"},
              {"frame": 160, "player_id": 1, "note": ""}]
    r = H.score_hits(hits, labels, tol_frames=3)
    assert (r["tp"], r["fp"], r["fn"]) == (3, 1, 1)  # 131 is extra, 160 is missed, 100 is ignored
    assert (r["hitter_right"], r["hitter_checked"]) == (2, 3)
    assert r["precision"] == r["recall"] == r["f1"] == 0.75


def test_json_tags_every_value_with_its_source(tmp_path):
    hits = H.detect_hits(track(rally()), players(), FPS, FRAME_H)
    doc = H.hits_json(hits, FPS, "v.mp4", names={"1": "A", "2": "B"})
    assert doc["schema"] == H.SCHEMA and set(doc["sources"]) == set(P.ALL)
    for h in doc["hits"]:
        assert h["frame"]["source"] == P.DERIVED and h["time_s"]["source"] == P.DERIVED
        assert h["shuttle_xy"]["source"] in (P.MEASURED, P.INTERPOLATED)
        assert h["player"]["source"] == P.DERIVED and h["player"]["wrist_source"] == P.MEASURED
        assert h["player"]["name"] in ("A", "B")
    json.dumps(doc)  # serialisable
    H.write_csv(hits, FPS, tmp_path / "h.csv")
    with open(tmp_path / "h.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["frame"]) for r in rows] == frames(hits) and rows[0]["source"] == P.DERIVED
    H.review_page(doc, tmp_path / "h.html", {"1": "A", "2": "B"})
    page = (tmp_path / "h.html").read_text()
    assert "/*DOC*/" not in page and '"fps": 30.0' in page


def test_manual_hits_are_tagged_manual(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("Frame,Player,Note\n10,1,\n40,2,serve\n90,1,ignore\n")
    hits = H.manual_hits(H.load_labels(p), {1: "near", 2: "far"})
    doc = H.hits_json(hits, FPS)
    assert [h["frame"]["value"] for h in doc["hits"]] == [10, 40]
    assert all(h["frame"]["source"] == P.MANUAL and h["player"]["source"] == P.MANUAL for h in doc["hits"])


def test_load_shuttle_reads_source_and_status_columns(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("Frame,Visibility,X,Y,Source\n0,1,10,20,measured\n1,1,12,22,interpolated\n2,0,0,0,\n")
    sh = H.load_shuttle(p)
    assert list(sh.source) == [P.MEASURED, P.INTERPOLATED, ""] and not sh.seen[2]
    q = tmp_path / "labels.csv"
    q.write_text("Frame,Visibility,X,Y,Status\n0,1,10,20,machine\n1,1,12,22,confirmed\n")
    assert list(H.load_shuttle(q).source) == [P.MEASURED, P.MANUAL]


def test_provenance_helpers():
    assert P.tagged(3, P.DERIVED) == {"value": 3, "source": "derived"}
    with pytest.raises(ValueError):
        P.tagged(3, "guess")
    s = P.share([P.MEASURED, P.MEASURED, P.INTERPOLATED, ""])
    assert s[P.MEASURED] == pytest.approx(2 / 3, abs=1e-3) and s[P.MANUAL] == 0


# --- time refinement and the hit / bounce classifier -----------------------------------------


def test_refine_moves_a_hit_to_the_bend_of_the_track():
    xy = rally()
    hits = [{"frame": f + (2 if k == 2 else 0)} for k, f in enumerate(HIT_FRAMES)]  # third hit 2 frames late
    out = H.refine_frames(hits, track(xy), FPS)
    assert abs(out[2]["frame"] - HIT_FRAMES[2]) <= 1
    assert out[2]["moved"] < 0
    assert hits[2]["frame"] == HIT_FRAMES[2] + 2  # input left alone


def test_refine_leaves_a_hit_without_a_clear_bend():
    xy = np.full((N, 2), np.nan)
    xy[:, 0] = np.arange(N) * 5.0  # one straight flight
    xy[:, 1] = 500.0
    out = H.refine_frames([{"frame": 100}], track(xy), FPS)
    assert out[0]["frame"] == 100 and "moved" not in out[0]


def _stump(feature, thr, left, right, classes=("hit", "bounce", "none")):
    """A model of one round: per class a tree that splits once on `feature`."""
    k = H.FEATURES.index(feature)
    trees = [{"feature": [k, 0, 0], "threshold": [thr, 0, 0], "left": [1, 0, 0], "right": [2, 0, 0],
              "value": [0.0, left[j], right[j]], "missing_left": [1, 0, 0], "leaf": [0, 1, 1]}
             for j in range(len(classes))]
    return {"features": list(H.FEATURES), "classes": list(classes), "baseline": [0.0] * len(classes),
            "trees": [trees]}


def test_classifier_trees_are_evaluated_with_numpy():
    m = H.HitClassifier(_stump("kink", 1.0, left=(0.0, 2.0, 0.0), right=(3.0, 0.0, 0.0)))
    X = np.full((3, len(H.FEATURES)), np.nan)
    X[:, 0] = [0.5, 2.0, np.nan]  # kink; NaN goes left
    p = m.proba(X)
    assert np.allclose(p.sum(1), 1)
    assert p[0].argmax() == 1 and p[1].argmax() == 0 and p[2].argmax() == 1


def test_classifier_missing_or_foreign_file_is_none(tmp_path):
    assert H.HitClassifier.load(tmp_path / "none.json") is None
    bad = _stump("kink", 1.0, (0, 0, 0), (0, 0, 0))
    bad["features"] = ["something_else"]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    assert H.HitClassifier.load(tmp_path / "bad.json") is None


def test_candidate_features_have_one_row_per_candidate():
    sh, pl = track(rally()), players()
    cands = H.candidates(sh, pl, FPS, FRAME_H, H.Options())
    X = H.candidate_features(cands, sh, pl, FPS, FRAME_H, 1920.0)
    assert X.shape == (len(cands), len(H.FEATURES))
    k = H.FEATURES.index("dist1")
    assert np.isfinite(X[:, k]).all()


def test_bounce_mode_only_takes_evidence_away():
    sh, pl = track(rally()), players()
    # a model that calls everything a bounce: in bounce mode it lowers every candidate,
    # in full mode it also says "not a hit"; with the default weight both drop the weak ones
    always_bounce = H.HitClassifier(_stump("kink", 1e9, left=(0.0, 5.0, 0.0), right=(0.0, 5.0, 0.0)))
    base = frames(H.detect_hits(sh, pl, FPS, FRAME_H))
    off = frames(H.detect_hits(sh, pl, FPS, FRAME_H, H.Options(model=always_bounce, classifier="off")))
    assert off == base
    low = frames(H.detect_hits(sh, pl, FPS, FRAME_H, H.Options(model=always_bounce, w_model=0.1)))
    assert low == base  # a little evidence taken away does not lose the clear hits
    always_hit = H.HitClassifier(_stump("kink", 1e9, left=(5.0, 0.0, 0.0), right=(5.0, 0.0, 0.0)))
    same = frames(H.detect_hits(sh, pl, FPS, FRAME_H, H.Options(model=always_hit)))
    assert same == base  # bounce mode ignores P(hit)


UP = np.array([0.0, -1.0])  # court_axis of a camera behind the near baseline: far is up in the image


def _side_stump(feature, thr, left, right):
    """A side model of one tree: logit of "near" is `left` at or below `thr` (and when unknown), else `right`."""
    k = H.SIDE_FEATURES.index(feature)
    tree = {"feature": [k, 0, 0], "threshold": [thr, 0, 0], "left": [1, 0, 0], "right": [2, 0, 0],
            "value": [0.0, left, right], "missing_left": [1, 0, 0], "leaf": [0, 1, 1]}
    return H.HitClassifier({"features": list(H.SIDE_FEATURES), "classes": ["near", "far"], "baseline": [0.0],
                            "trees": [[tree]]}, H.SIDE_FEATURES)


def test_side_features_follow_the_flight_along_the_court():
    sh, pl = track(rally()), players()
    cands = [c for c in H.candidates(sh, pl, FPS, FRAME_H, H.Options()) if min(abs(c["frame"] - f) for f in HIT_FRAMES) <= 1]
    X = H.side_features(cands, sh, pl, FPS, FRAME_H, UP)
    assert X.shape == (len(cands), len(H.SIDE_FEATURES))
    a_in, a_out = H.SIDE_FEATURES.index("along_in"), H.SIDE_FEATURES.index("along_out")
    near = np.array([min(range(len(HIT_FRAMES)), key=lambda k: abs(HIT_FRAMES[k] - c["frame"])) % 2 == 0 for c in cands])
    # the near half receives a shuttle flying towards it (down the image) and sends it away; the
    # far half's shuttle comes down out of its arc and goes up into the next, so it moves little
    assert X[near, a_in].max() < X[~near, a_in].min() and X[near, a_out].min() > X[~near, a_out].max()
    assert (X[near, a_in] < -0.5).all() and (X[near, a_out] > 0.5).all()


def test_court_axis_points_from_the_near_half_to_the_far_half():
    class Court:
        def to_image(self, xy):  # a camera behind the near baseline: far (larger y) is higher in the image
            return np.stack([100 * xy[:, 0], 1000 - 50 * xy[:, 1]], 1)

    assert np.allclose(H.court_axis(Court()), UP)
    assert H.court_axis(None) is None


def test_side_model_settles_who_hit_but_makes_no_hits():
    sh, pl = track(rally()), players()
    base = H.detect_hits(sh, pl, FPS, FRAME_H)
    right = _side_stump("along_out", 0.5, left=-4.0, right=4.0)  # near iff the shuttle leaves fast towards the far half
    with_model = H.detect_hits(sh, pl, FPS, FRAME_H, H.Options(side_model=right), axis=UP)
    assert [(h["frame"], h["side"]) for h in with_model] == [(h["frame"], h["side"]) for h in base]
    # a model that is sure of the opposite takes every hit's evidence away from the half that hit:
    # nothing is added, and what is left has less evidence
    wrong = _side_stump("along_out", 0.5, left=4.0, right=-4.0)
    hurt = H.detect_hits(sh, pl, FPS, FRAME_H, H.Options(side_model=wrong, w_side=1.0), axis=UP)
    assert len(hurt) <= len(base) and sum(h["evidence"] for h in hurt) < sum(h["evidence"] for h in base)
    # switched off, or without a calibration, the model is not asked
    for kw, ax in ((dict(side_model=wrong, use_side=False), UP), (dict(side_model=wrong), None)):
        off = H.detect_hits(sh, pl, FPS, FRAME_H, H.Options(**kw), axis=ax)
        assert [(h["frame"], h["side"]) for h in off] == [(h["frame"], h["side"]) for h in base]


def test_side_model_file_is_optional(tmp_path):
    assert H.HitClassifier.load(tmp_path / "none.json", features=H.SIDE_FEATURES) is None
