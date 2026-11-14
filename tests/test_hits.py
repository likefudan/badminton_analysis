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
