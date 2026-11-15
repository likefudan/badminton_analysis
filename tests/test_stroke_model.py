import numpy as np
import pytest
from test_bst import _calib, _skeleton
from test_strokes import _Track, _Tracking

from badminton import stroke_model as SM
from badminton import strokes as ST
from badminton.court import LENGTH, WIDTH

TEAM = {"A": 0, "B": 0, "C": 1, "D": 1}
POS = {"A": (1.0, 2.0), "B": (4.0, 5.0), "C": (2.0, 8.0), "D": (5.0, 12.0)}


def _shot(t, hitter, pos=POS):
    return SM.Shot(t, hitter, {k: np.array(v, float) for k, v in pos.items()}, TEAM)


def test_every_class_has_a_name_and_shuttleset_types():
    assert set(SM.NAMES) == set(SM.CLASSES)
    assert set(SM.SHUTTLESET_TO_CLASS) == set(ST.SHUTTLESET_TYPES)
    for zh, group, same in SM.NAMES.values():
        assert zh in {n for n, _, _ in ST.NAMES.values()} and same


def test_roles_and_mirroring():
    order, flip = SM.roles(_shot(0.0, "A"))
    assert order == ["A", "B", "C", "D"] and not flip  # C is the opponent nearer the net
    order, flip = SM.roles(_shot(0.0, "D"))
    assert order == ["D", "C", "B", "A"] and flip
    assert SM.roles(SM.Shot(0.0, "A", {}, TEAM)) is None
    # the same rally seen from the other end gives the same numbers
    turned = {k: (WIDTH - x, LENGTH - y) for k, (x, y) in POS.items()}
    a = SM.context_features([_shot(0.0, "A"), _shot(1.0, "C")], 0)
    b = SM.context_features([_shot(0.0, "A", turned), _shot(1.0, "C", turned)], 0)
    assert a.shape == (SM.CTX_DIM,)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_neighbours_only_count_from_the_other_side_and_soon_enough():
    shots = [_shot(0.0, "A"), _shot(1.0, "C"), _shot(1.5, "D"), _shot(9.0, "A")]
    f0 = SM.context_features(shots, 0)
    assert f0[12] == 1 and f0[13] == pytest.approx(1.0) and list(f0[27:31]) == [0, 0, 1, 0]
    assert f0[31] == 0 and f0[-1] == 1  # nothing before: the rally starts here
    f1 = SM.context_features(shots, 1)
    assert f1[12] == 0  # the next hit is the partner's: a hit in between was missed
    assert f1[31] == 1 and f1[-1] == 0
    f2 = SM.context_features(shots, 2)
    assert f2[12] == 0 and f2[31] == 0 and f2[-1] == 0  # next is too late, previous is the partner's
    assert SM.context_features(shots, 3)[-1] == 1


def test_mirrors_are_their_own_inverse_and_match_a_mirrored_court():
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(2, SM.SEQ_LEN, SM.SEQ_DIM)).astype(np.float32)
    np.testing.assert_allclose(SM.mirror_sequence(SM.mirror_sequence(seq)), seq)
    assert not np.allclose(SM.mirror_sequence(seq), seq)
    shots = [_shot(0.0, "A"), _shot(1.0, "C")]
    flipped = {k: (WIDTH - x, y) for k, (x, y) in POS.items()}
    np.testing.assert_allclose(SM.mirror_context(SM.context_features(shots, 0)),
                               SM.context_features([_shot(0.0, "A", flipped), _shot(1.0, "C", flipped)], 0), atol=1e-6)


def _scene(F=90):
    skel = _skeleton(F=F)
    tracks = [_Track(1, 0, np.zeros(F, int), np.tile([[4.0, 3.0]], (F, 1))),
              _Track(2, 1, np.ones(F, int), np.tile([[2.0, 10.0]], (F, 1)))]
    for t in tracks:
        t.raw_xy = t.xy
    shuttle = np.full((F, 2), np.nan)
    shuttle[20:40] = (400.0, 700.0)
    return skel, _Tracking(tracks), shuttle


def test_scene_features():
    skel, tracking, shuttle = _scene()
    scene = SM.Scene(skel, tracking, shuttle, _calib())
    feats = SM.hit_features(scene, [{"frame": 30, "player_id": 1}, {"frame": 60, "player_id": 2},
                                    {"frame": 70, "player_id": None}])
    assert feats[2] is None
    a, b = feats[0], feats[1]
    assert a["far"] == 0.0 and b["far"] == 1.0
    assert a["ctx"].shape == (SM.CTX_DIM,) and a["seq"].shape == (SM.SEQ_LEN, SM.SEQ_DIM)
    assert a["ctx"][12] == 1 and a["ctx"][13] == pytest.approx(1.0)  # the far player replies a second later
    assert a["seq"][:, 34].all()  # the hitter's pose is there in every sample
    assert a["seq"][0, 39] == 1 and a["seq"][-1, 39] == 0  # shuttle seen in frames 20..39 only
    np.testing.assert_allclose(a["seq"][0, 40:43], [(4.0 - WIDTH / 2) / 3, (3.0 - LENGTH / 2) / 6, 1], atol=1e-6)
    # the far hitter's own position is turned round to the near half
    np.testing.assert_allclose(b["seq"][0, 40:42], [(WIDTH - 2.0 - WIDTH / 2) / 3, (LENGTH - 10.0 - LENGTH / 2) / 6],
                               atol=1e-6)


def test_model_runs_and_weights_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    skel, tracking, shuttle = _scene()
    feats = SM.hit_features(SM.Scene(skel, tracking, shuttle, _calib()),
                            [{"frame": 30, "player_id": 1}, {"frame": 50, "player_id": None}])
    torch.manual_seed(0)
    clf = SM.StrokeClassifier(SM.build_model())
    p = clf.proba(feats)
    assert p.shape == (2, len(SM.CLASSES)) and p[0].sum() == pytest.approx(1) and np.isnan(p[1]).all()
    assert clf.model(torch.from_numpy(feats[0]["ctx"][None])).shape == (1, len(SM.CLASSES))  # context alone
    clf.save(tmp_path / "w.pt")
    again = SM.StrokeClassifier.load(tmp_path / "w.pt")
    np.testing.assert_allclose(again.proba(feats)[0], p[0], atol=1e-6)
    assert SM.StrokeClassifier.load(tmp_path / "missing.pt") is None


def test_doubles_model_picks_the_class_and_bst_the_type_inside_it():
    pytest.importorskip("torch")
    from badminton import bst

    class Fixed:
        def proba(self, feats):
            out = np.full((len(feats), len(SM.CLASSES)), np.nan)
            for i, f in enumerate(feats):
                if f is not None:
                    out[i] = 0.02
                    out[i, SM.CLASSES.index("lift")] = 0.84
            return out

    skel, tracking, shuttle = _scene()
    model = bst.build_model(n_class=ST.N_CLASSES_FULL).eval()
    hits = [{"frame": 20, "player_id": 1}, {"frame": 50, "player_id": 2}, {"frame": 70, "player_id": None}]
    res = ST.classify_hits(skel, tracking, hits, shuttle, _calib(), [model], 30.0, Fixed())
    assert [r["model"] for r in res] == ["dshuttlenet", "dshuttlenet", "bst"]
    for r in res[:2]:
        assert r["type"] in ("挑球", "防守挑球") and r["prob"] == pytest.approx(0.84) and not r["needs_review"]
        assert len(r["alternatives"]) == 2
    plain = ST.classify_hits(skel, tracking, hits, shuttle, _calib(), [model], 30.0)
    assert {r["model"] for r in plain} == {"bst"}


def test_a_file_that_is_not_weights_loads_as_none(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "junk.pt").write_bytes(b"not a model")
    assert SM.StrokeClassifier.load(tmp_path / "junk.pt") is None
