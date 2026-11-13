import numpy as np
import pytest

from badminton import bst
from badminton.court import LENGTH, MARKERS, WIDTH, CourtCalibration


def _calib():
    """A camera looking straight down: 100 px per metre, far baseline at the top of the image."""
    names = list(MARKERS)
    pts = {n: (100 + 100 * MARKERS[n][0], 100 + 100 * (LENGTH - MARKERS[n][1])) for n in names}
    return CourtCalibration(frame_size=(810, 1540), image_points=pts)


def _to_px(court_xy):
    x, y = court_xy
    return np.array([100 + 100 * x, 100 + 100 * (LENGTH - y)])


def _skeleton(F=40, far=(2.0, 10.0), near=(4.0, 3.0)):
    """Two standing players; slot 0 is the near one (side 0), slot 1 the far one (side 1)."""
    kp = np.zeros((F, 4, 17, 2))
    bbox = np.full((F, 4, 4), np.nan)
    valid = np.zeros((F, 4), bool)
    side = np.full((F, 4), -1)
    for slot, (s, xy) in enumerate(((0, near), (1, far))):
        foot = _to_px(xy)
        kp[:, slot] = foot + np.linspace(-150, 0, 17)[:, None] * (0, 1)  # head at the top, ankles at the feet
        kp[:, slot, 15] = foot - (20, 0)
        kp[:, slot, 16] = foot + (20, 0)
        kp[:, slot, 9] = foot + (-40, -100)
        kp[:, slot, 10] = foot + (40, -100)
        bbox[:, slot] = (foot[0] - 50, foot[1] - 160, foot[0] + 50, foot[1])
        valid[:, slot] = True
        side[:, slot] = s
    return {"fps": np.float64(30), "frame_size": np.array([810, 1540]), "frame": np.arange(F), "valid": valid,
            "bbox": bbox, "keypoints": kp, "scores": np.ones((F, 4, 17)), "side": side}


def test_sequence_is_normalised_like_bst():
    skel = _skeleton()
    shuttle = np.full((40, 2), np.nan)
    shuttle[:30] = (405, 770)
    seq = bst.build_sequence(skel, shuttle, _calib())
    assert seq.ok.all()
    # far player first; x from the left sideline, y from the far baseline, both 0..1 on the doubles court
    np.testing.assert_allclose(seq.pos[0], [[2.0 / WIDTH, 1 - 10.0 / LENGTH], [4.0 / WIDTH, 1 - 3.0 / LENGTH]], atol=1e-6)
    # joints: relative to the box centre, in box diagonals
    diag = np.hypot(100, 160)
    np.testing.assert_allclose(seq.joints[0, 0, 15], [-20 / diag, 80 / diag], atol=1e-9)
    np.testing.assert_allclose(seq.joints[0, 1, 9], [-40 / diag, -20 / diag], atol=1e-9)
    # shuttle: pixels over the video size, zero when not seen
    np.testing.assert_allclose(seq.shuttle[0], [0.5, 0.5])
    assert (seq.shuttle[30:] == 0).all()


def test_frames_without_two_players_in_court_are_zeroed():
    skel = _skeleton()
    skel["valid"][5, 1] = False  # far player not found
    foot = _to_px((3.0, -1.0))  # near player a metre behind the baseline
    skel["keypoints"][7, 0, 15:17] = foot
    seq = bst.build_sequence(skel, np.full((40, 2), 300.0), _calib())
    assert not seq.ok[5] and not seq.ok[7] and seq.ok.sum() == 38
    for a in (seq.joints, seq.pos, seq.shuttle):
        assert not a[5].any() and not a[7].any()


def test_skeleton_must_have_every_frame():
    skel = _skeleton()
    skel["frame"] = np.arange(40) * 2
    with pytest.raises(ValueError, match="stride 1"):
        bst.build_sequence(skel, np.zeros((80, 2)), _calib())


def test_model_inputs_shapes_padding_and_bones():
    skel = _skeleton(F=100)
    seq = bst.build_sequence(skel, np.full((100, 2), 300.0), _calib())
    x = bst.model_inputs(seq, [50, 5])
    assert x["pose"].shape == (2, 30, 2, 72) and x["pos"].shape == (2, 30, 2, 2) and x["shuttle"].shape == (2, 30, 2)
    assert x["length"].tolist() == [30, 20]  # the second clip starts at frame 0: 5 + 15 frames, then padding
    assert not x["pose"][1, 20:].any() and x["pose"][1, 19].any()
    j = x["pose"][0, 0, 0].reshape(36, 2)
    # bone 16: knee (13) -> ankle (15). The knee sits exactly on the box centre line (x = 0), and like BST
    # a coordinate that is exactly zero counts as "missing", so that bone's x stays 0.
    np.testing.assert_allclose(j[17 + 16], [0, j[15, 1] - j[13, 1]], atol=1e-6)
    assert bst.model_inputs(seq, [])["pose"].shape == (0, 30, 2, 72)


def test_longer_clips_are_subsampled_like_bst():
    j, p, s = np.zeros((60, 2, 17, 2)), np.zeros((60, 2, 2)), np.arange(120.0).reshape(60, 2)
    _, _, s2, n = bst.make_seq_len_same(30, j, p, s)
    assert n == 30 and s2[:, 0].tolist() == list(range(0, 120, 4))  # 60 fps: every other frame
    _, _, s3, n = bst.make_seq_len_same(30, j[:50], p[:50], s[:50])  # 50 % 30 > 15: stride 2, 25 frames + padding
    assert n == 25 and s3.shape == (30, 2) and not s3[25:].any()


def test_find_hits_on_a_two_shot_rally():
    """Shuttle flies from the near player's racket to the far player's and back."""
    F = 90
    skel = _skeleton(F=F)
    near_hand = _to_px((4.0, 3.0)) + (40, -130)
    far_hand = _to_px((2.0, 10.0)) + (-40, -130)
    xy = np.empty((F, 2))
    for f in range(F):
        if f < 10:
            xy[f] = near_hand + (0, -30) + (10 - f) * np.array([0, -8.0])  # dropping towards the near player
        elif f < 50:
            xy[f] = near_hand + (far_hand - near_hand) * (f - 10) / 40
        else:
            xy[f] = far_hand + (near_hand - far_hand) * (f - 50) / 60
    seq = bst.build_sequence(skel, xy, _calib())
    hits = bst.find_hits(seq, frame_height=1540)
    assert [(h.frame, h.who) for h in hits] == [(10, "near"), (50, "far")]


def test_class_names():
    assert bst.class_name(0) == (None, "未知")
    assert bst.class_name(3) == ("far", "杀球")
    assert bst.class_name(15, "en") == ("near", "smash")
    assert bst.class_name(24, "raw") == ("near", "發長球")


def test_network_runs_and_respects_clip_length():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = bst.build_model().eval()
    for prm in model.parameters():
        torch.nn.init.normal_(prm, std=0.1)
    x = {"pose": np.random.default_rng(0).normal(size=(3, 30, 2, 72)).astype(np.float32),
         "pos": np.random.default_rng(1).random((3, 30, 2, 2), dtype=np.float32),
         "shuttle": np.random.default_rng(2).random((3, 30, 2), dtype=np.float32),
         "length": np.array([30, 20, 20])}
    x["pose"][2] = x["pose"][1]
    x["pos"][2] = x["pos"][1]
    x["shuttle"][2] = x["shuttle"][1]
    x["shuttle"][2, 25:] += 5.0  # beyond the clip's length: masked out of the attention
    probs = bst.classify([model], x)
    assert probs.shape == (3, bst.N_CLASSES)
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-5)
    assert set(model.state_dict()) >= {"embedding_tem", "tcn_pose.net.0.weight", "mlp_clean.mlp.3.bias",
                                       "cross_trans.cross_attn.to_kv.weight", "mlp_head.mlp.mlp.3.weight"}
    assert model.state_dict()["embedding_tem"].shape == (1, 31, 100)
