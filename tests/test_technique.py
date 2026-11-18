import numpy as np
import pytest

from badminton import technique as TQ

FPS = 60.0
TORSO = 100.0  # px


def _sig(x):
    return 1 / (1 + np.exp(-x))


def swing(peaks_s=(-0.12, -0.08, -0.05, -0.02), contact=1.4, turn_px=40.0, jump_px=0.0, hand="R",
          fps=FPS, score=0.9, origin=(600.0, 500.0)):
    """A synthetic overhead stroke seen from behind: (kp, scores, t).
    peaks_s: when hip, racket shoulder, elbow and wrist each move fastest (s from the hit).
    contact: wrist height above the shoulders at the hit, in torso lengths."""
    t = np.arange(-round(TQ.BEFORE_S * fps), round(TQ.AFTER_S * fps) + 1) / fps
    n = len(t)
    base = {TQ.NOSE: (0, -125), TQ.L_SH: (-40, -100), TQ.R_SH: (40, -100), TQ.L_EL: (-55, -60), TQ.R_EL: (55, -60),
            TQ.L_WR: (-60, -25), TQ.R_WR: (60, -25), TQ.L_HIP: (-25, 0), TQ.R_HIP: (25, 0), TQ.L_KN: (-25, 50),
            TQ.R_KN: (25, 50), TQ.L_AN: (-25, 100), TQ.R_AN: (25, 100), 1: (-8, -130), 2: (8, -130), 3: (-15, -125),
            4: (15, -125)}
    kp = np.zeros((n, 17, 2))
    for j, (x, y) in base.items():
        kp[:, j] = (x, y)
    w = 0.02
    th, ts, te, tw = peaks_s
    s = 1 if hand == "R" else -1
    sh, el, wr = (TQ.R_SH, TQ.R_EL, TQ.R_WR) if hand == "R" else (TQ.L_SH, TQ.L_EL, TQ.L_WR)
    for j in (TQ.L_HIP, TQ.R_HIP):
        kp[:, j, 0] += 15 * _sig((t - th) / w)
    kp[:, sh, 0] -= s * turn_px * _sig((t - ts) / w)  # racket shoulder comes round
    kp[:, el, 1] = -100 - 60 * _sig((t - te) / w)
    kp[:, wr, 1] = -100 - contact * TORSO + 60 * (t / 0.3) ** 2  # highest at the hit
    kp[:, wr, 0] = s * 60 - s * 150 * _sig((t - tw) / w)
    if jump_px:
        bump = np.exp(-((t + 0.05) / 0.12) ** 2)
        kp[:, [TQ.L_AN, TQ.R_AN], 1] -= jump_px * bump[:, None]
    kp += origin
    return kp, np.full((n, 17), score), t


def test_overhead_numbers():
    kp, sc, t = swing(contact=1.4, jump_px=40)
    r = TQ.stroke_metrics(kp, sc, t, FPS, "杀球")
    assert r["ok"] and r["racket_hand"] == "R"
    m = r["metrics"]
    assert m["contact_height"] == pytest.approx(1.4, abs=0.05)
    assert m["chain_order_ok"] is True
    assert m["chain_ms"]["wrist"] == pytest.approx(-20, abs=20)
    assert m["jump"] == pytest.approx(0.4, abs=0.08)
    assert m["turn"] == pytest.approx(0.4, abs=0.05)
    assert m["follow_through"] == pytest.approx(0.4, abs=0.1)  # the rest of the 150 px sweep after the hit
    assert r["prep"].shape == (TQ.PREP_SAMPLES, 17, 2)


def test_chain_out_of_order_is_flagged():
    kp, sc, t = swing(peaks_s=(-0.02, -0.05, -0.08, -0.15))  # wrist first, hip last
    r = TQ.stroke_metrics(kp, sc, t, FPS, "杀球")
    assert r["metrics"]["chain_order_ok"] is False
    notes, _ = TQ.flags_for("杀球", r["metrics"], None)
    assert any("发力顺序" in n for n in notes)


def test_low_contact_and_no_turn_flagged_without_reference():
    kp, sc, t = swing(contact=0.6, turn_px=5)
    r = TQ.stroke_metrics(kp, sc, t, FPS, "高远/平高")
    notes, pct = TQ.flags_for("高远/平高", r["metrics"], None)
    assert any("击球点偏低" in n for n in notes) and any("转体" in n for n in notes) and pct == {}


def test_left_hander():
    kp, sc, t = swing(hand="L")
    r = TQ.stroke_metrics(kp, sc, t, FPS, "杀球")
    assert r["racket_hand"] == "L" and r["metrics"]["contact_height"] == pytest.approx(1.4, abs=0.05)
    assert r["metrics"]["chain_order_ok"] is True
    assert r["metrics"]["follow_through"] == pytest.approx(0.4, abs=0.1)


def test_poorly_seen_stroke_is_skipped():
    kp, sc, t = swing()
    sc[: len(t) // 2, TQ.R_WR] = 0.1  # wrist lost for half the window
    r = TQ.stroke_metrics(kp, sc, t, FPS, "杀球", wrist=TQ.R_WR)
    assert not r["ok"] and "看得清" in r["reason"]


def test_low_stroke_numbers():
    kp, sc, t = swing(turn_px=0)
    kp[:, TQ.L_AN, 0] -= 80  # wide lunge
    r = TQ.stroke_metrics(kp, sc, t, FPS, "放网")
    m = r["metrics"]
    assert m["stance"] == pytest.approx(1.3, abs=0.05)
    assert m["knee_deg"] < 180 and m["trunk_lean"] < 10
    assert "chain_ms" not in m


def test_disguise_compares_drops_with_own_smashes():
    rng = np.random.default_rng(0)
    res = []
    for k in range(6):
        kp, sc, t = swing(contact=1.4 + rng.normal(0, 0.05), turn_px=40 + rng.normal(0, 5))
        kp += rng.normal(0, 2, kp.shape)
        res.append({"player_id": 1, "group": "杀球", **TQ.stroke_metrics(kp, sc, t, FPS, "杀球")})
    kp, sc, t = swing(contact=1.42)
    look_alike = {"player_id": 1, "group": "吊球", **TQ.stroke_metrics(kp, sc, t, FPS, "吊球")}
    kp, sc, t = swing(contact=0.5, peaks_s=(-0.3, -0.3, -0.3, -0.3))
    obvious = {"player_id": 1, "group": "吊球", **TQ.stroke_metrics(kp, sc, t, FPS, "吊球")}
    TQ.add_disguise(res + [look_alike, obvious])
    assert look_alike["metrics"]["disguise"] < 2 < obvious["metrics"]["disguise"]


class _Track:
    def __init__(self, tid, side, det, airborne=None):
        self.id, self.side = tid, side
        self.det_index = np.asarray(det)
        self.airborne = np.zeros(len(det), bool) if airborne is None else airborne


class _Tracking:
    def __init__(self, tracks):
        self.tracks = tracks


def test_analyse_video_and_reference():
    hits = [100, 200, 300, 400]
    n = 500
    keypoints = np.full((n, 2, 17, 2), np.nan)
    scores = np.zeros((n, 2, 17))
    for i, h in enumerate(hits):
        kp, sc, t = swing(contact=1.2 + 0.1 * i)
        f = (h + np.round(t * FPS)).astype(int)
        keypoints[f, 0], scores[f, 0] = kp, sc
    data = {"frame": np.arange(n), "keypoints": keypoints, "scores": scores}
    tracking = _Tracking([_Track(7, 0, np.where(scores[:, 0].any(1), 0, -1)), _Track(9, 1, np.full(n, -1))])
    strokes = [{"frame": h, "player_id": 7, "side": "near", "type": "杀球", "group": "杀球", "needs_review": False}
               for h in hits]
    strokes += [{"frame": 450, "player_id": 9, "side": "far", "type": "杀球", "group": "杀球", "needs_review": False},
                {"frame": 300, "player_id": 7, "side": "near", "type": "发短球", "group": "发球", "needs_review": False}]
    res = TQ.analyse(data, tracking, strokes, FPS)
    assert len(res) == 4 and all(r["ok"] for r in res)  # far player and serve left out
    heights = [r["metrics"]["contact_height"] for r in res]
    np.testing.assert_allclose(heights, [1.2, 1.3, 1.4, 1.5], atol=0.05)
    doc = TQ.technique_json(res, FPS)
    assert "prep" not in doc["strokes"][0]
    assert doc["summary"]["杀球"]["players"]["7"]["n"] == 4

    ref = TQ.build_reference([doc], min_n=4)
    assert ref["groups"]["杀球"]["contact_height"]["p50"] == pytest.approx(1.35, abs=0.05)
    notes, pct = TQ.flags_for("杀球", {"contact_height": 0.9}, ref)
    assert pct["contact_height"] == 0 and "偏低" in notes[0]


def test_analyse_skeleton_run_every_other_frame():
    """--stride 2 skeleton: frames without a skeleton row are not counted as unseen, and speeds stay per second."""
    hits = [100, 201, 300]
    n = 400
    keypoints = np.full((n, 2, 17, 2), np.nan)
    scores = np.zeros((n, 2, 17))
    for h in hits:
        kp, sc, t = swing()
        f = (h + np.round(t * FPS)).astype(int)
        keypoints[f, 0], scores[f, 0] = kp, sc
    full = {"frame": np.arange(n), "keypoints": keypoints, "scores": scores}
    half = {"frame": np.arange(0, n, 2), "keypoints": keypoints[::2], "scores": scores[::2]}
    strokes = [{"frame": h, "player_id": 7, "side": "near", "type": "杀球", "group": "杀球", "needs_review": False}
               for h in hits]
    out = {}
    for name, data in (("full", full), ("half", half)):
        det = np.where(data["scores"][:, 0].any(1), 0, -1)
        res = TQ.analyse(data, _Tracking([_Track(7, 0, det)]), strokes, FPS)
        assert all(r["ok"] for r in res), [r["reason"] for r in res]
        out[name] = res
    for a, b in zip(out["full"], out["half"]):
        assert b["seen_share"] == pytest.approx(1.0)
        assert b["airborne_at_hit"] is False  # looked up at the nearest processed frame
        assert b["metrics"]["contact_height"] == pytest.approx(a["metrics"]["contact_height"], abs=0.05)
        assert b["metrics"]["wrist_speed"] == pytest.approx(a["metrics"]["wrist_speed"], rel=0.2)


def test_steadiness_needs_two_players():
    def r(pid, v):
        return {"ok": True, "group": "杀球", "player_id": pid, "metrics": {"contact_height": v}}
    one = [r(1, v) for v in (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)]
    assert TQ.summary(one)["杀球"]["steadiness"] == {}  # within == overall: always 1.0, says nothing
    two = [r(1, v) for v in (0.5, 0.55, 0.6)] + [r(2, v) for v in (1.4, 1.45, 1.5)]
    assert TQ.summary(two)["杀球"]["steadiness"]["contact_height"] < 0.2


def test_percentile_interpolation():
    ref = {"p10": 1.0, "p25": 2.0, "p50": 3.0, "p75": 4.0, "p90": 5.0}
    assert TQ.percentile_of(3.0, ref) == 50
    assert TQ.percentile_of(4.5, ref) == pytest.approx(82.5)
    assert TQ.percentile_of(9.0, ref) == 100


def test_csv_written(tmp_path):
    kp, sc, t = swing()
    r = {"frame": 1, "time_s": 0.1, "player_id": 1, "side": "near", "type": "杀球", "group": "杀球",
         **TQ.stroke_metrics(kp, sc, t, FPS, "杀球")}
    r["notes"] = ["a", "b"]
    TQ.write_csv([r], tmp_path / "x.csv")
    text = (tmp_path / "x.csv").read_text()
    assert "contact_height" in text and "a；b" in text


def test_left_right_label_flips_do_not_change_the_numbers():
    rng = np.random.default_rng(1)
    kp, sc, t = swing(contact=1.3, jump_px=30)
    clean = TQ.stroke_metrics(kp, sc, t, FPS, "杀球")["metrics"]
    flipped = kp.copy()
    pairs = [(TQ.L_SH, TQ.R_SH), (TQ.L_EL, TQ.R_EL), (TQ.L_WR, TQ.R_WR)]
    whole = rng.random(len(t)) < 0.3  # the model swaps the whole body's labels in these frames
    for a, b in pairs:
        one = rng.random(len(t)) < 0.1  # ... and single joints in these
        f = whole ^ one
        flipped[f, a], flipped[f, b] = kp[f, b], kp[f, a]
    r = TQ.stroke_metrics(flipped, sc, t, FPS, "杀球")
    m = r["metrics"]
    for k in ("contact_height", "wrist_speed", "jump", "backswing_elbow"):
        assert m[k] == pytest.approx(clean[k], abs=0.1), k
    assert m["chain_order_ok"] is True


def test_contact_is_the_top_of_the_swing():
    kp, sc, t = swing(contact=1.4)
    r = TQ.stroke_metrics(kp, sc, t - 0.1, FPS, "杀球")  # hit detected 0.1 s late
    assert r["metrics"]["contact_shift_ms"] == pytest.approx(-100, abs=20)
    assert r["metrics"]["contact_height"] == pytest.approx(1.4, abs=0.05)


def test_lunge_free_arm_reaching_back_is_not_the_racket_arm():
    kp, sc, t = swing(turn_px=0)
    # net shot: racket wrist (R) stays out front and moves a little; free arm (L) flung far back and still
    kp[:, TQ.R_WR] = kp[:, TQ.R_SH] + np.stack([40 + 20 * np.sin(t * 10), 40 + 0 * t], 1)
    kp[:, TQ.L_WR] = kp[:, TQ.L_SH] + (-150, 30)
    r = TQ.stroke_metrics(kp, sc, t, FPS, "放网")
    assert r["racket_hand"] == "R"  # by speed, not by "farther out"
    shuttle = kp[len(t) // 2, TQ.L_WR] + (5, 5)  # ... but the shuttle says otherwise and wins
    assert TQ.stroke_metrics(kp, sc, t, FPS, "放网", shuttle=shuttle)["racket_hand"] == "L"


def test_no_clear_top_keeps_the_detected_hit():
    kp, sc, t = swing(contact=1.4)
    kp[:, TQ.R_WR, 1] = 500 - 100 - 140 - 200 * t  # still rising through the whole search window
    m = TQ.stroke_metrics(kp, sc, t, FPS, "杀球")["metrics"]
    assert m["contact_shift_ms"] == 0 and m["contact_unclear"] is True
