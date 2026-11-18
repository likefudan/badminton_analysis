import json

import numpy as np

from badminton.node import use_region
from badminton.region import Region, combine_filters

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


def test_keep_and_drop():
    r = Region.from_dict({"keep": [SQUARE], "drop": [[[0.4, 0.4], [0.6, 0.4], [0.6, 0.6]]]})
    got = r.contains([0.1, 0.3, 0.55, 0.9], [0.5, 0.3, 0.45, 0.5])
    assert got.tolist() == [False, True, False, False]


def test_drop_only_keeps_the_rest():
    r = Region.from_dict({"drop": [SQUARE]})
    assert r.contains([0.1, 0.5], [0.1, 0.5]).tolist() == [True, False]


def test_bad_input_is_ignored():
    assert Region.from_dict(None) is None
    assert Region.from_dict({"keep": [[[0, 0], [1, 1]]], "drop": "x"}) is None
    r = Region.from_dict({"keep": [[[0, 0], [2, 0], [1, 1]]]})
    assert r.keep[0].max() == 1.0


def test_filters_in_pixels():
    r = Region.from_dict({"keep": [SQUARE]})
    cands = [np.array([[256, 144, 0.9], [10, 10, 0.8]]), np.zeros((0, 3))]
    out = r.filter_points(cands, (512, 288))
    assert out[0].tolist() == [[256, 144, 0.9]] and len(out[1]) == 0
    boxes = np.array([[900, 300, 1000, 800], [100, 100, 200, 1000]], float)  # feet at (950, 800), (150, 1000)
    keep = r.box_filter((1920, 1080))
    assert keep(boxes).tolist() == [True, False]
    both = combine_filters(lambda b: np.array([False, True]), keep)
    assert both(boxes).tolist() == [False, False]
    assert combine_filters(None, None) is None


def test_changed_region_throws_old_results_away(tmp_path):
    out = tmp_path / "analysis"
    out.mkdir()
    (out / "v.skeleton.npz").write_text("x")
    (out / "v.players.json").write_text("{}")
    assert use_region(None, out, "v") is None
    assert (out / "v.skeleton.npz").exists()  # no region before, none now: nothing changes

    path = use_region({"keep": [SQUARE]}, out, "v")
    assert json.loads(path.read_text())["keep"][0][0] == [0.2, 0.2]
    assert not (out / "v.skeleton.npz").exists() and (out / "v.players.json").exists()

    (out / "v.skeleton.npz").write_text("x")
    assert use_region({"keep": [SQUARE]}, out, "v") == path  # same region: results kept
    assert (out / "v.skeleton.npz").exists()

    assert use_region({}, out, "v") is None  # region taken away: redo again
    assert not (out / "v.skeleton.npz").exists() and not path.exists()


def test_shuttle_cli_blanks_frames_outside(tmp_path, monkeypatch):
    import cv2

    from badminton import shuttle
    from badminton.cli import main

    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    for _ in range(3):
        w.write(np.zeros((180, 320, 3), np.uint8))
    w.release()
    rows = [(0, 1, 160, 90), (1, 1, 10, 10), (2, 0, 0, 0)]

    class Tm:
        measured = np.array([True, True, False])

        def summary(self):
            return ""

    monkeypatch.setattr(shuttle, "track_shuttle", lambda *a, **k: (list(rows), Tm()))
    reg = tmp_path / "region.json"
    reg.write_text(json.dumps({"keep": [SQUARE]}))
    out = tmp_path / "s.csv"
    assert main(["shuttle", str(video), "-o", str(out), "--region", str(reg)]) == 0
    lines = out.read_text().splitlines()[1:]
    assert [l.split(",")[1] for l in lines] == ["1", "0", "0"]
