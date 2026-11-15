"""scripts/dshuttlenet_hits.py: DShuttleNet CSVs -> rally/shot labels, hitter by position, hitter scoring."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dshuttlenet_hits as D  # noqa: E402


def _write(path, header, rows):
    path.write_text(",".join(header) + "\n" + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n")


def _raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "match.csv", ["match_id", "video_name", "event"], [[7, "X_vs_Y_Final", 1], [8, "Other", 2]])
    _write(raw / "set.csv", ["set_id", "match_id", "set_num"], [[1, 7, 1], [2, 7, 2], [3, 8, 1]])
    _write(raw / "rally.csv", ["rally_id", "set_id", "rally_num", "start_frame", "end_frame"],
           [[10, 1, 1, 100, 160], [11, 2, 1, 300, 340], [12, 3, 1, 900, 950]])
    head = ["shot_id", "rally_id", "shot_num", "start_frame_num", "end_frame_num", "player", "ball_type",
            "A", "B", "C", "D"] + [f"player_{k}_{c}" for k in "ABCD" for c in "xy"]
    pos = [20, 14, 40, 34, 25, 114, 45, 104]  # A, B near (small y), C, D far
    _write(raw / "shot.csv", head, [
        [1, 10, 1, 100, 120, 5.0, "short serve", 5, 6, 7, 8] + pos,
        [2, 10, 2, 120, 160, 8.0, "push", 5, 6, 7, 8] + pos,
        [3, 11, 1, 300, 340, "", "drive", 5, 6, 7, 8] + pos,
        [4, 12, 1, 900, 950, 5.0, "lift", 5, 6, 7, 8] + pos,  # another match
    ])
    return raw


def test_read_match_letters_sides_and_metres(tmp_path):
    doc = D.read_match(_raw(tmp_path), "X_vs_Y_Final")
    assert doc == D.read_match(tmp_path / "raw", "7")
    assert doc["rallies"] == [(100, 160), (300, 340)]
    shots = doc["shots"]
    assert [s["frame"] for s in shots] == [100, 120, 300]
    assert [s["player"] for s in shots] == ["A", "D", None]
    assert [s["side"] for s in shots] == ["near", "far", None]
    assert shots[0]["positions"]["A"] == (2.0, 1.4)
    assert shots[1]["set"] == 1 and shots[2]["set"] == 2


def test_labels_writes_bfmd_compatible_rallies(tmp_path):
    D.main(["labels", str(_raw(tmp_path)), "--match", "7", "-o", str(tmp_path / "out")])
    doc = json.loads((tmp_path / "out" / "X_vs_Y_Final.rallies.json").read_text())
    assert doc["schema"] == D.SCHEMA and doc["event"] == "MD"
    assert doc["rallies"] == [[100, 160], [300, 340]] and doc["hits"] == [100, 120, 300]
    assert doc["shots"][1]["player"] == "D"


def test_label_hitters_by_position_and_scoring(tmp_path):
    shots = D.read_match(_raw(tmp_path), "7")["shots"]
    feet = {1: np.full((400, 2), np.nan), 2: np.full((400, 2), np.nan)}
    feet[1][100] = (2.2, 1.5)  # next to A
    feet[2][121] = (4.4, 10.5)  # next to D (4.5, 10.4)
    dets = [{"frame": 100, "clip_frame": 100, "track": 1, "side": "near"},
            {"frame": 121, "clip_frame": 121, "track": 2, "side": "near"},
            {"frame": 300, "clip_frame": 300, "track": 3, "side": "far"}]  # track 3 unknown
    D.label_hitters(dets, shots, feet)
    assert [d["letter"] for d in dets] == ["A", "D", None]
    labels = [{"frame": s["frame"], "side": s["side"], "letter": s["player"]} for s in shots]
    r = D.hitter_scores(dets, labels, 3)
    # the third shot has no hitter label, so only two are checked; the second det's side is wrong
    assert r["letter"] == {"checked": 2, "right": 2}
    assert r["side"] == {"checked": 2, "right": 1}


def test_nearest_letter_gives_up_without_a_position():
    assert D.nearest_letter(np.array([np.nan, 1.0]), {"A": (0, 0)}) == (None, float("inf"))
    k, d = D.nearest_letter(np.array([1.0, 1.0]), {"A": (0, 0), "B": (1, 2)})
    assert k == "B" and d == 1.0
