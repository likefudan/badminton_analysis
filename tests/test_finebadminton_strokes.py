import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "finebadminton_strokes", Path(__file__).resolve().parents[1] / "scripts/finebadminton_strokes.py")
FB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FB)


class _Track:
    def __init__(self, tid, side, det):
        self.id, self.side, self.det_index = tid, side, np.asarray(det)


class _Tracking:
    def __init__(self, tracks):
        self.tracks = tracks


def test_type_mapping():
    assert FB.stroke_type("kill", ["slice smash", "jump smash"]) == ("劈杀", "杀球")
    assert FB.stroke_type("kill", ["jump smash", "slice smash"]) == ("跳杀", "杀球")
    assert FB.stroke_type("drop shot", ["blocked drop shot"]) == ("拦吊", "吊球")
    assert FB.stroke_type("drop shot", ["劈吊"]) == ("吊球", "吊球")  # unknown subtype: the type alone
    assert FB.stroke_type("clear", ["attacking clear"]) == ("平高球", "高远/平高")
    assert FB.stroke_type("cross-court net shot", []) == ("勾对角", "勾对角")
    assert FB.stroke_type("", []) is None
    groups = {g for _, g in FB.TYPES.values()}
    assert groups == {"发球", "杀球", "高远/平高", "吊球", "放网", "勾对角", "挑球", "推扑", "抽挡"}


def test_convert_maps_frames_sides_and_players(tmp_path):
    labels = [{"fps": 25.0, "hitting": [
        {"start_frame": 1000, "player": "A", "Foundational Actions Level": {"hit type": "kill", "subtype": [], "hitter": "bottom"}},
        {"start_frame": 1020, "player": "B", "Foundational Actions Level": {"hit type": "clear", "subtype": [], "hitter": "top"}},
        {"start_frame": 5000, "player": "A", "Foundational Actions Level": {"hit type": "kill", "subtype": [], "hitter": "bottom"}},
        {"start_frame": 1030, "player": "A", "Foundational Actions Level": {"hit type": "", "subtype": [], "hitter": "bottom"}},
        {"start_frame": None, "Foundational Actions Level": {"hitter": "top"}},
    ]}]
    p = tmp_path / "m.json"
    p.write_text(json.dumps(labels))
    shots, fps = FB.load_shots(p)
    assert fps == 25.0 and len(shots) == 4 and shots[0]["frame"] == 1000 + FB.LABEL_TO_CONTACT
    frame_map = list(range(990, 1040))  # clip frame j shows video frame 990 + j
    n = len(frame_map)
    data = {"frame": np.arange(n)}
    near = _Track(1, 0, np.where(np.arange(n) < 40, 0, -1))
    far = _Track(2, 1, np.zeros(n))
    strokes, skipped = FB.convert(shots, data, _Tracking([near, far]), 25.0, frame_map)
    assert skipped == {"not_in_clip": 1, "unknown_type": 1, "no_player": 0}
    assert [(s["frame"], s["side"], s["player_id"], s["group"]) for s in strokes] == [
        (13, "near", 1, "杀球"), (33, "far", 2, "高远/平高")]
    assert all(s["prob"] == 1.0 and s["needs_review"] is False for s in strokes)
