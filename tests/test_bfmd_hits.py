"""scripts/bfmd_hits.py: BFMD Label Studio export -> labels, rally clip frame map, time-only scoring."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bfmd_hits as B  # noqa: E402

from badminton import hits as H  # noqa: E402


def _export(tmp_path):
    def ev(label, s, e):
        return {"value": {"ranges": [{"start": s, "end": e}], "timelinelabels": [label]},
                "type": "timelinelabels", "origin": "manual"}

    result = [ev("Rally", 100, 160), ev("hit", 110, 110), ev("hit", 140, 140), ev("Rally", 300, 350),
              ev("hit", 310, 310), ev("hit", 500, 500), ev("bounce", 158, 158), ev("replay", 400, 450),
              {"value": {"choices": ["x"]}, "type": "choices"}]  # entries without a timeline label are skipped
    p = tmp_path / "match.json"
    p.write_text(json.dumps([{"id": 1, "annotations": [{"result": result}]}]))
    return p


def test_read_bfmd_and_labels(tmp_path):
    ev = B.read_bfmd(_export(tmp_path))
    assert ev["hit"] == [110, 140, 310, 500]
    assert ev["Rally"] == [(100, 160), (300, 350)]
    assert ev["bounce"] == [158] and ev["replay"] == [(400, 450)]
    B.main(["labels", str(_export(tmp_path)), "-o", str(tmp_path / "out")])
    labels = H.load_labels(tmp_path / "out" / "match.hits.labels.csv")
    assert [l["frame"] for l in labels] == [110, 140, 310, 500]
    assert all(l["player_id"] is None for l in labels)
    doc = json.loads((tmp_path / "out" / "match.rallies.json").read_text())
    assert doc["rallies"] == [[100, 160], [300, 350]] and doc["hits_in_rallies"] == 3


def test_plan_clip_margins_holds_and_joins():
    fmap = B.plan_clip([(10, 12), (14, 15), (30, 31)], n_frames=32, margin=1, hold=2)
    # (9..13) and (13..16) touch and are joined; (29..31) is cut at the end of the video
    assert fmap == [9, 10, 11, 12, 13, 14, 15, 16, -1, -1, 29, 30, 31, -1, -1]


def test_detections_map_back_and_scoring_ignores_the_missing_hitter(tmp_path):
    fmap = [9, 10, 11, -1, -1, 29, 30]
    p = tmp_path / "clip.hits.csv"
    p.write_text("frame,time_s,player,side,found_by,source\n1,0,3,near,track,derived\n4,0,1,far,track,derived\n"
                 "6,0,2,far,track,derived\n")
    dets = B.detections_in_video_frames(p, fmap)
    assert [d["frame"] for d in dets] == [10, 30]  # the hit on a held frame is dropped
    # BFMD doubles have no hitter: only the moment is compared
    r = H.score_hits([{"frame": 10, "player_id": 3}, {"frame": 33, "player_id": 2}],
                     [{"frame": 11, "player_id": None, "note": ""}, {"frame": 30, "player_id": None, "note": ""}], 3)
    assert (r["tp"], r["fp"], r["fn"], r["hitter_checked"]) == (2, 0, 0, 0)
