"""Score shuttle CSVs against the hand-checked frames in this folder.

    python scripts/shuttle_audit/score.py scripts/shuttle_audit/IMG_8923_stab.audit.json a.shuttle.csv b.shuttle.csv

The audit files hold 64 frames per video (the 40 of the earlier spot checks plus 24
random ones). For each frame the shuttle's position is known only if one of the
methods compared in PR "球追踪分块" found it, so "right" counts are comparable between
methods but are not a recall against the true number of frames with a shuttle.
"""
import csv
import json
import math
import sys

audit = json.load(open(sys.argv[1]))
known = sum(v is not None for v in audit["frames"].values())
print(f"{audit['video']}: {len(audit['frames'])} 帧，其中 {known} 帧知道球的位置")
for path in sys.argv[2:]:
    rows = {int(r["Frame"]): (int(r["Visibility"]), float(r["X"]), float(r["Y"])) for r in csv.DictReader(open(path))}
    right = wrong = 0
    for f, truth in audit["frames"].items():
        v, x, y = rows[int(f)]
        if not v:
            continue
        if truth is not None and math.hypot(x - truth[0], y - truth[1]) < audit["radius_px"]:
            right += 1
        else:
            wrong += 1
    print(f"{path}: 对 {right}，错 {wrong}，输出的点里对的占 {100 * right / max(1, right + wrong):.0f}%")
