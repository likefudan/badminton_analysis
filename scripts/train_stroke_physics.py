"""Train the physics model of `badminton strokes` on DShuttleNet's labels (docs/stroke_types.md).

    python scripts/train_stroke_physics.py raw/ -o ~/.cache/badminton/stroke_physics.json [--without 23,29,41,42,53]

Needs no video: every labelled shot gives the kind of stroke (serves left out) and what
strokes.physics_features measures on our own tracks: how far from the net the hitter stood, how
far from the net the other side hit the reply, the seconds in between, whether the shuttle was
hit above the net (DShuttleNet's hit_height) and how far across the court the reply was hit.
The last shot of a rally has no reply; it is kept with those numbers missing, as it will be at
run time. The model is scikit-learn's HistGradientBoostingClassifier, written as the JSON that
hits.HitClassifier evaluates with numpy. Before the final fit the script prints the accuracy
with matches held out; --without leaves matches out of the final fit (to score them later with
our own tracking, see docs).

The downloaded files are untrusted data: this only reads CSV.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

PARAMS = dict(max_iter=150, learning_rate=0.08)
MAX_REPLY_S = 2.5  # strokes.NEXT_HIT_MAX_S


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def table(raw: str | Path, classes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, match id) over the labelled shots of every match; y indexes `classes`."""
    raw = Path(raw)

    def rows(name):
        with open(raw / name, newline="") as f:
            return list(csv.DictReader(f))

    set_match = {r["set_id"]: r["match_id"] for r in rows("set.csv")}
    rally_match = {r["rally_id"]: set_match.get(r["set_id"]) for r in rows("rally.csv")}
    by = collections.defaultdict(list)
    for r in rows("shot.csv"):
        by[r["rally_id"]].append(r)
    X, y, g = [], [], []

    def who(r):
        pid = r["player"].split(".")[0]
        return next((k for k in "ABCD" if pid and r[k].split(".")[0] == pid), None)

    def from_net(r, k):  # the labels are decimetres, the net at 67
        return abs(_num(r[f"player_{k}_y"]) - 67.0) / 10

    for rid, shots in by.items():
        if rally_match.get(rid) is None:
            continue
        shots.sort(key=lambda r: int(r["shot_num"]))
        for a, b in zip(shots, shots[1:] + [None]):
            ka = who(a)
            if ka is None or a["ball_type"] not in classes or not np.isfinite(from_net(a, ka)):
                continue
            f = [from_net(a, ka), np.nan, np.nan, {"1": 1.0, "2": 0.0}.get(a["hit_height"], np.nan), np.nan]
            kb = who(b) if b is not None else None
            if kb is not None:
                dt = (_num(b["start_frame_num"]) - _num(a["start_frame_num"])) / 30.0
                if 0 < dt <= MAX_REPLY_S and np.isfinite(from_net(b, kb)):
                    f[1], f[2] = from_net(b, kb), dt
                    f[4] = abs(_num(b[f"player_{kb}_x"]) - _num(a[f"player_{ka}_x"])) / 10
            X.append(f), y.append(classes.index(a["ball_type"])), g.append(int(rally_match[rid]))
    return np.array(X, float), np.array(y, int), np.array(g, int)


def export(model, features, classes) -> dict:
    """HistGradientBoostingClassifier -> the JSON hits.HitClassifier reads (one tree per class and round)."""
    trees = []
    for rnd in model._predictors:
        trees.append([{"feature": p.nodes["feature_idx"].astype(int).tolist(),
                       "threshold": p.nodes["num_threshold"].astype(float).tolist(),
                       "left": p.nodes["left"].astype(int).tolist(), "right": p.nodes["right"].astype(int).tolist(),
                       "value": p.nodes["value"].astype(float).tolist(),
                       "missing_left": p.nodes["missing_go_to_left"].astype(int).tolist(),
                       "leaf": p.nodes["is_leaf"].astype(int).tolist()} for p in rnd])
    return {"schema": "badminton.stroke_physics/1", "features": list(features),
            "classes": [classes[int(c)] for c in model.classes_],
            "baseline": np.ravel(model._baseline_prediction).astype(float).tolist(), "trees": trees, "params": PARAMS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("raw", help="解压后的 DShuttleNet 文件夹（match.csv、set.csv、rally.csv、shot.csv）")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--without", default="", help="最后训练时不用的 match_id，逗号分隔")
    a = ap.parse_args(argv)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from badminton.strokes import PHYSICS_CLASSES, PHYSICS_FEATURES

    X, y, g = table(a.raw, list(PHYSICS_CLASSES))
    print(f"{len(y)} 拍，{len(set(g))} 场；各类 " + "、".join(f"{c} {int((y == k).sum())}" for k, c in enumerate(PHYSICS_CLASSES)))
    right = np.zeros(len(y), bool)
    sure = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        p = HistGradientBoostingClassifier(random_state=0, **PARAMS).fit(X[tr], y[tr]).predict_proba(X[te])
        right[te], sure[te] = p.argmax(1) == y[te], p.max(1)
    print(f"按比赛留出：{right.mean():.1%}；把握 85% 以上的 {np.mean(sure >= 0.85):.0%} 里对 {right[sure >= 0.85].mean():.1%}")
    out = [int(v) for v in a.without.split(",") if v.strip()]
    keep = ~np.isin(g, out)
    model = HistGradientBoostingClassifier(random_state=0, **PARAMS).fit(X[keep], y[keep])
    Path(a.output).expanduser().write_text(json.dumps(export(model, PHYSICS_FEATURES, list(PHYSICS_CLASSES))))
    print(f"已写入 {a.output}（{int(keep.sum())} 拍）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
