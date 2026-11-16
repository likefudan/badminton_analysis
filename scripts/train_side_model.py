"""Train the "which half of the court hit" model of `badminton hits` on DShuttleNet (docs/hits.md).

    python scripts/train_side_model.py eval/<match>/<match>.rallies.mp4 [...] -o ~/.cache/badminton/hit_side_model.json

Each clip is one that scripts/bfmd_hits.py cut from a DShuttleNet match (skeleton,
shuttle track, court and frame map next to it; scripts/dshuttlenet_hits.py wrote the
labels). The candidate stage of `badminton hits` runs on it; every candidate within
3 frames of a labelled shot (labels moved 1 frame earlier, docs/hits.md) becomes one
row: hits.side_features and the half the labelled hitter stands in. The model is
scikit-learn's HistGradientBoostingClassifier, written as the plain JSON that
hits.HitClassifier evaluates with numpy.

Before the final fit the script prints leave-one-match-out accuracy; with --lomo DIR
it also writes those models (model_<k>.json, k in the order of the clips), which is
what the numbers in docs/hits.md use: each match is scored with a model that never saw it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hit_classifier import SHIFT, TOL, clip_data  # noqa: E402

PARAMS = dict(max_iter=200, learning_rate=0.05, max_depth=3)


def labelled_rows(clip: Path, shuttle_suffix: str = ".shuttle.csv") -> tuple[np.ndarray, np.ndarray]:
    """(X, y) of one clip: side features of the candidates at labelled shots, y = 1 for the near half."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from badminton import hits as H
    from badminton.court import CourtCalibration

    _, _, _, ctx = clip_data(clip, shuttle_suffix)
    base = clip.with_suffix("")
    axis = H.court_axis(CourtCalibration.load(str(base) + ".court.json"))
    X = H.side_features(ctx["cands"], ctx["shuttle"], ctx["players"], ctx["fps"], ctx["frame_h"], axis)
    doc = json.loads(Path(json.loads(Path(str(base) + ".framemap.json").read_text())["rallies_json"]).read_text())
    if "shots" not in doc:
        raise SystemExit(f"{clip.name}：对照里没有击球人（要 DShuttleNet 的片段）")
    fmap = np.asarray(ctx["fmap"])
    frames = np.array([c["frame"] for c in ctx["cands"]])
    ok = frames < len(fmap)
    video_frame = np.where(ok, fmap[np.minimum(frames, len(fmap) - 1)], -1)
    rows, y = [], []
    for s in doc["shots"]:
        if s.get("side") is None:
            continue
        d = np.where(video_frame >= 0, np.abs(video_frame - (s["frame"] + SHIFT)), 10**9)
        k = int(d.argmin())
        if d[k] <= TOL:
            rows.append(k), y.append(int(s["side"] == "near"))
    return X[rows], np.array(y, int)


def export(model, features) -> dict:
    """HistGradientBoostingClassifier (binary, classes 0 far / 1 near) -> the JSON hits.HitClassifier reads."""
    from train_hit_classifier import export as export_trees

    doc = export_trees(model, features)
    doc.update(schema="badminton.hit_side_model/1", classes=["near", "far"], params=PARAMS)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clips", nargs="+", help="DShuttleNet 的 <比赛>.rallies.mp4（scripts/bfmd_hits.py clip 输出）")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--shuttle-suffix", default=".shuttle.csv")
    ap.add_argument("--lomo", help="另存按比赛留出的模型到这个文件夹（model_<序号>.json）")
    a = ap.parse_args(argv)
    from sklearn.ensemble import HistGradientBoostingClassifier

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from badminton.hits import SIDE_FEATURES, HitClassifier

    data = [labelled_rows(Path(c), a.shuttle_suffix) for c in a.clips]
    for c, (X, y) in zip(a.clips, data):
        print(f"{Path(c).name[:50]}：{len(y)} 拍，近端 {int(y.sum())}")

    def fit(keep):
        X = np.concatenate([data[k][0] for k in keep])
        y = np.concatenate([data[k][1] for k in keep])
        return export(HistGradientBoostingClassifier(random_state=0, **PARAMS).fit(X, y), SIDE_FEATURES)

    if len(data) > 1:
        right = total = 0
        for m in range(len(data)):
            doc = fit([k for k in range(len(data)) if k != m])
            p = HitClassifier(doc, SIDE_FEATURES).proba(data[m][0])[:, 0]
            right += int(((p > 0.5) == (data[m][1] == 1)).sum())
            total += len(p)
            if a.lomo:
                Path(a.lomo).mkdir(parents=True, exist_ok=True)
                (Path(a.lomo) / f"model_{m}.json").write_text(json.dumps(doc))
        print(f"按比赛留出：单拍判对上下半场 {right}/{total}（{right / total:.1%}）")
    Path(a.output).expanduser().write_text(json.dumps(fit(range(len(data)))))
    print(f"已写入 {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
