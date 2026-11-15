"""Train the hit / bounce / neither classifier of `badminton hits` on BFMD (docs/hits.md).

Two steps, because the second needs scikit-learn and `badminton` itself does not:

    python scripts/train_hit_classifier.py features eval/<match>/<match>.rallies.mp4 [...] -o cands.npz [--shuttle-suffix .shuttle.csv]
    python scripts/train_hit_classifier.py train cands.npz [cands_other_track.npz] -o ~/.cache/badminton/hit_classifier.json [--oof oof.npz]

`features` runs the candidate stage of `badminton hits` on every clip that
scripts/bfmd_hits.py made (skeleton, shuttle track, court and frame map next to
it) and computes hits.candidate_features for each candidate. Its class comes
from BFMD, with the labels moved 1 frame earlier (docs/hits.md, 帧号对齐): hit
when a hit label is within 3 frames, else bounce when a bounce label is, else
neither. Only candidates inside BFMD's Rally spans count (outside them nothing
is labelled).

`train` fits scikit-learn's HistGradientBoostingClassifier and writes its trees
as plain JSON that hits.HitClassifier evaluates with numpy. With --oof it also
writes leave-one-match-out probabilities (each match predicted by a model that
never saw it), which is what the numbers in docs/hits.md use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SHIFT, TOL = -1, 3
PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0)


def clip_data(clip: Path, shuttle_suffix: str = ".shuttle.csv"):
    """Candidates, features and BFMD classes of one rally clip; also returns the context
    (players, shuttle, fps …) for callers that run the detector on it."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from badminton import hits as H
    from badminton.court import CourtCalibration
    from badminton.tracking import load_npz, track_players

    base = clip.with_suffix("")
    data = load_npz(str(base) + ".skeleton.npz")
    counts = [max(((data["side"][i] == s) & data["valid"][i]).sum() for s in (0, 1)) for i in range(len(data["valid"]))]
    per_side = min(2, max(1, int(np.percentile(counts, 90)))) if counts else 2
    tracking = track_players(data, CourtCalibration.load(str(base) + ".court.json"), per_side)
    side = json.loads(Path(str(base) + ".framemap.json").read_text())
    fmap = side["frame_map"]
    doc = json.loads(Path(side["rallies_json"]).read_text())
    n = max(len(fmap), int(data["frame"][-1]) + 1)
    fps, frame_w, frame_h = float(data["fps"]), float(data["frame_size"][0]), float(data["frame_size"][1])
    shuttle = H.drop_static(H.load_shuttle(str(base) + shuttle_suffix, n))
    players = H.player_signals(data, tracking, len(shuttle.xy))
    cands = H.candidates(shuttle, players, fps, frame_h, H.Options())
    X = H.candidate_features(cands, shuttle, players, fps, frame_h, frame_w)
    to_clip = {f: j for j, f in enumerate(fmap) if f >= 0}
    rallies = doc["rallies"]

    def in_rally(f):
        return any(a <= f <= b for a, b in rallies)

    hits = np.array([to_clip[h] + SHIFT for h in doc["hits"] if h in to_clip and in_rally(h)])
    bounces = np.array([to_clip[f] + SHIFT for f in doc["events"].get("bounce", []) if f in to_clip])
    y = np.full(len(cands), 2)
    keep = np.zeros(len(cands), bool)
    for r, c in enumerate(cands):
        j = c["frame"]
        keep[r] = 0 <= j < len(fmap) and fmap[j] >= 0 and in_rally(fmap[j])
        if len(hits) and np.abs(hits - j).min() <= TOL:
            y[r] = 0
        elif len(bounces) and np.abs(bounces - j).min() <= TOL:
            y[r] = 1
    ctx = {"shuttle": shuttle, "players": players, "fps": fps, "frame_h": frame_h, "frame_w": frame_w,
           "cands": cands, "fmap": fmap, "rallies": rallies, "hits": hits, "match": doc["match"]}
    return X, y, keep, ctx


def cmd_features(a) -> int:
    Xs, ys, ms, fs = [], [], [], []
    for m, clip in enumerate(a.clips):
        X, y, keep, ctx = clip_data(Path(clip), a.shuttle_suffix)
        Xs.append(X[keep]), ys.append(y[keep]), ms.append(np.full(keep.sum(), m))
        fs.append(np.array([c["frame"] for c in ctx["cands"]])[keep])
        print(f"{ctx['match'][:50]}：候选 {keep.sum()}，击球 {(y[keep] == 0).sum()}，落地 {(y[keep] == 1).sum()}")
    np.savez(a.output, X=np.concatenate(Xs), y=np.concatenate(ys), match=np.concatenate(ms),
             frame=np.concatenate(fs), clips=np.array([str(c) for c in a.clips]))
    return 0


def export(model, features) -> dict:
    """HistGradientBoostingClassifier -> the JSON hits.HitClassifier reads."""
    trees = []
    for rnd in model._predictors:
        out = []
        for p in rnd:
            nodes = p.nodes
            out.append({"feature": nodes["feature_idx"].astype(int).tolist(),
                        "threshold": nodes["num_threshold"].astype(float).tolist(),
                        "left": nodes["left"].astype(int).tolist(), "right": nodes["right"].astype(int).tolist(),
                        "value": nodes["value"].astype(float).tolist(),
                        "missing_left": nodes["missing_go_to_left"].astype(int).tolist(),
                        "leaf": nodes["is_leaf"].astype(int).tolist()})
        trees.append(out)
    names = ("hit", "bounce", "none")
    classes = [names[int(c)] for c in model.classes_]
    if len(classes) == 2:  # binary model: a single logit for classes_[1]
        classes = [classes[1], classes[0]]
    return {"schema": "badminton.hit_classifier/1", "features": list(features), "classes": classes,
            "baseline": np.ravel(model._baseline_prediction).astype(float).tolist(), "trees": trees,
            "params": PARAMS}


def cmd_train(a) -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from badminton.hits import FEATURES, HitClassifier

    ds = [np.load(c) for c in a.cands]
    names = [Path(str(c)).name for c in ds[0]["clips"]]
    for d in ds[1:]:  # same matches, maybe another shuttle track: match numbers must agree
        if [Path(str(c)).name for c in d["clips"]] != names:
            raise SystemExit("几个 npz 的比赛不一样（要用同一组、同样顺序的片段）")
    allX = np.concatenate([d["X"] for d in ds])
    feats = [f for f in FEATURES if f not in set(a.drop.split(",") if a.drop else [])]
    X = allX[:, [FEATURES.index(f) for f in feats]]
    y, match = np.concatenate([d["y"] for d in ds]), np.concatenate([d["match"] for d in ds])
    track = np.concatenate([np.full(len(d["y"]), k) for k, d in enumerate(ds)])
    frame = np.concatenate([d["frame"] for d in ds])
    if a.oof:
        oof = np.zeros((len(y), 3))
        for m in np.unique(match):
            tr = match != m
            clf = HistGradientBoostingClassifier(random_state=0, **PARAMS).fit(X[tr], y[tr])
            oof[match == m] = clf.predict_proba(X[match == m])
            acc = (oof[match == m].argmax(1) == y[match == m]).mean()
            print(f"留出第 {m} 场：{(match == m).sum()} 个候选，三类准确率 {100 * acc:.1f}%")
        np.savez(a.oof, p=oof, y=y, match=match, frame=frame, track=track, clips=ds[0]["clips"])
    clf = HistGradientBoostingClassifier(random_state=0, **PARAMS).fit(X, y)
    doc = export(clf, feats)
    check = HitClassifier(doc).proba(allX)
    err = np.abs(check - clf.predict_proba(X)).max()
    if err > 1e-6:
        raise SystemExit(f"导出的树和 scikit-learn 的结果对不上（差 {err}）")
    Path(a.output).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).expanduser().write_text(json.dumps(doc))
    print(f"已写入 {a.output}（{len(doc['trees'])} 轮，{len(y)} 个候选）")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("features", help="每个候选的特征和 BFMD 类别 -> npz")
    s.add_argument("clips", nargs="+", help="scripts/bfmd_hits.py clip 输出的 <比赛>.rallies.mp4")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--shuttle-suffix", default=".shuttle.csv", help="球轨迹文件的后缀（默认 .shuttle.csv）")
    s = sub.add_parser("train", help="训练并导出 JSON（需要 scikit-learn）")
    s.add_argument("cands", nargs="+", help="features 写的 npz；几个（同一组比赛、不同球轨迹）会合在一起训练")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--oof", help="另存按比赛留出的预测概率")
    s.add_argument("--drop", help="不用的特征，逗号分隔（名字见 hits.FEATURES）")
    a = p.parse_args(argv)
    return {"features": cmd_features, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
