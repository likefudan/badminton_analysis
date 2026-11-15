"""Train and score the doubles stroke-type model (badminton.stroke_model) on DShuttleNet.

    python scripts/train_stroke_classifier.py features out/*/*.rallies.mp4 -o cache/ [--bst]
    python scripts/train_stroke_classifier.py train raw/ --cache cache/ [-o weights.pt]

`features` reads what scripts/dshuttlenet_hits.py and the pipeline left beside each rally
clip (framemap, court, skeleton, shuttle, hits) and stores, per match, the model's inputs
for two sets of hits: the labelled ones (label frame, the track standing where the labels
put the hitter) and the ones `badminton hits` found (its frame and its hitter; the label
is that of the labelled shot within ±3 frames, if any). With --bst it also stores what
`badminton strokes` (BST with the singles weights) says for the same hits.

`train` first learns the context part from the labels of every match that has no video
here, then, once per held-out match, the whole model from the other matches with video,
and prints the scores on the held-out one. With -o it then trains on all of them and
saves the weights.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dshuttlenet_hits as D  # noqa: E402

from badminton import stroke_model as SM  # noqa: E402

TEAM = {"A": 0, "B": 0, "C": 1, "D": 1}
LABEL_SHIFT = -1  # DShuttleNet's frame is about one frame after the shuttle turns (docs/hits.md)
MATCH_TOL = 3
HITTER_MAX_M = 2.0


# ---------------------------------------------------------------- labels only


def label_rallies(raw: str | Path) -> dict[str, list[list[dict]]]:
    """video_name -> rallies -> shots ({"shot": SM.Shot, "type": class index or -1}) from the CSVs."""
    raw = Path(raw)
    name = {r["match_id"]: r["video_name"] for r in D._rows(raw / "match.csv")}
    set_match = {r["set_id"]: r["match_id"] for r in D._rows(raw / "set.csv")}
    rally_match = {r["rally_id"]: set_match.get(r["set_id"]) for r in D._rows(raw / "rally.csv")}
    by: dict = collections.defaultdict(list)
    for r in D._rows(raw / "shot.csv"):
        frame = D._num(r["start_frame_num"])
        if rally_match.get(r["rally_id"]) is None or frame is None:
            continue
        ids = {k: r[k].split(".")[0] for k in D.LETTERS}
        pid = r["player"].split(".")[0] if r["player"] else ""
        who = next((k for k in D.LETTERS if ids[k] == pid and pid), None)
        pos = {}
        for k in D.LETTERS:
            x, y = D._num(r[f"player_{k}_x"]), D._num(r[f"player_{k}_y"])
            if x is not None and y is not None:
                pos[k] = np.array(D.to_metres(x, y))
        kind = SM.CLASSES.index(r["ball_type"]) if r["ball_type"] in SM.CLASSES else -1
        by[r["rally_id"]].append((int(r["shot_num"]), {"shot": SM.Shot(frame / 30.0, who, pos, TEAM), "type": kind}))
    out: dict = collections.defaultdict(list)
    for rid, shots in by.items():
        out[name[rally_match[rid]]].append([s for _, s in sorted(shots, key=lambda t: t[0])])
    return out


def label_context(rallies: list[list[dict]], rng: np.random.Generator | None = None, drop: float = 0.0,
                  noise_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Context features and classes of the labelled shots. With `rng`: some shots are left out
    (as when a hit is not found) and the positions jittered (as tracked positions are)."""
    X, Y = [], []
    for rally in rallies:
        shots = rally
        if rng is not None:
            shots = [s for s in rally if rng.random() >= drop]
            shots = [{"type": s["type"], "shot": SM.Shot(
                s["shot"].t + rng.normal(0, 1 / 30), s["shot"].hitter,
                {k: v + rng.normal(0, noise_m, 2) for k, v in s["shot"].pos.items() if rng.random() >= 0.03},
                TEAM)} for s in shots]
        seq = [s["shot"] for s in shots]
        for i, s in enumerate(shots):
            if s["type"] < 0:
                continue
            f = SM.context_features(seq, i)
            if f is not None:
                X.append(f)
                Y.append(s["type"])
    return np.stack(X), np.asarray(Y)


# ---------------------------------------------------------------- features from the pipeline


def _match_frames(found: list[int], labels: list[int], tol: int) -> dict[int, int]:
    """found index -> label index, one to one, nearest first, within tol frames."""
    pairs = sorted((abs(f - l), i, j) for i, f in enumerate(found) for j, l in enumerate(labels) if abs(f - l) <= tol)
    used_i, used_j, out = set(), set(), {}
    for _, i, j in pairs:
        if i not in used_i and j not in used_j:
            out[i] = j
            used_i.add(i)
            used_j.add(j)
    return out


def clip_features(clip: Path, bst_models: list | None = None) -> dict:
    from badminton import bst
    from badminton import hits as H
    from badminton.court import CourtCalibration
    from badminton.tracking import load_npz, track_players

    side = json.loads(clip.with_suffix(".framemap.json").read_text())
    doc = json.loads(Path(side["rallies_json"]).read_text())
    clip_of = {f: j for j, f in enumerate(side["frame_map"]) if f >= 0}
    data = load_npz(clip.with_suffix(".skeleton.npz"))
    calib = CourtCalibration.load(clip.with_suffix(".court.json"))
    tracking = track_players(data, calib, 2)
    n = int(data["frame"][-1]) + 1
    shuttle_px = bst.read_shuttle_csv(clip.with_suffix(".shuttle.csv"), n)
    scene = SM.Scene(data, tracking, shuttle_px, calib)

    # labelled hits on the clip's frame axis, hitter = the track standing where the labels say
    labelled = []
    for s in doc["shots"]:
        j = clip_of.get(s["frame"] + LABEL_SHIFT)
        if j is None:
            continue
        tid, where = None, s["positions"].get(s["player"]) if s["player"] else None
        if where is not None:
            dist = {t: float(np.linalg.norm(p - np.asarray(where))) for t in scene.tracks
                    if (p := scene.position(t, j)) is not None}
            if dist and min(dist.values()) <= HITTER_MAX_M:
                tid = min(dist, key=dist.get)
        labelled.append({"frame": j, "player_id": tid, "side": s["side"],
                         "type": SM.CLASSES.index(s["ball_type"]) if s["ball_type"] in SM.CLASSES else -1})
    found = [{"frame": int(h["frame"]["value"]), "player_id": h.get("player", {}).get("value"),
              "side": h.get("player", {}).get("side")}
             for h in json.loads(clip.with_suffix(".hits.json").read_text())["hits"]]
    pairs = _match_frames([h["frame"] for h in found], [h["frame"] for h in labelled], MATCH_TOL)
    for i, h in enumerate(found):
        lab = labelled[pairs[i]] if i in pairs else None
        h["type"] = lab["type"] if lab else -1
        h["hitter_right"] = bool(lab and lab["player_id"] is not None and lab["player_id"] == h["player_id"])

    out = {"match": doc["match"], "event": doc.get("event", "")}
    for name, hits in (("gt", labelled), ("det", found)):
        feats = SM.hit_features(scene, hits)
        ok = np.array([f is not None for f in feats])
        zero = {"ctx": np.zeros(SM.CTX_DIM, np.float32), "seq": np.zeros((SM.SEQ_LEN, SM.SEQ_DIM), np.float32), "far": 0.0}
        feats = [f or zero for f in feats]
        out[f"{name}_ok"] = ok
        out[f"{name}_ctx"] = np.stack([f["ctx"] for f in feats])
        out[f"{name}_seq"] = np.stack([f["seq"] for f in feats])
        out[f"{name}_far"] = np.array([f["far"] for f in feats], np.float32)
        out[f"{name}_y"] = np.array([h["type"] for h in hits])
        out[f"{name}_frame"] = np.array([h["frame"] for h in hits])
        if name == "det":
            out["det_hitter_right"] = np.array([h["hitter_right"] for h in hits])
        if bst_models:
            from badminton import strokes as ST

            res = ST.classify_hits(data, tracking, [{k: h[k] for k in ("frame", "player_id", "side")} for h in hits],
                                   shuttle_px, calib, bst_models, scene.fps)
            out[f"{name}_bst"] = np.array([SM.SHUTTLESET_TO_CLASS.get(r["shuttleset_type"], -1) for r in res])
    return out


def cmd_features(a) -> int:
    models = None
    if a.bst:
        from badminton import strokes as ST

        models = ST.load_models([ST.DEFAULT_WEIGHTS])
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for c in a.clips:
        f = clip_features(Path(c), models)
        np.savez_compressed(Path(a.out_dir) / f"{f['match']}.npz", **f)
        print(f"{f['match']}：标注 {len(f['gt_y'])} 拍（{int(f['gt_ok'].sum())} 拍找到击球人的追踪），"
              f"识别 {len(f['det_y'])} 拍（{int((f['det_y'] >= 0).sum())} 拍对上标注）", flush=True)
    return 0


# ---------------------------------------------------------------- training


def _device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def fit(model, ctx, y, seq=None, far=None, epochs: int = 30, lr: float = 2e-3, seed: int = 0, batch: int = 256):
    """Train in place. Without `seq` only the context part (and its own head)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    dev = _device()
    model.to(dev).train()
    ctx_m = torch.from_numpy(SM.mirror_context(ctx)).to(dev)
    ctx, y = torch.from_numpy(ctx).to(dev), torch.from_numpy(y).to(dev)
    if seq is not None:
        seq_m = torch.from_numpy(SM.mirror_sequence(seq)).to(dev)
        seq, far = torch.from_numpy(seq).to(dev), torch.from_numpy(far).to(dev)
        slow = [p for n, p in model.named_parameters() if n.startswith("ctx")]
        fast = [p for n, p in model.named_parameters() if not n.startswith("ctx")]
        groups = [{"params": fast, "lr": lr}, {"params": slow, "lr": lr / 5}]
    else:
        groups = [{"params": list(model.parameters()), "lr": lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    n = len(y)
    steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps)
    for _ in range(epochs):
        for b in torch.randperm(n, device=dev).split(batch):
            flip = torch.rand(len(b), device=dev) < 0.5
            c = torch.where(flip[:, None], ctx_m[b], ctx[b])
            if seq is None:
                logits = model(c)
            else:
                logits = model(c, torch.where(flip[:, None, None], seq_m[b], seq[b]), far[b])
            loss = F.cross_entropy(logits, y[b], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.to("cpu").eval()


def predict(model, ctx, seq=None, far=None) -> np.ndarray:
    import torch

    with torch.no_grad():
        args = [torch.from_numpy(ctx)] + ([torch.from_numpy(seq), torch.from_numpy(far)] if seq is not None else [])
        return model(*args).softmax(-1).numpy()


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    """Accuracy and macro F1 over the classes present; p = -1 counts as wrong."""
    f1 = []
    for c in np.unique(y):
        tp = int(((y == c) & (p == c)).sum())
        f1.append(2 * tp / max(1, int((y == c).sum() + (p == c).sum())))
    return {"n": int(len(y)), "acc": float((y == p).mean()) if len(y) else 0.0, "f1": float(np.mean(f1)) if f1 else 0.0}


def _train_set(caches: list[dict]) -> tuple:
    parts = []
    for c in caches:
        for name in ("gt", "det"):
            keep = c[f"{name}_ok"] & (c[f"{name}_y"] >= 0)
            parts.append([c[f"{name}_{k}"][keep] for k in ("ctx", "seq", "far", "y")])
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def cmd_train(a) -> int:
    import copy

    caches = [dict(np.load(p, allow_pickle=False)) for p in sorted(Path(a.cache).glob("*.npz"))]
    for c in caches:
        c["match"] = str(c["match"])
    held = {c["match"] for c in caches}
    rallies = label_rallies(a.raw)
    rng = np.random.default_rng(0)
    pre = [r for m, rs in rallies.items() if m not in held for r in rs]
    X, Y = zip(*[label_context(pre, rng, a.drop, a.noise_m) for _ in range(a.replicas)])
    X, Y = np.concatenate(X), np.concatenate(Y)
    print(f"站位和节奏部分：{len(rallies) - len(held)} 场没有视频的比赛，{len(Y) // a.replicas} 拍 × {a.replicas} 份扰动", flush=True)
    base = fit(SM.build_model(), X, Y, epochs=a.ctx_epochs, seed=0)
    Xh, Yh = label_context([r for m in held for r in rallies[m]])
    print(f"  留出的 {len(held)} 场，用标注里的站位：{scores(Yh, predict(base, Xh).argmax(1))}", flush=True)

    rows, cm = [], np.zeros((len(SM.CLASSES),) * 2, int)
    tot: dict = collections.defaultdict(lambda: [[], []])
    for test in caches:
        model = copy.deepcopy(base)
        if not a.ctx_only:
            ctx, seq, far, y = _train_set([c for c in caches if c is not test])
            fit(model, ctx, y, seq, far, epochs=a.epochs, lr=a.lr, seed=0)
        row = {"match": test["match"], "event": str(test["event"])}
        for name in ("gt", "det"):
            y = test[f"{name}_y"]
            lab = y >= 0
            pred = np.full(len(y), -1)
            ok = test[f"{name}_ok"]
            pc = predict(base, test[f"{name}_ctx"][ok]).argmax(1)
            pctx = np.full(len(y), -1)
            pctx[ok] = pc
            if a.ctx_only:
                pred = pctx
            else:
                pred[ok] = predict(model, test[f"{name}_ctx"][ok], test[f"{name}_seq"][ok], test[f"{name}_far"][ok]).argmax(1)
            for key, p in ((name, pred), (f"{name}_ctx", pctx), (f"{name}_bst", test.get(f"{name}_bst"))):
                if p is None:
                    continue
                row[key] = scores(y[lab], p[lab])
                tot[key][0].append(y[lab])
                tot[key][1].append(p[lab])
            if name == "det":
                right = lab & test["det_hitter_right"]
                row["det_hitter_right"] = scores(y[right], pred[right])
                tot["det_hitter_right"][0].append(y[right])
                tot["det_hitter_right"][1].append(pred[right])
                np.add.at(cm, (y[lab & (pred >= 0)], pred[lab & (pred >= 0)]), 1)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    total = {k: scores(np.concatenate(v[0]), np.concatenate(v[1])) for k, v in tot.items()}
    print("合计", json.dumps(total, ensure_ascii=False))
    print("识别出的击球（对上标注的），行 = 标注，列 = 模型：", SM.CLASSES)
    print(cm)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "total": total, "confusion": cm.tolist(),
                                            "classes": SM.CLASSES}, ensure_ascii=False, indent=1))
    if a.out:
        model = copy.deepcopy(base)
        ctx, seq, far, y = _train_set(caches)
        fit(model, ctx, y, seq, far, epochs=a.epochs, lr=a.lr, seed=0)
        SM.StrokeClassifier(model, {"trained_on": sorted(held), "pretrain_matches": len(rallies) - len(held),
                                    "held_out_scores": total}).save(a.out)
        print(f"已写入 {a.out}（{len(y)} 拍，{len(held)} 场）")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("features", help="每场比赛存一份模型输入（标注的击球和识别出的击球两套）")
    s.add_argument("clips", nargs="+", help="bfmd_hits clip 输出的视频（旁边要有 framemap、court、skeleton、shuttle、hits）")
    s.add_argument("-o", "--out-dir", required=True)
    s.add_argument("--bst", action="store_true", help="同时存 badminton strokes（BST）对同一批击球的判断")
    s = sub.add_parser("train", help="按场次留出训练和打分；给 -o 时再用全部比赛训练并保存权重")
    s.add_argument("raw", help="解压后的 DShuttleNet 文件夹")
    s.add_argument("--cache", required=True, help="features 的输出目录")
    s.add_argument("-o", "--out", help="保存权重的路径")
    s.add_argument("--json", help="另存逐场分数")
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--ctx-epochs", type=int, default=30)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--replicas", type=int, default=3, help="站位和节奏部分每拍做几份扰动")
    s.add_argument("--drop", type=float, default=0.2, help="扰动时每拍被去掉的概率（模拟漏检）")
    s.add_argument("--noise-m", type=float, default=0.3, help="扰动时站位加的噪声（米）")
    s.add_argument("--ctx-only", action="store_true", help="只用站位和节奏部分（不看姿态和球）")
    a = p.parse_args(argv)
    return {"features": cmd_features, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
