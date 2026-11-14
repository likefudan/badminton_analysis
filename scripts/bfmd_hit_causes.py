"""Rough cause of every miss / extra of one BFMD match (see docs/hits.md, BFMD 职业双打对照).

Runs on the files scripts/bfmd_hits.py clip and `badminton hits` wrote:

    python scripts/bfmd_hit_causes.py eval/<match>/<match>.rallies [causes.json]

Clip frames, BFMD labels shifted by SHIFT; rules only (camera cuts from a
thumbnail difference, shuttle seen or not, nearest player's side, BFMD's own
bounce / net_hit labels), so the split is approximate.
"""
import collections, csv, json, sys
from pathlib import Path
import cv2, numpy as np
from badminton import hits as H
from badminton.court import CourtCalibration, NET_Y

SHIFT, TOL = -1, 3
base = Path(sys.argv[1])  # …/<match>.rallies (no suffix)
out_json = sys.argv[2] if len(sys.argv) > 2 else None
side = json.loads(Path(str(base) + ".framemap.json").read_text())
fmap = side["frame_map"]
doc = json.loads(Path(side["rallies_json"]).read_text())
to_clip = {f: j for j, f in enumerate(fmap) if f >= 0}
rallies = doc["rallies"]
labels = [to_clip[h] + SHIFT for h in doc["hits"] if h in to_clip and any(a <= h <= b for a, b in rallies)]
ev = {k: [to_clip[f] + SHIFT for f in doc["events"].get(k, []) if f in to_clip] for k in ("bounce", "net_hit")}
dets = []
for r in csv.DictReader(open(str(base) + ".hits.csv")):
    j = int(r["frame"])
    if fmap[j] >= 0 and any(a - 6 <= fmap[j] <= b + 6 for a, b in rallies):
        dets.append({"frame": j, "player_id": None, "by": r["found_by"]})
res = H.score_hits(dets, [{"frame": l, "player_id": None, "note": ""} for l in labels], TOL)
sh = H.drop_static(H.load_shuttle(str(base) + ".shuttle.csv", len(fmap)))
calib = CourtCalibration.load(str(base) + ".court.json")

# scene cuts: big change of a tiny grey thumbnail between consecutive frames
cache = Path(str(base) + ".cuts.npy")
if cache.exists():
    diff = np.load(cache)
else:
    cap = cv2.VideoCapture(str(base) + ".mp4"); prev = None; diff = []
    while True:
        ok, im = cap.read()
        if not ok: break
        g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        diff.append(0.0 if prev is None else float(np.abs(g - prev).mean())); prev = g
    diff = np.array(diff); np.save(cache, diff)
cuts = np.array([j for j in np.flatnonzero(diff > 25) if j > 0 and fmap[j] >= 0 and fmap[j - 1] == fmap[j] - 1])

# rally (clip) spans and first label of each
spans, cur = [], None
for j, f in enumerate(fmap):
    if f >= 0 and cur is None: cur = j
    if f < 0 and cur is not None: spans.append((cur, j - 1)); cur = None
if cur is not None: spans.append((cur, len(fmap) - 1))
first = {min([l for l in labels if a <= l <= b], default=None) for a, b in spans}
last = {max([l for l in labels if a <= l <= b], default=None) for a, b in spans}

# people boxes around the shuttle (four players close together)
npz = np.load(str(base) + ".skeleton.npz", allow_pickle=True)
fr = np.asarray(npz["frame"]); row_of = {int(f): i for i, f in enumerate(fr)}


def near_people(j, xy):
    i = row_of.get(j)
    if i is None or xy is None: return 0
    n = 0
    for p in range(npz["bbox"].shape[1]):
        if not npz["valid"][i, p]: continue
        x0, y0, x1, y1 = npz["bbox"][i, p]; h = y1 - y0
        if x0 - 0.5 * h <= xy[0] <= x1 + 0.5 * h and y0 - 0.3 * h <= xy[1] <= y1 + 0.3 * h: n += 1
    return n


def nearest_side(j, xy):
    i = row_of.get(j)
    if i is None or xy is None: return None
    best = None
    for p in range(npz["bbox"].shape[1]):
        if not npz["valid"][i, p]: continue
        x0, y0, x1, y1 = npz["bbox"][i, p]
        d = np.hypot(max(x0 - xy[0], 0, xy[0] - x1), max(y0 - xy[1], 0, xy[1] - y1)) / (y1 - y0)
        if best is None or d < best[0]: best = (d, int(npz["side"][i, p]))
    return best[1] if best else None


def shuttle_near(j, w=3):
    k = [i for i in range(j - w, j + w + 1) if 0 <= i < len(sh.xy) and sh.seen[i]]
    return sh.xy[min(k, key=lambda i: abs(i - j))] if k else None


def miss_cause(l):
    if any(TOL < abs(d["frame"] - l) <= 8 for d in dets): return "时刻偏 4–8 帧"
    if len(cuts) and np.abs(cuts - l).min() <= 10: return "换机位/切镜头"
    seen = sh.seen[l - 3:l + 4].sum()
    xy = shuttle_near(l, 8)
    if seen <= 2:
        if xy is not None and xy[1] < 60: return "球出画面（上沿）"
        return "球轨迹丢失"
    if l in first: return "发球"
    cxy = calib.to_court(np.array([xy]))[0]
    if abs(cxy[1] - NET_Y) < 1.5 and xy[1] > 0: return "网前小球"
    if near_people(l, xy) >= 2: return "多人挨着球（遮挡/分不清）"
    sd = nearest_side(l, xy)
    return {1: "远端", 0: "近端"}.get(sd, "其他")


def extra_cause(d):
    if any(abs(d - b) <= 6 for b in ev["bounce"]): return "落地/出界当成击球"
    if any(abs(d - b) <= 6 for b in ev["net_hit"]): return "擦网/下网"
    if any(TOL < abs(d - l) <= 8 for l in labels): return "时刻偏 4–8 帧"
    if len(cuts) and np.abs(cuts - d).min() <= 10: return "换机位/切镜头"
    rl = [l for l in last if l is not None and 0 < d - l <= 120 and not any(l < x < d for x in labels)]
    if rl: return "回合结束后（捡球、收拍）"
    rf = [l for l in first if l is not None and 0 < l - d <= 120 and not any(d < x < l for x in labels)]
    if rf: return "发球前（准备、拍球）"
    return "其他（挥拍没碰球、轨迹错点）"


mc = collections.Counter(miss_cause(l) for l in res["missed_frames"])
xc = collections.Counter(extra_cause(d) for d in res["extra_frames"])
print(f"{base.name[:50]}  tp {res['tp']} fp {res['fp']} fn {res['fn']}  切镜头 {len(cuts)}")
print("  漏检:", dict(mc.most_common()))
print("  误检:", dict(xc.most_common()))
if out_json:
    Path(out_json).write_text(json.dumps({"missed": {str(l): miss_cause(l) for l in res["missed_frames"]},
                                          "extra": {str(d): extra_cause(d) for d in res["extra_frames"]},
                                          "miss_counts": mc, "extra_counts": xc, "cuts": len(cuts)},
                                         ensure_ascii=False, indent=1))
