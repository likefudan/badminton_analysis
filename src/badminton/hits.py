"""Step 6 prototype: when was the shuttle hit, and by whom.

Two signals, both from files earlier steps already wrote:

  * shuttle track (badminton shuttle, one row per video frame): a hit shows as
    a sudden change of the shuttle's image velocity. Velocity before and after
    each frame is taken over a short window; the score is the larger of
    "velocity change" (also catches a smash off a falling shuttle, which does
    not turn in the image) and "direction turn" (slow net shots, where the
    change is small but the direction flips). Contacts hidden inside a short
    gap of the track are scored across the gap; a flight that starts or ends
    in a player's hands (a serve, a stroke the track lost) is a candidate too.
  * wrists (badminton skeleton, COCO keypoints 9/10 of the tracked players):
    wrist speed in body heights per second, and the distance from the shuttle
    to the nearer wrist in body heights. A fast wrist next to the shuttle is a
    swing that met it.

Every candidate gets an evidence value per player (kink + wrist speed - wrist
distance - bias). The hits are the set of candidates with the largest total
evidence in which consecutive hits of a rally come from opposite sides of the
net: the follow-through of a stroke or the preparation for the next one is a
fast wrist near the shuttle as well, but the same side cannot hit twice in a
row. A kink with nobody within reach is the shuttle landing or a tracking
error. The hitter is the player the chosen evidence belongs to.

Speeds and distances are in body heights of the player nearest to the shuttle
(not pixels, not frames): the same stroke moves the shuttle fewer pixels on
the far side of the court, and one set of thresholds then serves both sides,
30 and 60 fps, 1080p and 4K.

Read before writing this (ideas only, no code taken): COURTSIDE
(github.com/jayclim/BadmintonAI, MIT; hits.py: velocity change or turn, gap and
serve passes, wrist peaks to recover missed contacts) and BadmintonTrackNet
(github.com/ZSHYC/BadmintonTrackNet, MIT; docs/events: rule candidates on
kinematics, then a classifier). Hit sound was tried as a third signal and
dropped; docs/hits.md has the numbers.

Every value written carries where it came from (see provenance.py).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .provenance import DERIVED, INTERPOLATED, MANUAL, MEASURED, legend

SCHEMA = "badminton.hits/1"

WINDOW_S = 0.134  # velocity before / after a frame is taken over this long (4 frames at 30 fps)
INTERP_S = 0.2  # gaps of the track up to this long are bridged before scoring
GAP_MAX_S = 0.47  # a contact may hide in a gap of the raw track up to this long
MIN_GAP_S = 0.27  # two hits are at least this far apart (fastest exchanges ~0.3 s)
RALLY_GAP_S = 2.5  # no hit for this long: the next one starts a new rally (any side may begin)
SAME_SIDE_MIN_S = 0.6  # two hits from the same side closer than this are impossible in one rally
SAME_SIDE_COST = 1.0  # ...and further apart they cost this much evidence (a hit in between was missed)
REF_BH_FRAC = 0.22  # body height / frame height assumed when no player is tracked
DV_THR = 5.0  # body heights / s: velocity change that alone makes a hit
TURN_THR = 1.0  # body heights / s: (1 - cos) * slower of the two speeds
TURN_COS = 0.5  # a "turn" needs at least 60 degrees
MIN_SPEED = 0.5  # body heights / s: slower than this before or after is not a flight
WEAK = 0.4  # kinks from this fraction of the threshold up become candidates
KINK_CAP = 3.0
EDGE_BH = 0.7  # body heights: a track starting / ending this close to a wrist is a candidate
EDGE_KINK = {"start": 1.5, "end": 1.0}  # what such an edge counts for, on the kink scale
WRIST_THR = 2.5  # body heights / s: wrist speed peak that is a swing
REACH_BH = 1.3  # body heights: shuttle further than this from every wrist was not hit
WRIST_SYNC_S = 0.1  # wrist peak and kink belong together within this
W_KINK, W_WRIST, W_DIST, BIAS = 1.0, 0.5, 1.0, 0.8  # evidence weights, see evidence()
KPT_THR = 0.3
_WRISTS = (9, 10)


# --- inputs ---------------------------------------------------------------------------------


@dataclass
class ShuttleTrack:
    xy: np.ndarray  # (N, 2) video pixels, NaN where not seen; index = video frame
    source: np.ndarray  # (N,) "" / measured / interpolated / manual

    @property
    def seen(self) -> np.ndarray:
        return np.isfinite(self.xy[:, 0])


def load_shuttle(path: str | Path, n_frames: int | None = None) -> ShuttleTrack:
    """Read `badminton shuttle` output (Frame, Visibility, X, Y[, Source]) or an
    annotation file (…, Status). Without a Source column every seen point counts
    as measured (older files did not record which frames were filled in)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    n = max([int(r["Frame"]) for r in rows], default=-1) + 1
    n = max(n, n_frames or 0)
    xy = np.full((n, 2), np.nan)
    src = np.full(n, "", dtype=object)
    for r in rows:
        if not int(r["Visibility"]):
            continue
        i = int(r["Frame"])
        xy[i] = (float(r["X"]), float(r["Y"]))
        status = r.get("Status", "")
        src[i] = r.get("Source") or (MANUAL if status and status != "machine" else MEASURED)
    return ShuttleTrack(xy, src)


@dataclass
class PlayerSignals:
    id: int
    side: int  # 0 near, 1 far
    wrists: np.ndarray  # (N, 2, 2) video pixels per video frame, NaN when not seen
    body_h: np.ndarray  # (N,) body height in pixels (smoothed), NaN when unknown
    measured: np.ndarray  # (N,) pose model ran on this frame and saw a wrist
    name: str = ""


def player_signals(data: dict, tracking, n_frames: int) -> list[PlayerSignals]:
    """Wrists and body height of every tracked player, on the video's frame axis.

    `data` is the skeleton .npz, `tracking` the result of track_players on it. The
    pose model may have run on every `stride`-th frame only; frames in between are
    interpolated (and flagged as such)."""
    frames = np.asarray(data["frame"], int)
    out = []
    for tr in tracking.tracks:
        w = np.full((n_frames, 2, 2), np.nan)
        bh = np.full(n_frames, np.nan)
        for i in np.flatnonzero(tr.det_index >= 0):
            f, p = frames[i], tr.det_index[i]
            if f >= n_frames:
                continue
            for j, k in enumerate(_WRISTS):
                if data["scores"][i, p, k] >= KPT_THR:
                    w[f, j] = data["keypoints"][i, p, k]
            b = data["bbox"][i, p]
            bh[f] = b[3] - b[1]
        measured = np.isfinite(w[:, :, 0]).any(1)
        stride = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
        for j in range(2):
            w[:, j] = _interp_gaps(w[:, j], max_gap=max(2, 2 * stride))
        out.append(PlayerSignals(tr.id, tr.side, w, _upright_height(bh), measured, getattr(tr, "name", "")))
    return out


def _upright_height(bh: np.ndarray, window: int = 91) -> np.ndarray:
    """Standing body height in pixels around each frame: a high percentile of the
    box height in a sliding window (lunges and crouches shrink the box, a raised
    racket arm does not grow it much)."""
    n = len(bh)
    out = np.full(n, np.nan)
    if not np.isfinite(bh).any():
        return out
    h = window // 2
    for i in range(n):
        seg = bh[max(0, i - h):i + h + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.percentile(seg, 80)
    good = np.isfinite(out)
    out[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), out[good])
    return out


def _interp_gaps(xy: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over runs of NaN rows of up to `max_gap` frames (interior only)."""
    out = xy.copy()
    seen = np.flatnonzero(np.isfinite(xy[:, 0]))
    for a, b in zip(seen[:-1], seen[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a)[:, None] / (b - a)
            out[a + 1:b] = (1 - w) * xy[a] + w * xy[b]
    return out


# --- signals --------------------------------------------------------------------------------


def drop_static(shuttle: ShuttleTrack, min_frames: int = 12) -> ShuttleTrack:
    """Forget positions the track sits on for `min_frames` frames or more in total
    (exactly the same pixel): TrackNetV3's fixed "nothing found" point, a shuttle
    lying on the floor, a lamp. A shuttle in play never returns to the same pixel
    that often. Tracks from `badminton shuttle --court` have these removed already."""
    xy = shuttle.xy.copy()
    src = shuttle.source.copy()
    seen = np.flatnonzero(shuttle.seen)
    keys, inverse, counts = np.unique(xy[seen].round().astype(int), axis=0, return_inverse=True, return_counts=True)
    bad = seen[counts[inverse.ravel()] >= min_frames]
    xy[bad] = np.nan
    src[bad] = ""
    return ShuttleTrack(xy, src)


def local_scale(players: list[PlayerSignals], xy: np.ndarray, frame_h: float) -> np.ndarray:
    """(N,) pixels per body height where the shuttle is: the body height of the
    player whose wrist is nearest to it. The same swing moves the shuttle fewer
    pixels on the far side of the court; this puts both sides on one scale.
    Without players: a fixed fraction of the frame height."""
    n = len(xy)
    out = np.full(n, REF_BH_FRAC * frame_h)
    if not players:
        return out
    d = np.stack([wrist_distance(p, xy) for p in players])
    bh = np.stack([p.body_h for p in players])
    k = np.argmin(d, axis=0)
    near = bh[k, np.arange(n)]
    ok = np.isfinite(d.min(axis=0)) & np.isfinite(near)
    out[ok] = near[ok]
    if ok.any():  # carry the last known scale through frames without shuttle or pose
        idx = np.flatnonzero(ok)
        out[~ok] = np.interp(np.flatnonzero(~ok), idx, out[idx])
    return out


def kink_score(v_in: np.ndarray, v_out: np.ndarray) -> float:
    """>= 1 means "hit": velocity change or direction turn; velocities in body heights / s.

    Both flights must really move (MIN_SPEED): a shuttle that hangs at the top of a
    lift, tumbles at the net or stops on the floor turns without being hit."""
    s_in, s_out = float(np.hypot(*v_in)), float(np.hypot(*v_out))
    if min(s_in, s_out) < MIN_SPEED:
        return 0.0
    score = float(np.hypot(*(v_out - v_in))) / DV_THR
    cos = float(v_in @ v_out) / (s_in * s_out)
    if cos < TURN_COS:
        score = max(score, (1 - cos) * min(s_in, s_out) / TURN_THR)
    return score


def kink_series(shuttle: ShuttleTrack, fps: float, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per video frame: kink score of the gap-bridged track, and True where the
    score was taken across a gap of the raw track (contact not seen itself).
    `scale` is pixels per body height at each frame (local_scale)."""
    n = len(shuttle.xy)
    d = max(2, int(round(WINDOW_S * fps)))
    xy = _interp_gaps(shuttle.xy, int(round(INTERP_S * fps)))
    score = np.zeros(n)
    for i in range(d, n - d):
        if not (np.isfinite(xy[i - d, 0]) and np.isfinite(xy[i, 0]) and np.isfinite(xy[i + d, 0])):
            continue
        k = fps / d / scale[i]
        score[i] = kink_score((xy[i] - xy[i - d]) * k, (xy[i + d] - xy[i]) * k)
    # contacts inside a gap: bridging draws a straight line through it and flattens
    # the kink, so compare the flight before the gap with the flight after it
    in_gap = np.zeros(n, bool)
    seen = shuttle.seen
    idx = np.flatnonzero(seen)
    for a, b in zip(idx[:-1], idx[1:]):
        if not (2 <= b - a <= GAP_MAX_S * fps) or a < 2 or b + 2 >= n:
            continue
        if not (seen[a - 2:a + 1].all() and seen[b:b + 3].all()):
            continue
        mid = (a + b) // 2
        k = fps / 2 / scale[mid]
        s = kink_score((shuttle.xy[a] - shuttle.xy[a - 2]) * k, (shuttle.xy[b + 2] - shuttle.xy[b]) * k)
        if s > score[a:b + 1].max():
            score[mid] = s
            in_gap[mid] = True
    return score, in_gap


def wrist_speed(player: PlayerSignals, fps: float) -> np.ndarray:
    """(N,) faster of the two wrists, body heights per second (central difference)."""
    w = player.wrists
    v = np.zeros((len(w), 2))
    v[1:-1] = np.linalg.norm(w[2:] - w[:-2], axis=2) * fps / 2
    return np.nan_to_num(np.nan_to_num(v, nan=0.0).max(axis=1) / player.body_h, nan=0.0)


def wrist_distance(player: PlayerSignals, xy: np.ndarray) -> np.ndarray:
    """(N,) distance from the shuttle to the nearer wrist, in body heights (inf if unknown)."""
    d = np.linalg.norm(player.wrists - xy[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf).min(axis=1) / player.body_h
    return np.where(np.isfinite(d), d, np.inf)


def _peaks(x: np.ndarray, thr: float, min_gap: int) -> list[int]:
    """Indices of local maxima >= thr, strongest first, at least `min_gap` apart."""
    out: list[int] = []
    taken = np.zeros(len(x), bool)
    for i in np.argsort(-x, kind="stable"):
        if x[i] < thr:
            break
        if taken[max(0, i - min_gap):i + min_gap + 1].any():
            continue
        taken[i] = True
        out.append(int(i))
    return out


# --- detection ------------------------------------------------------------------------------


@dataclass
class Options:
    use_wrist: bool = True  # wrist speed as evidence and as a source of candidates
    alternate: bool = True  # consecutive hits of a rally come from opposite sides of the net
    reach_bh: float = REACH_BH
    wrist_thr: float = WRIST_THR
    weak: float = WEAK
    w_kink: float = W_KINK
    w_wrist: float = W_WRIST
    w_dist: float = W_DIST
    bias: float = BIAS
    same_side_cost: float = SAME_SIDE_COST


def candidates(shuttle: ShuttleTrack, players: list[PlayerSignals], fps: float, frame_h: float,
               opts: Options) -> list[dict]:
    """Every moment worth asking "was this a hit?", with the evidence per player.

    A candidate is a peak of the kink score (>= opts.weak) or, with use_wrist, a
    wrist-speed peak of a player who has the shuttle within reach. Each carries
    kink, in_gap and per player: dist_bh (shuttle to nearer wrist) and
    wrist (speed peak) within +-WRIST_SYNC_S."""
    n = len(shuttle.xy)
    gap = max(1, int(round(MIN_GAP_S * fps)))
    sync = max(1, int(round(WRIST_SYNC_S * fps)))
    xy = _interp_gaps(shuttle.xy, int(round(GAP_MAX_S * fps)))
    score, in_gap = kink_series(shuttle, fps, local_scale(players, xy, frame_h))
    speeds = [wrist_speed(p, fps) for p in players]
    dists = [wrist_distance(p, xy) for p in players]

    def window(a: np.ndarray, i: int) -> np.ndarray:
        return a[max(0, i - sync):i + sync + 1]

    frames = {i: "track" for i in _peaks(score, opts.weak, max(1, gap // 2))}
    edge = {}
    if players:
        # a flight that starts or ends in a player's hands: the track often loses the
        # shuttle against the body or in the blur of the stroke, and a serve has no
        # flight before it at all. No kink can be measured there; the edge itself,
        # next to a wrist, is the sign.
        seen = shuttle.seen
        scale = local_scale(players, xy, frame_h)
        lost = int(round(GAP_MAX_S * fps))
        idx = np.flatnonzero(seen)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > lost) + 1) if len(idx) else []:
            a, b = int(run[0]), int(run[-1])
            if b - a < 3:
                continue
            for i, j, kind in ((a, min(a + 3, b), "start"), (b, max(b - 3, a), "end")):
                speed = float(np.hypot(*(xy[j] - xy[i]))) * fps / abs(j - i) / scale[i]
                near = min(float(d[min(i, j):max(i, j) + 1].min()) for d in dists)
                if speed >= MIN_SPEED and near <= EDGE_BH and not any(abs(i - f) < gap for f in frames):
                    frames[i] = kind
                    edge[i] = EDGE_KINK[kind]
    if opts.use_wrist:
        for k in range(len(players)):
            for i in _peaks(speeds[k], opts.wrist_thr, gap):
                if window(dists[k], i).min() > opts.reach_bh:
                    continue
                if not any(abs(i - j) < gap for j in frames):
                    frames[i] = "wrist"
    out = []
    for i, origin in sorted(frames.items()):
        c = {"frame": i, "origin": origin, "kink": float(edge.get(i, score[i])),
             "in_gap": bool(in_gap[i]),
             "dist_bh": [float(window(d, i).min()) for d in dists],
             "wrist": [float(window(s, i).max()) for s in speeds]}
        c["shuttle_xy"] = None if not np.isfinite(xy[i, 0]) else [round(float(xy[i, 0]), 1), round(float(xy[i, 1]), 1)]
        c["shuttle_source"] = (shuttle.source[i] or INTERPOLATED) if c["shuttle_xy"] else ""
        out.append(c)
    return out


def evidence(c: dict, k: int | None, opts: Options) -> float:
    """> 0 means "player k hit the shuttle at candidate c" (k None: nobody tracked).

    A weighted sum on one scale: the kink (capped, a huge one is no more of a hit
    than a clear one), how fast that player's wrist moved and how far the shuttle
    was from that wrist; minus a bias."""
    s = opts.w_kink * min(c["kink"], KINK_CAP) - opts.bias
    if k is None:
        return s
    d = c["dist_bh"][k]
    if d > opts.reach_bh:
        return -np.inf
    if opts.use_wrist:
        s += opts.w_wrist * min(c["wrist"][k] / opts.wrist_thr, 2.0)
    return s - opts.w_dist * d


def detect_hits(shuttle: ShuttleTrack, players: list[PlayerSignals], fps: float, frame_h: float,
                opts: Options | None = None) -> list[dict]:
    """Hits as dicts (frame, evidence, hitter); see hits_json for the saved form.

    Picks the set of candidates with the largest total evidence such that two hits
    are at least MIN_GAP_S apart and, within a rally, come from opposite sides of
    the net in turn (true in singles and doubles; a pause longer than RALLY_GAP_S
    starts a new rally). Two hits in a row from one side are impossible when
    close together and cost evidence otherwise (then the detector missed the hit
    in between). The rule removes most swings that did not meet the shuttle (the
    follow-through of a stroke, the preparation for the next) and settles who hit
    when both players are close to the shuttle at the net."""
    opts = opts or Options()
    cands = candidates(shuttle, players, fps, frame_h, opts)
    gap = max(1, int(round(MIN_GAP_S * fps)))
    rally_gap = RALLY_GAP_S * fps
    same_side = SAME_SIDE_MIN_S * fps
    sides = sorted({p.side for p in players}) or [None]
    # per candidate and side: the best player on that side and the evidence
    ev = []
    for c in cands:
        row = {}
        for s in sides:
            ks = [k for k, p in enumerate(players) if p.side == s] if s is not None else [None]
            row[s] = max(((evidence(c, k, opts), k) for k in ks), key=lambda t: t[0])
        ev.append(row)
    # best[j][s]: largest total of a selection whose last hit is candidate j by side s
    n = len(cands)
    best = [{s: -np.inf for s in sides} for _ in range(n)]
    prev = [{s: None for s in sides} for _ in range(n)]
    for j in range(n):
        for s in sides:
            e = ev[j][s][0]
            if not e > 0:
                continue
            best[j][s] = e
            for i in range(j):
                dt = cands[j]["frame"] - cands[i]["frame"]
                if dt < gap:
                    continue
                for s0 in sides:
                    cost = 0.0
                    if opts.alternate and s is not None and s0 == s and dt <= rally_gap:
                        if dt < same_side:
                            continue  # no time for the shuttle to cross the net and come back
                        cost = opts.same_side_cost  # the other side's hit in between was missed
                    if best[i][s0] + e - cost > best[j][s]:
                        best[j][s], prev[j][s] = best[i][s0] + e - cost, (i, s0)
    end = max(((best[j][s], j, s) for j in range(n) for s in sides), key=lambda t: t[0], default=None)
    hits: list[dict] = []
    if end is None or not np.isfinite(end[0]):
        return hits
    node = (end[1], end[2])
    while node is not None:
        j, s = node
        e, k = ev[j][s]
        c = cands[j]
        h = {"frame": c["frame"], "kink": c["kink"], "in_gap": c["in_gap"],
             "evidence": float(e), "shuttle_xy": c["shuttle_xy"], "shuttle_source": c["shuttle_source"],
             "wrist": 0.0, "dist_bh": float("nan"), "player": k}
        if k is not None:
            p = players[k]
            h.update(wrist=c["wrist"][k], dist_bh=c["dist_bh"][k], player_id=p.id, side=("near", "far")[p.side],
                     pose_measured=bool(p.measured[max(0, c["frame"] - 1):c["frame"] + 2].any()))
        if c["origin"] in ("start", "end"):
            parts = ["track-" + c["origin"]]
        else:
            parts = ["track"] if c["kink"] >= 1.0 else (["track(weak)"] if c["kink"] >= opts.weak else [])
        if opts.use_wrist and k is not None and h["wrist"] >= opts.wrist_thr:
            parts.append("wrist")
        h["by"] = "+".join(parts) or "weak"
        hits.append(h)
        node = prev[j][s]
    hits.reverse()
    return hits


def hits_json(hits: list[dict], fps: float, video: str = "", inputs: dict | None = None,
              names: dict | None = None) -> dict:
    """The saved form: every value with its source (measured / interpolated / derived / manual)."""
    names = names or {}
    out = []
    for h in hits:
        src = h.get("source", DERIVED)
        e = {
            "frame": {"value": int(h["frame"]), "source": src},
            "time_s": {"value": round(h["frame"] / fps, 3), "source": src},
            "found_by": h["by"],
        }
        if "evidence" in h:
            e["signals"] = {"evidence": round(h["evidence"], 2), "kink": round(h["kink"], 2),
                            "kink_across_gap": h["in_gap"], "wrist_speed_bh_per_s": round(h["wrist"], 2)}
            e["shuttle_xy"] = {"value": h["shuttle_xy"], "source": h["shuttle_source"] or None}
        if h.get("player_id") is not None:
            e["player"] = {"value": h["player_id"], "name": names.get(str(h["player_id"]), ""),
                           "side": h.get("side"), "source": src}
            if "dist_bh" in h:
                e["player"]["wrist_distance_bh"] = round(h["dist_bh"], 2)
                e["player"]["wrist_source"] = MEASURED if h["pose_measured"] else INTERPOLATED
        else:
            e["player"] = {"value": None, "source": None}
        out.append(e)
    return {"schema": SCHEMA, "video": video, "fps": fps, "inputs": inputs or {},
            "sources": legend(), "hits": out}


def manual_hits(labels: list[dict], side_of: dict | None = None) -> list[dict]:
    """Hits a person entered or confirmed (the review page's CSV): source "manual"."""
    side_of = side_of or {}
    return [{"frame": l["frame"], "by": "manual", "source": MANUAL, "player_id": l["player_id"],
             "side": side_of.get(l["player_id"])} for l in labels if l.get("note") != "ignore"]


def write_csv(hits: list[dict], fps: float, path: str | Path) -> None:
    """One row per hit: frame, time_s, player, side, found_by, source (for tools that only need the frames)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "player", "side", "found_by", "source"])
        for h in hits:
            w.writerow([h["frame"], round(h["frame"] / fps, 3), h.get("player_id", ""), h.get("side") or "",
                        h["by"], h.get("source", DERIVED)])


def review_page(doc: dict, path: str | Path, names: dict | None = None) -> None:
    """A single HTML file to check the hits against the video by eye and keyboard
    (confirm, reject, move by a frame, add a missed one) and export them as labels."""
    from importlib import resources

    html = resources.files("badminton").joinpath("hits_review.html").read_text(encoding="utf-8")
    a, b = html.index("/*DOC*/"), html.index("/*END*/")
    small = {"fps": doc["fps"], "video": Path(doc.get("video", "")).name, "players": names or {},
             "hits": [{"frame": h["frame"], "player": {"value": h["player"]["value"]}, "found_by": h["found_by"]}
                      for h in doc["hits"]]}
    payload = json.dumps(small, ensure_ascii=False).replace("</", "<\\/")
    Path(path).write_text(html[:a] + payload + html[b + len("/*END*/"):], encoding="utf-8")


# --- reference labels and scoring -----------------------------------------------------------


def load_labels(path: str | Path) -> list[dict]:
    """Reference hits: CSV with Frame, Player (track id, may be empty), optional Note."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("Frame", "").strip() and not r["Frame"].startswith("#")]
    return [{"frame": int(r["Frame"]), "player_id": int(r["Player"]) if r.get("Player", "").strip() else None,
             "note": (r.get("Note") or "").strip()} for r in rows]


def score_hits(hits: list[dict], labels: list[dict], tol_frames: int) -> dict:
    """Greedy one-to-one matching by frame distance (closest pairs first) within
    `tol_frames`; precision, recall, F1 and how often the hitter is right.

    A label with Note "ignore" (a shuttle knocked back between rallies, a contact
    cut off by the end of the video) is neither required nor punished: a detection
    matched to it is dropped from the count."""
    skip = [l["frame"] for l in labels if l.get("note") == "ignore"]
    labels = [l for l in labels if l.get("note") != "ignore"]
    hits = [h for h in hits if not any(abs(h["frame"] - f) <= tol_frames for f in skip)]
    pairs = sorted((abs(h["frame"] - l["frame"]), a, b) for a, h in enumerate(hits)
                   for b, l in enumerate(labels) if abs(h["frame"] - l["frame"]) <= tol_frames)
    used_h, used_l, matched = set(), set(), []
    for _, a, b in pairs:
        if a in used_h or b in used_l:
            continue
        used_h.add(a), used_l.add(b)
        matched.append((a, b))
    tp, fp, fn = len(matched), len(hits) - len(matched), len(labels) - len(matched)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    who = [(hits[a].get("player_id"), labels[b]["player_id"]) for a, b in matched if labels[b]["player_id"] is not None]
    return {"labels": len(labels), "detected": len(hits), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "hitter_checked": len(who), "hitter_right": sum(a == b for a, b in who),
            "missed_frames": sorted(labels[b]["frame"] for b in range(len(labels)) if b not in used_l),
            "extra_frames": sorted(hits[a]["frame"] for a in range(len(hits)) if a not in used_h),
            "offset_frames": [hits[a]["frame"] - labels[b]["frame"] for a, b in sorted(matched)]}


def run(shuttle_csv, skeleton_npz, court=None, players_per_side: int = 1, opts: Options | None = None,
        names: dict | None = None) -> tuple[list[dict], dict]:
    """Files in, hits out. Returns (hits, context: fps, frame_h, shuttle, players, tracking)."""
    from .tracking import load_npz, track_players

    data = load_npz(skeleton_npz)
    fps = float(data["fps"])
    frame_h = float(data["frame_size"][1])
    shuttle = drop_static(load_shuttle(shuttle_csv, int(data["frame"][-1]) + 1))
    tracking = track_players(data, court, players_per_side)
    players = player_signals(data, tracking, len(shuttle.xy))
    for p in players:
        p.name = (names or {}).get(str(p.id), p.name)
    hits = detect_hits(shuttle, players, fps, frame_h, opts)
    return hits, {"fps": fps, "frame_h": frame_h, "shuttle": shuttle, "players": players, "tracking": tracking,
                  "stride": int(np.median(np.diff(data["frame"]))) if len(data["frame"]) > 1 else 1}


def save(path: str | Path, doc: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
