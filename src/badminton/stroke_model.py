"""Doubles stroke type from where the four players stand, the rhythm of the rally,
the hitter's pose and the shuttle's path: a small model trained on DShuttleNet.

DShuttleNet (58 BWF doubles matches, CC BY-NC-SA 4.0, non-commercial) labels every
shot with one of nine types and where the four players stood. Two inputs per hit:

* context (CTX_DIM numbers): the four players' floor positions at this hit, at the
  next one and at the one before (hitter, partner, front opponent, back opponent;
  mirrored so the hitter's half is always the near one), the time to those hits and
  who made them (only when they came from the other side: otherwise a hit in between
  was missed), and whether the rally starts here. All of it is in the labels
  themselves, so this part is learnt from all the matches without any video;
* sequence (SEQ_LEN samples at 30 per second around the hit, SEQ_DIM numbers each):
  the hitter's 17 joints, the shuttle relative to the hitter and to the court in
  the picture, and the four floor positions. This needs our own skeleton, shuttle
  track and court calibration, so it is learnt from the matches we have video of.

`scripts/train_stroke_classifier.py` trains it and scores it per held-out match.
The weights (STROKE_MODEL_FILE in the badminton cache folder) are not in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .court import LENGTH, WIDTH, CourtCalibration

STROKE_MODEL_FILE = "stroke_classifier_dshuttlenet.pt"
CLASSES = ["short serve", "long serve", "net shot", "push", "lift", "drive", "smash", "drop", "clear"]
# DShuttleNet type -> (our name, group in strokes.py, ShuttleSet types that count as the same)
NAMES = {
    "short serve": ("发短球", "发球", ("發短球",)),
    "long serve": ("发长球", "发球", ("發長球",)),
    "net shot": ("放网", "放网", ("放小球", "擋小球", "勾球")),
    "push": ("推球", "推扑", ("推球", "撲球")),
    "lift": ("挑球", "挑球", ("挑球", "防守回挑")),
    "drive": ("平抽", "抽挡", ("平球", "後場抽平球", "防守回抽")),
    "smash": ("杀球", "杀球", ("殺球", "點扣")),
    "drop": ("吊球", "吊球", ("切球", "過渡切球")),
    "clear": ("高远球", "高远/平高", ("長球",)),
}
SHUTTLESET_TO_CLASS = {s: CLASSES.index(k) for k, v in NAMES.items() for s in v[2]}

NEIGHBOUR_MAX_S = 3.0  # a hit further away in time than this is not the next / previous one of the rally
SEQ_LEN = 30
SEQ_BEFORE = 10  # samples before the hit; the rest (the shuttle leaving) come after
N_JOINTS = 17
SEQ_DIM = 2 * N_JOINTS + 1 + 2 + 2 + 1 + 4 * 3
CTX_DIM = 4 * 3 + 2 * (3 + 4 * 3 + 4) + 1
_FLIP_JOINTS = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # COCO left <-> right


# ---------------------------------------------------------------- context


@dataclass
class Shot:
    """One hit as the context features see it. Keys are whatever names the players
    (track ids, or A–D in the labels); `team` gives each one's side of the net."""

    t: float  # seconds
    hitter: object
    pos: dict  # key -> (x, y) metres on the court, missing or NaN when unknown
    team: dict  # key -> 0 / 1


def _xy(pos: dict, key) -> np.ndarray | None:
    if key is None or key not in pos:
        return None
    q = np.asarray(pos[key], float)
    if not np.isfinite(q).all() or not (-3 < q[0] < WIDTH + 3 and -4 < q[1] < LENGTH + 4):
        return None
    return q


def roles(shot: Shot) -> tuple[list, bool] | None:
    """([hitter, partner, front opponent, back opponent], flip): who is who at this hit, and
    whether the court is turned round so the hitter's half is the near one. None when the
    hitter has no position. Missing players are None."""
    h = _xy(shot.pos, shot.hitter)
    if h is None:
        return None
    flip = bool(h[1] > LENGTH / 2)
    mine = shot.team.get(shot.hitter)
    mates = [k for k in shot.pos if k != shot.hitter and shot.team.get(k) == mine and _xy(shot.pos, k) is not None]
    opps = [k for k in shot.pos if shot.team.get(k) != mine and _xy(shot.pos, k) is not None]

    def depth(k):  # distance from the net on their own half
        return abs(_xy(shot.pos, k)[1] - LENGTH / 2)

    opps = sorted(opps, key=depth)[:2]
    mate = min(mates, key=lambda k: float(np.linalg.norm(_xy(shot.pos, k) - h))) if mates else None
    return [shot.hitter, mate, *(opps + [None, None])[:2]], flip


def _norm(q: np.ndarray, flip: bool) -> tuple[float, float]:
    x, y = (WIDTH - q[0], LENGTH - q[1]) if flip else (q[0], q[1])
    return (x - WIDTH / 2) / 3.0, (y - LENGTH / 2) / 6.0


def _block(shot: Shot, order: list, flip: bool) -> list[float]:
    out: list[float] = []
    for k in order:
        q = _xy(shot.pos, k)
        out.extend([*_norm(q, flip), 1.0] if q is not None else [0.0, 0.0, 0.0])
    return out


def context_features(shots: list[Shot], i: int) -> np.ndarray | None:
    """CTX_DIM numbers for shots[i] (shots in time order), None when the hitter has no position."""
    s = shots[i]
    r = roles(s)
    if r is None:
        return None
    order, flip = r
    f = _block(s, order, flip)
    first = True
    mine = s.team.get(s.hitter)
    for j, sign in ((i + 1, 1), (i - 1, -1)):
        dt = sign * (shots[j].t - s.t) if 0 <= j < len(shots) else -1.0
        near = 0 < dt < NEIGHBOUR_MAX_S
        first &= not (near and sign < 0)
        # a neighbour from the hitter's own side means the opponents' hit in between was missed:
        # its timing and positions say nothing about this shot, so it counts as unknown
        if near and shots[j].hitter is not None and shots[j].team.get(shots[j].hitter) != mine:
            t = shots[j]
            f.extend([1.0, dt, float(np.log(dt + 0.05))])
            f.extend(_block(t, order, flip))
            f.extend([float(t.hitter == k) for k in order])
        else:
            f.extend([0.0] * (3 + 12 + 4))
    f.append(float(first))
    return np.asarray(f, np.float32)


def mirror_context(x: np.ndarray) -> np.ndarray:
    """Left-right mirror image of context features (..., CTX_DIM)."""
    x = x.copy()
    for start in (0, 12 + 3, 12 + 19 + 3):
        x[..., start:start + 12:3] *= -1
    return x


# ---------------------------------------------------------------- sequence


class Scene:
    """The skeleton .npz, the player tracks, the shuttle track and the court of one video,
    laid out for per-hit features."""

    def __init__(self, data: dict, tracking, shuttle_px: np.ndarray, calib: CourtCalibration):
        self.data, self.calib = data, calib
        self.fps = float(data["fps"])
        frames = np.asarray(data["frame"], int)
        self.n = int(frames[-1]) + 1
        idx = np.clip(np.searchsorted(frames, np.arange(self.n)), 0, len(frames) - 1)
        left = np.maximum(idx - 1, 0)
        pick = np.where(np.abs(frames[left] - np.arange(self.n)) <= np.abs(frames[idx] - np.arange(self.n)), left, idx)
        self.row_of = np.where(np.abs(frames[pick] - np.arange(self.n)) <= 1, pick, -1)  # video frame -> npz row
        self.tracks = {t.id: t for t in tracking.tracks}
        self.team = {t.id: int(t.side) for t in tracking.tracks}
        self.shuttle = np.full((self.n, 2), np.nan)
        m = min(self.n, len(shuttle_px))
        self.shuttle[:m] = shuttle_px[:m]
        mid = calib.to_image(np.array([[WIDTH / 2, LENGTH / 2], [WIDTH / 2, 0.0], [WIDTH / 2, LENGTH]]))
        self.court_centre, self.court_scale = mid[0], float(np.linalg.norm(mid[1] - mid[2])) or 1.0

    def position(self, tid, frame: int) -> np.ndarray | None:
        t = self.tracks.get(tid)
        r = self.row_of[frame] if 0 <= frame < self.n else -1
        if t is None or r < 0:
            return None
        xy = t.xy[r]
        if not np.isfinite(xy).all():
            xy = t.raw_xy[r]
        return xy if np.isfinite(xy).all() else None

    def shot(self, frame: int, hitter) -> Shot:
        pos = {tid: p for tid in self.tracks if (p := self.position(tid, frame)) is not None}
        return Shot(frame / self.fps, hitter, pos, self.team)

    def sample_frames(self, frame: int) -> np.ndarray:
        return np.round(frame + (np.arange(SEQ_LEN) - SEQ_BEFORE) * self.fps / 30.0).astype(int)

    def sequence(self, frame: int, order: list, flip: bool) -> np.ndarray:
        """(SEQ_LEN, SEQ_DIM) around a hit; `order`, `flip` from roles()."""
        out = np.zeros((SEQ_LEN, SEQ_DIM), np.float32)
        hitter = self.tracks.get(order[0])
        for k, f in enumerate(self.sample_frames(frame)):
            if not 0 <= f < self.n:
                continue
            r = self.row_of[f]
            sh = self.shuttle[f]
            seen = bool(np.isfinite(sh).all())
            d = int(hitter.det_index[r]) if hitter is not None and r >= 0 else -1
            if d >= 0:
                kp, bb = self.data["keypoints"][r, d], self.data["bbox"][r, d]
                centre, h = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]), max(float(bb[3] - bb[1]), 1.0)
                out[k, :34] = ((kp - centre) / h).reshape(-1)
                out[k, 34] = 1.0
                if seen:
                    out[k, 35:37] = np.clip((sh - centre) / h, -6, 6)
            if seen:
                out[k, 37:39] = (sh - self.court_centre) / self.court_scale
                out[k, 39] = 1.0
            for j, tid in enumerate(order):
                q = self.position(tid, f) if tid is not None else None
                if q is not None:
                    out[k, 40 + 3 * j:43 + 3 * j] = [*_norm(q, flip), 1.0]
        return out


def mirror_sequence(x: np.ndarray) -> np.ndarray:
    """Left-right mirror image of sequence features (..., SEQ_LEN, SEQ_DIM)."""
    x = x.copy()
    j = x[..., :34].reshape(*x.shape[:-1], N_JOINTS, 2)[..., _FLIP_JOINTS, :].copy()
    j[..., 0] *= -1
    x[..., :34] = j.reshape(*x.shape[:-1], 34)
    x[..., 35] *= -1
    x[..., 37] *= -1
    x[..., 40:52:3] *= -1
    return x


def hit_features(scene: Scene, hits: list[dict]) -> list[dict | None]:
    """Per hit ({"frame", "player_id"}, in time order): {"ctx", "seq", "far"} or None when the
    hitter is not a tracked player with a position."""
    shots = [scene.shot(int(h["frame"]), h.get("player_id")) for h in hits]
    out: list[dict | None] = []
    for i, s in enumerate(shots):
        ctx = context_features(shots, i)
        if ctx is None:
            out.append(None)
            continue
        order, flip = roles(s)
        out.append({"ctx": ctx, "seq": scene.sequence(int(hits[i]["frame"]), order, flip), "far": float(flip)})
    return out


# ---------------------------------------------------------------- model


def build_model(n_class: int = len(CLASSES), width: int = 128, hidden: int = 256):
    import torch
    from torch import nn

    class StrokeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.ctx = nn.Sequential(nn.Linear(CTX_DIM, hidden), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1))
            self.ctx_head = nn.Linear(hidden, n_class)
            self.embed = nn.Linear(SEQ_DIM, width)
            self.when = nn.Parameter(torch.zeros(SEQ_LEN, width))
            layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.Linear(width + hidden + 1, hidden), nn.GELU(), nn.Dropout(0.2),
                                      nn.Linear(hidden, n_class))

        def forward(self, ctx, seq=None, far=None):
            """Logits. Without `seq`: from the context alone."""
            c = self.ctx(ctx)
            if seq is None:
                return self.ctx_head(c)
            s = self.encoder(self.embed(seq) + self.when).mean(1)
            return self.head(torch.cat([s, c, far[:, None]], 1))

    return StrokeNet()


class StrokeClassifier:
    def __init__(self, model, meta: dict | None = None):
        self.model, self.meta = model.eval(), meta or {}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "StrokeClassifier | None":
        """The model in `path` (default: the badminton cache folder), None if there is none."""
        import pickle

        import torch

        if path is None:
            from .coreml_native import CACHE_DIR

            path = CACHE_DIR / STROKE_MODEL_FILE
        try:
            doc = torch.load(path, map_location="cpu", weights_only=True)
            if doc.get("classes") != CLASSES:
                return None
            model = build_model()
            model.load_state_dict(doc["state"])
        except (OSError, KeyError, RuntimeError, AttributeError, ValueError, pickle.UnpicklingError):
            return None
        return cls(model, doc.get("meta"))

    def save(self, path: str | Path) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"classes": CLASSES, "state": self.model.state_dict(), "meta": self.meta}, path)

    def proba(self, feats: list[dict | None]) -> np.ndarray:
        """(n, classes); rows of NaN for hits without features."""
        import torch

        out = np.full((len(feats), len(CLASSES)), np.nan)
        keep = [i for i, f in enumerate(feats) if f is not None]
        if keep:
            ctx = torch.from_numpy(np.stack([feats[i]["ctx"] for i in keep]))
            seq = torch.from_numpy(np.stack([feats[i]["seq"] for i in keep]))
            far = torch.tensor([feats[i]["far"] for i in keep], dtype=torch.float32)
            with torch.no_grad():
                out[keep] = self.model(ctx, seq, far).softmax(-1).numpy()
        return out
