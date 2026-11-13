"""Stroke types with BST (Badminton Stroke-type Transformer), from our own pose and shuttle data.

BST: Chang, "BST: Badminton Stroke-type Transformer for Skeleton-based Action
Recognition in Racket Sports", CVPR 2026 workshops, MIT licence,
https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer

BST classifies one stroke from a short clip around the hit. It takes three
things per frame, for exactly two players (far player first, then near player):

  joints   17 COCO keypoints, relative to the centre of the player's box and
           divided by the box diagonal            <- `badminton skeleton`
  pos      midpoint of the two ankles on the court, 0..1 across (left to right
           in the image) and 0..1 along (far baseline to near baseline),
           measured on the doubles court           <- court calibration
  shuttle  pixel position divided by the video size, (0, 0) when not seen
           <- `badminton shuttle`

This file turns our files into those arrays the way BST's own
prepare_train_on_shuttleset.py does (so the released weights see what they
were trained on), holds a copy of the BST-CG-AP network that loads the released
weights as they are, and has a simple hit finder to cut clips until the proper
hit detection lands.

torch is only imported by the functions that run the network.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .court import LENGTH, WIDTH, CourtCalibration

SEQ_LEN = 30  # the released "seq_len=30" weights: one second around the hit at 30 fps
N_JOINTS = 17
FAR, NEAR = 1, 0  # `side` in skeleton.npz
IN_COURT_EPS = 0.01

# COCO bones, in BST's order (get_bone_pairs)
BONE_PAIRS = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# The 12 merged ShuttleSet stroke types, in BST's order (get_merged_stroke_types):
# (ShuttleSet name, simplified Chinese, English)
STROKES = [
    ("放小球", "放网前球", "net shot"),
    ("擋小球", "挡网前球", "return net"),
    ("殺球", "杀球", "smash"),
    ("挑球", "挑球", "lob"),
    ("長球", "高远球", "clear"),
    ("平球", "平抽球", "drive"),
    ("切球", "吊球", "drop"),
    ("推球", "推球", "push"),
    ("撲球", "扑球", "rush"),
    ("勾球", "勾对角", "cross-court net shot"),
    ("發短球", "发短球", "short service"),
    ("發長球", "发长球", "long service"),
]
N_CLASSES = 1 + 2 * len(STROKES)  # 0 = unknown, 1..12 = far player, 13..24 = near player


def class_name(idx: int, lang: str = "zh") -> tuple[str | None, str]:
    """(who, stroke) for a class index of the 25-class model; who is 'far', 'near' or None."""
    k = {"raw": 0, "zh": 1, "en": 2}[lang]
    if idx == 0:
        return None, ("未知球種", "未知", "unknown")[k]
    who = "far" if idx <= len(STROKES) else "near"
    return who, STROKES[(idx - 1) % len(STROKES)][k]


# ---------------------------------------------------------------- our files -> per-frame arrays


@dataclass
class Sequence:
    """Per video frame, already normalised the way BST expects.

    A frame is `ok` when both players were found inside the court; other
    frames are all zeros, shuttle included (as in BST's training data).
    """

    fps: float
    joints: np.ndarray  # (F, 2, 17, 2)  far player, near player
    pos: np.ndarray  # (F, 2, 2)
    shuttle: np.ndarray  # (F, 2)
    ok: np.ndarray  # (F,) bool
    shuttle_px: np.ndarray  # (F, 2) video pixels, NaN when not seen (for the hit finder)
    wrists_px: np.ndarray  # (F, 2, 2, 2) far/near, left/right wrist, NaN when missing
    height_px: np.ndarray  # (F, 2) player box height, NaN when missing


def read_shuttle_csv(path: str | Path, n_frames: int) -> np.ndarray:
    """(n_frames, 2) pixel positions from a TrackNetV3-style CSV; NaN where not visible."""
    out = np.full((n_frames, 2), np.nan)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["Frame"])
            if 0 <= i < n_frames and int(float(row["Visibility"])):
                out[i] = float(row["X"]), float(row["Y"])
    return out


def normalize_joints(kp: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """BST's normalize_joints(center_align=True): kp (..., 17, 2), bbox (..., 4) xyxy."""
    tl, br = bbox[..., None, :2], bbox[..., None, 2:]
    diag = np.linalg.norm(br - tl, axis=-1, keepdims=True)
    rel = np.where(kp != 0.0, (kp - tl) / diag, 0.0)
    return rel - (br - tl) / 2 / diag


def court_position(ankles_px: np.ndarray, calib: CourtCalibration) -> np.ndarray:
    """Ankles (..., 2, 2) in pixels -> BST's position (..., 2): ankles to the court, then their midpoint.

    Our court has y = 0 at the near baseline; BST's has 0 at the far one."""
    court = calib.to_court(ankles_px.reshape(-1, 2)).reshape(ankles_px.shape).mean(axis=-2)
    return np.stack([court[..., 0] / WIDTH, 1.0 - court[..., 1] / LENGTH], axis=-1)


def build_sequence(skel: dict, shuttle_px: np.ndarray, calib: CourtCalibration) -> Sequence:
    """skel: a skeleton.npz made with --stride 1 (one row per video frame) and a court calibration."""
    frames = np.asarray(skel["frame"])
    if len(frames) > 1 and not np.all(np.diff(frames) == 1):
        raise ValueError("skeleton.npz skips frames: run `badminton skeleton --stride 1` (BST needs every frame)")
    F = len(frames)
    if len(shuttle_px) < F:
        raise ValueError(f"shuttle track has {len(shuttle_px)} frames, skeleton has {F}")
    shuttle_px = shuttle_px[frames[0]:frames[0] + F] if frames[0] else shuttle_px[:F]
    w, h = (float(v) for v in skel["frame_size"])
    valid, side = skel["valid"], skel["side"]

    joints = np.zeros((F, 2, N_JOINTS, 2))
    pos = np.zeros((F, 2, 2))
    wrists = np.full((F, 2, 2, 2), np.nan)
    height = np.full((F, 2), np.nan)
    found = np.zeros((F, 2), bool)
    for m, s in enumerate((FAR, NEAR)):
        cand = valid & (side == s)
        found[:, m] = cand.any(axis=1)
        slot = cand.argmax(axis=1)
        idx = np.flatnonzero(found[:, m])
        kp = skel["keypoints"][idx, slot[idx]]
        bb = skel["bbox"][idx, slot[idx]]
        joints[idx, m] = normalize_joints(kp, bb)
        pos[idx, m] = court_position(kp[:, 15:17], calib)
        wrists[idx, m] = kp[:, 9:11]
        height[idx, m] = bb[:, 3] - bb[:, 1]

    inside = ((pos > -IN_COURT_EPS) & (pos < 1 + IN_COURT_EPS)).all(axis=-1)
    ok = (found & inside).all(axis=1)
    shuttle = np.where(np.isfinite(shuttle_px), shuttle_px / (w, h), 0.0)
    joints[~ok] = 0
    pos[~ok] = 0
    shuttle[~ok] = 0
    return Sequence(float(skel["fps"]), joints, pos, shuttle, ok, shuttle_px, wrists, height)


# ---------------------------------------------------------------- hits (simple, until the real detector)


@dataclass
class Hit:
    frame: int
    who: str  # 'far' or 'near': the player whose wrist is closest to the shuttle
    turn: float  # change of shuttle velocity, px/frame scaled to 1080p at 30 fps
    reach: float  # wrist-to-shuttle distance in player-box heights


def find_hits(seq: Sequence, frame_height: float, min_turn: float = 12.0, max_reach: float = 1.0,
              min_gap_s: float = 0.3, alternate: bool = True) -> list[Hit]:
    """Frames where the shuttle's velocity changes abruptly next to a player.

    Rough on purpose: velocity before and after each frame over two frames,
    peaks of the change above `min_turn`, kept when a wrist is within
    `max_reach` box heights (a racket and an arm). With `alternate`, of two
    neighbouring hits by the same player only the sharper one stays.
    """
    xy = seq.shuttle_px
    F = len(xy)
    k = max(1, round(seq.fps / 15))  # two frames at 30 fps
    scale = (1080.0 / frame_height) * (seq.fps / 30.0)
    turn = np.zeros(F)
    a, b, c = xy[:-2 * k], xy[k:-k], xy[2 * k:]
    d = np.linalg.norm((c - b) - (b - a), axis=1) / k * scale
    turn[k:F - k] = np.where(np.isfinite(d), d, 0.0)

    gap = max(1, round(min_gap_s * seq.fps))
    hits: list[Hit] = []
    for f in np.argsort(-turn):
        if turn[f] < min_turn:
            break
        if any(abs(f - h.frame) < gap for h in hits):
            continue
        dist = np.linalg.norm(seq.wrists_px[f] - xy[f], axis=-1).min(axis=-1) / seq.height_px[f]  # (2,)
        if not np.isfinite(dist).any():
            continue
        m = int(np.nanargmin(dist))
        if dist[m] > max_reach:
            continue
        hits.append(Hit(int(f), ("far", "near")[m], float(turn[f]), float(dist[m])))
    hits.sort(key=lambda h: h.frame)
    if alternate:
        out: list[Hit] = []
        for h in hits:
            if out and out[-1].who == h.who and h.frame - out[-1].frame < 1.5 * seq.fps:
                if h.turn > out[-1].turn:
                    out[-1] = h
            else:
                out.append(h)
        hits = out
    return hits


# ---------------------------------------------------------------- clips -> model input


def make_seq_len_same(target: int, joints: np.ndarray, pos: np.ndarray, shuttle: np.ndarray):
    """BST's make_seq_len_same: subsample longer clips by a whole stride, zero-pad shorter ones."""
    n = len(pos)
    if n > target:
        pad = (n % target) > (target // 2)
        stride = n // target + int(pad)
        joints, pos, shuttle = (a[::stride][:target] for a in (joints, pos, shuttle))
        n = len(pos)
    if n < target:
        p = target - n
        joints = np.pad(joints, ((0, p), (0, 0), (0, 0), (0, 0)))
        pos = np.pad(pos, ((0, p), (0, 0), (0, 0)))
        shuttle = np.pad(shuttle, ((0, p), (0, 0)))
    return joints, pos, shuttle, n


def create_bones(joints: np.ndarray) -> np.ndarray:
    """(t, m, 17, 2) -> (t, m, 19, 2): end minus start, zero when either joint is exactly zero."""
    out = []
    for s, e in BONE_PAIRS:
        a, b = joints[:, :, s], joints[:, :, e]
        out.append(np.where((a != 0.0) & (b != 0.0), b - a, 0.0))
    return np.stack(out, axis=-2)


def clip_window(hit_frame: int, fps: float, n_frames: int) -> tuple[int, int]:
    """Half a second each side of the hit (BST's 'middle_in_a_sec' clips), cut at the video ends."""
    t = int(fps // 2)
    return max(0, hit_frame - t), min(n_frames, hit_frame + t)


def model_inputs(seq: Sequence, hit_frames: list[int], seq_len: int = SEQ_LEN) -> dict[str, np.ndarray]:
    """Arrays for the network, one row per hit:
    pose (n, seq_len, 2, 72) joints then bones ('JnB_bone'), pos (n, seq_len, 2, 2),
    shuttle (n, seq_len, 2), length (n,)."""
    pose, pos, shuttle, length = [], [], [], []
    for f in hit_frames:
        a, b = clip_window(f, seq.fps, len(seq.ok))
        j, p, s, n = make_seq_len_same(seq_len, seq.joints[a:b], seq.pos[a:b], seq.shuttle[a:b])
        jb = np.concatenate([j, create_bones(j)], axis=-2)
        pose.append(jb.reshape(seq_len, 2, -1))
        pos.append(p)
        shuttle.append(s)
        length.append(n)
    return {
        "pose": np.asarray(pose, np.float32).reshape(-1, seq_len, 2, 2 * (N_JOINTS + len(BONE_PAIRS))),
        "pos": np.asarray(pos, np.float32).reshape(-1, seq_len, 2, 2),
        "shuttle": np.asarray(shuttle, np.float32).reshape(-1, seq_len, 2),
        "length": np.asarray(length, np.int64),
    }


# ---------------------------------------------------------------- the network (BST-CG-AP)


def build_model(seq_len: int = SEQ_LEN, n_class: int = N_CLASSES, in_dim: int = 72):
    """BST-CG-AP with the layer names of the original, so its state_dict loads strictly.

    Same computation as BST_CG_AP in the BST repository (model/bst.py and
    model/tempose.py), written for inference: the positional embeddings come
    from the weights file, so the `positional_encodings` package is not needed.
    """
    import torch
    from torch import nn

    d, d_head, n_head, drop = 100, 128, 6, 0.3

    class MLP(nn.Module):
        def __init__(self, i, o, h):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, o))

        def forward(self, x):
            return self.mlp(x)

    class Head(nn.Module):
        def __init__(self, i, o, h):
            super().__init__()
            self.layer_norm = nn.LayerNorm(i)
            self.mlp = MLP(i, o, h)

        def forward(self, x):
            return self.mlp(self.layer_norm(x))

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP(d, d, 4 * d)

        def forward(self, x):
            return self.mlp(x)

    def attend(q, kv, mask, tail):
        # q (b, t, heads * d_head); kv (b, t, heads * 2 * d_head), key and value side by side per head
        b, t, _ = q.shape
        q = q.view(b, t, n_head, -1).transpose(1, 2)
        k, v = (x.transpose(1, 2) for x in kv.view(b, t, n_head, -1).chunk(2, dim=-1))
        dots = (q @ k.transpose(-1, -2)) * d_head ** -0.5
        dots = dots.masked_fill(~mask.view(b, 1, 1, t), -torch.inf)
        return tail((dots.softmax(-1) @ v).transpose(1, 2).reshape(b, t, -1))

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_qkv = nn.Linear(d, 3 * d_head * n_head, bias=False)
            self.tail = nn.Sequential(nn.Linear(d_head * n_head, d), nn.Dropout(drop))

        def forward(self, x, mask):
            b, t, _ = x.shape
            qkv = self.to_qkv(x).view(b, t, n_head, 3 * d_head)  # per head: query, key, value
            return attend(qkv[..., :d_head].reshape(b, t, -1), qkv[..., d_head:].reshape(b, t, -1), mask, self.tail)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm1 = nn.LayerNorm(d)
            self.attn = Attention()
            self.layer_norm2 = nn.LayerNorm(d)
            self.ff = FeedForward()

        def forward(self, x, mask):
            x = self.attn(self.layer_norm1(x), mask) + x
            return self.ff(self.layer_norm2(x)) + x

    class Encoder(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.layers = nn.ModuleList(Layer() for _ in range(depth))

        def forward(self, x, mask):
            for layer in self.layers:
                x = layer(x, mask)
            return x

    class CrossAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(d, d_head * n_head, bias=False)
            self.to_kv = nn.Linear(d, 2 * d_head * n_head, bias=False)
            self.tail = nn.Sequential(nn.Linear(d_head * n_head, d), nn.Dropout(drop))

        def forward(self, x1, x2, mask):
            return attend(self.to_q(x1), self.to_kv(x2), mask, self.tail)

    class Cross(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm1_x1 = nn.LayerNorm(d)
            self.layer_norm1_x2 = nn.LayerNorm(d)
            self.cross_attn = CrossAttention()
            self.layer_norm2 = nn.LayerNorm(d)
            self.ff = FeedForward()

        def forward(self, x1, x2, mask):
            x = self.cross_attn(self.layer_norm1_x1(x1), self.layer_norm1_x2(x2), mask)
            return self.ff(self.layer_norm2(x)) + x

    class TCN(nn.Module):
        def __init__(self, i, channels):
            super().__init__()
            layers = []
            for n, o in enumerate(channels):
                dil = 2 * n + 1
                layers += [nn.Conv1d(i if n == 0 else channels[n - 1], o, 5, padding=2 * dil, dilation=dil),
                           nn.BatchNorm1d(o), nn.GELU(), nn.Dropout(drop)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class BSTCGAP(nn.Module):
        def __init__(self):
            super().__init__()
            self.learned_token_tem = nn.Parameter(torch.zeros(1, d))
            self.embedding_tem = nn.Parameter(torch.zeros(1, 1 + seq_len, d))
            self.embedding_cross = nn.Parameter(torch.zeros(1, seq_len, d))
            self.learned_token_inter = nn.Parameter(torch.zeros(1, d))
            self.embedding_inter = nn.Parameter(torch.zeros(1, 1 + seq_len, d))
            self.mlp_positions = MLP(2, in_dim, 256)
            self.tcn_pose = TCN(in_dim, [d, d])
            self.tcn_shuttle = TCN(2, [d // 2, d])
            self.encoder_tem = Encoder(2)
            self.cross_trans = Cross()
            self.encoder_inter = Encoder(1)
            self.mlp_clean = MLP(d, d, d)
            self.mlp_head = Head(3 * d, n_class, 4 * d)

        def forward(self, pose, shuttle, pos, length):
            # pose (b, t, 2, in_dim), shuttle (b, t, 2), pos (b, t, 2, 2), length (b,)
            b, t, n, _ = pose.shape
            pose = pose.permute(0, 2, 3, 1).reshape(b * n, in_dim, t)
            impact = self.mlp_positions(pos).permute(0, 2, 3, 1).reshape(b * n, in_dim, t)
            pose = self.tcn_pose(pose * impact + pose).view(b, n, d, t).transpose(-2, -1)
            sh = self.tcn_shuttle(shuttle.transpose(1, 2)).unsqueeze(1).transpose(-2, -1)
            x = torch.cat((pose, sh), dim=1).reshape(b * 3, t, d)
            x = torch.cat((self.learned_token_tem.expand(b * 3, 1, d), x), dim=1) + self.embedding_tem
            mask = torch.arange(1 + t, device=x.device).unsqueeze(0) < (1 + length.unsqueeze(-1))
            x = self.encoder_tem(x, mask.repeat_interleave(3, dim=0)).view(b, 3, 1 + t, d)
            p1, p2, sh = x[:, 0], x[:, 1], x[:, 2]
            p1_cls, p2_cls, sh_cls = p1[:, 0], p2[:, 0], sh[:, 0]
            sh = sh[:, 1:] + self.embedding_cross
            token = self.learned_token_inter.expand(b, 1, d)
            both = []
            for p in (p1, p2):
                c = self.cross_trans(p[:, 1:] + self.embedding_cross, sh, mask[:, 1:])
                c = torch.cat((token, c), dim=1) + self.embedding_inter
                both.append(self.encoder_inter(c, mask)[:, 0])
            alpha = ((torch.cosine_similarity(both[0], sh_cls) - torch.cosine_similarity(both[1], sh_cls) + 2) / 4)
            alpha = alpha.unsqueeze(1)
            sh_cls = sh_cls - self.mlp_clean(torch.minimum(both[0], both[1]))
            x = torch.cat((alpha * (p1_cls + both[0]), (1 - alpha) * (p2_cls + both[1]), sh_cls), dim=1)
            return self.mlp_head(x)

    return BSTCGAP()


def load_model(weights: str | Path, seq_len: int = SEQ_LEN, n_class: int = N_CLASSES):
    import torch

    model = build_model(seq_len, n_class)
    model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def classify(models: list, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Class probabilities (n, n_class), averaged over the models (the released serials of one setup)."""
    import torch

    if len(inputs["length"]) == 0:
        return np.zeros((0, N_CLASSES))
    args = [torch.from_numpy(inputs[k]) for k in ("pose", "shuttle", "pos", "length")]
    with torch.no_grad():
        probs = [m(*args).softmax(-1) for m in models]
    return torch.stack(probs).mean(0).numpy()
