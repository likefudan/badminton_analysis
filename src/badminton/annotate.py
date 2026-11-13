"""Shuttle annotation: labels, model uncertainty, and a local web page to label with.

`badminton annotate VIDEO` shows one frame at a time in the browser with
TrackNetV3's prediction drawn on it; the user confirms it or clicks the real
shuttle. The labels are meant for fine-tuning TrackNetV3, which trains on
windows of consecutive frames, so whole rallies are labelled - but the rallies
(and, inside a rally, the frames to look at first) are ordered by how unsure
the model is.

Files, next to the video (`match.mp4`):
  match.labels.csv   Frame,Visibility,X,Y,Status - TrackNetV3's label columns
                     (video frame number, 1/0, video pixels; 0,0 when there is
                     no shuttle) plus Status:
                       human    confirmed or clicked by a person, one frame at a time
                       bulk     accepted by a person in one go ("accept the confident rest")
                       machine  the model's prediction, nobody has looked at it
                       skip     occluded / cannot tell: never used for training
  match.labels.json  who/when/tool version, the rallies, counts per status
  match.annotate.npz cached predictions (top-1, chosen path, all candidates)

The server uses only the standard library and binds to 127.0.0.1.
"""

from __future__ import annotations

import csv
import getpass
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import __version__

TOOL = "badminton annotate"
STATUSES = ("human", "bulk", "machine", "skip")
REVIEWED = ("human", "bulk")  # statuses a person stands behind
CSV_HEADER = ["Frame", "Visibility", "X", "Y", "Status"]

# uncertainty
AGREE_PX = 15.0  # top-1 and chosen path further apart than this disagree
NEAR_PX = 30.0  # a candidate this close to the chosen point is "its" blob
JUMP_FREE = 60.0  # px/frame at 1080p the shuttle may move without suspicion (scaled with the frame)
JUMP_SCALE = 200.0
W_DISAGREE, W_LOW, W_MANY, W_JUMP = 0.6, 0.5, 0.3, 0.6
RAW_ON_STATIC = 0.3  # weight of a disagreement when top-1 sits on a static distractor (the path is usually right)
CONFIDENT = 0.3  # frames below this may be accepted in bulk
DOUBTFUL = 0.5  # above this the prediction is probably wrong (IMG_8923: 44 % of rally frames; 27-50 % are wrong)
# rough labelling speed until the session has its own numbers (seconds per frame)
SECONDS_CONFIRM, SECONDS_CLICK = 0.4, 2.5


# --- labels ------------------------------------------------------------------------


@dataclass
class Label:
    visibility: int
    x: float
    y: float
    status: str


class LabelStore:
    """Labels of one video, kept in memory and written to CSV + JSON on every change."""

    def __init__(self, csv_path: str | Path, video_info: dict | None = None, annotator: str | None = None,
                 rallies: list | None = None):
        self.csv_path = Path(csv_path)
        self.meta_path = self.csv_path.with_suffix(".json")
        self.labels: dict[int, Label] = {}
        self.meta: dict = {}
        self.lock = threading.Lock()
        if self.csv_path.exists():
            self.labels = read_labels(self.csv_path)
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text())
        now = _now()
        self.meta.setdefault("created", now)
        self.meta.setdefault("sessions", [])
        if video_info:
            self.meta["video"] = video_info
        if rallies is not None:
            self.meta["rallies"] = [[int(a), int(b)] for a, b in rallies]
        self.annotator = annotator or getpass.getuser()
        self.session = {"annotator": self.annotator, "start": now, "end": now, "frames_changed": 0}

    def set(self, frame: int, visibility: int, x: float = 0.0, y: float = 0.0, status: str = "human") -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        visibility = int(bool(visibility)) if status != "skip" else 0
        if not visibility:
            x = y = 0.0
        self.labels[int(frame)] = Label(visibility, float(x), float(y), status)
        if status != "machine":  # a session counts what a person did
            self.session["frames_changed"] += 1

    def remove(self, frame: int) -> None:
        if self.labels.pop(int(frame), None) is not None:
            self.session["frames_changed"] += 1

    def counts(self) -> dict:
        c = {s: 0 for s in STATUSES}
        for lab in self.labels.values():
            c[lab.status] += 1
        return c

    def save(self) -> None:
        with self.lock:
            write_labels(self.labels, self.csv_path)
            self.session["end"] = _now()
            sessions = [s for s in self.meta["sessions"] if s is not self.session]
            if self.session["frames_changed"]:
                sessions.append(self.session)
            self.meta.update({
                "tool": TOOL, "tool_version": __version__, "updated": self.session["end"],
                "label_file": self.csv_path.name, "columns": CSV_HEADER,
                "status_meaning": {"human": "confirmed or clicked by a person",
                                   "bulk": "accepted by a person in bulk (low model uncertainty)",
                                   "machine": "model prediction, NOT reviewed",
                                   "skip": "occluded or unsure: excluded from training"},
                "counts": self.counts(), "sessions": sessions,
            })
            self.meta["sessions"] = sessions
            _atomic_write(self.meta_path, json.dumps(self.meta, ensure_ascii=False, indent=1))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


def write_labels(labels: dict[int, Label], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(CSV_HEADER)
        for frame in sorted(labels):
            lab = labels[frame]
            wr.writerow([frame, lab.visibility, _num(lab.x), _num(lab.y), lab.status])
    os.replace(tmp, path)


def read_labels(path: str | Path) -> dict[int, Label]:
    """Also reads a plain TrackNetV3 CSV (no Status column): those rows count as "machine"."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            status = row.get("Status") or "machine"
            if status not in STATUSES:
                raise ValueError(f"{path}: unknown Status {status!r} at frame {row['Frame']}")
            out[int(row["Frame"])] = Label(int(float(row["Visibility"])), float(row["X"]), float(row["Y"]), status)
    return out


# --- predictions -------------------------------------------------------------------


def rows_to_array(rows, n_frames: int | None = None) -> np.ndarray:
    """(frame, visibility, x, y) rows -> (F, 3) array of visibility, x, y indexed by frame."""
    rows = list(rows)
    n = n_frames or (max(r[0] for r in rows) + 1 if rows else 0)
    out = np.zeros((n, 3))
    for f, v, x, y in rows:
        if 0 <= f < n:
            out[int(f)] = (v, x, y)
    return out


def read_prediction_csv(path: str | Path, n_frames: int | None = None) -> np.ndarray:
    with open(path, newline="") as f:
        rows = [(int(r["Frame"]), int(float(r["Visibility"])), float(r["X"]), float(r["Y"])) for r in csv.DictReader(f)]
    return rows_to_array(rows, n_frames)


def save_predictions(path: str | Path, path_pred: np.ndarray, raw: np.ndarray | None, candidates: list | None) -> None:
    """candidates: per frame an array (k, >=3) of x, y, peak in VIDEO pixels."""
    data = {"path": path_pred}
    if raw is not None:
        data["raw"] = raw
    if candidates is not None:
        flat = [np.column_stack([np.full(len(c), f), np.asarray(c)[:, :3]]) for f, c in enumerate(candidates) if len(c)]
        data["cand"] = np.concatenate(flat) if flat else np.zeros((0, 4))
        data["cand_frames"] = np.array(len(candidates))
    np.savez_compressed(path, **data)


def load_predictions(path: str | Path):
    d = np.load(path)
    raw = d["raw"] if "raw" in d else None
    cands = None
    if "cand" in d:
        cands = [np.zeros((0, 3))] * int(d["cand_frames"])
        flat = d["cand"]
        if len(flat):
            fr = flat[:, 0].astype(int)
            for f in np.unique(fr):
                cands[f] = flat[fr == f, 1:]
    return d["path"], raw, cands


def predict(video: str | Path, court: str | Path | None = None, tracknet_dir=None, progress=None):
    """Run TrackNetV3 on the video: (path, raw, candidates) as save_predictions takes them.
    With a court calibration the path uses the airspace mask as `badminton shuttle --court` does."""
    from .shuttle import HEIGHT, WIDTH, track_shuttle
    from .shuttle_path import rows_from_candidates
    from .video import probe

    info = probe(video)
    debug: dict = {}
    rows, tm = track_shuttle(video, tracknet_dir, progress=progress, debug=debug)
    calib = None
    if court:
        from .court import CourtCalibration

        calib = CourtCalibration.load(court)
    # the same path selection as `badminton shuttle` (gravity check on flights included)
    path_rows = rows_from_candidates(debug["candidates"], (info.width, info.height), calib,
                                     measured=None if getattr(tm, "measured", None) is None else list(tm.measured),
                                     fps=info.fps, n_frames=len(rows))
    n = len(rows)
    sx, sy = info.width / WIDTH, info.height / HEIGHT
    cands = [np.column_stack([c[:, 0] * sx, c[:, 1] * sy, c[:, 2]]) if len(c) else np.zeros((0, 3))
             for c in debug["candidates"][:n]]
    return rows_to_array(path_rows, n), rows_to_array(rows, n), cands


# --- uncertainty and sampling ------------------------------------------------------


def static_grid(candidates: list, size: tuple[int, int]) -> np.ndarray:
    """shuttle_path.static_map for candidates given in video pixels: cells where a blob
    sits in more than 5 % of the frames (wall lights, shuttles on the floor)."""
    from .shuttle import HEIGHT, WIDTH
    from .shuttle_path import static_map

    sx, sy = WIDTH / size[0], HEIGHT / size[1]
    return static_map([np.column_stack([np.asarray(c)[:, 0] * sx, np.asarray(c)[:, 1] * sy]) if len(c) else np.zeros((0, 2))
                       for c in candidates])


def on_static(grid: np.ndarray, xy: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Whether points (n, 2) in video pixels fall on a static cell."""
    from .shuttle import HEIGHT, WIDTH
    from .shuttle_path import STATIC_CELL

    xy = np.asarray(xy, float).reshape(-1, 2)
    gy = np.clip((xy[:, 1] * HEIGHT / size[1] // STATIC_CELL).astype(int), 0, grid.shape[0] - 1)
    gx = np.clip((xy[:, 0] * WIDTH / size[0] // STATIC_CELL).astype(int), 0, grid.shape[1] - 1)
    return grid[gy, gx]


def moving_candidates(candidates: list, size: tuple[int, int]) -> tuple[list, np.ndarray]:
    """Candidates without the static distractors, and the static grid. A typical phone
    video has ~7 blobs per frame, most of them the same lights in every frame; they
    say nothing about how sure the model is and only clutter the picture."""
    grid = static_grid(candidates, size)
    return [np.asarray(c)[~on_static(grid, np.asarray(c)[:, :2], size)] if len(c) else np.zeros((0, 3))
            for c in candidates], grid


def frame_uncertainty(path: np.ndarray, raw: np.ndarray | None = None, candidates: list | None = None,
                      frame_height: int = 1080, raw_weight: np.ndarray | None = None) -> np.ndarray:
    """0 (the model is sure) .. 1 per frame, from four signs that the prediction may be wrong:
      * top-1 and the chosen path disagree (one sees a shuttle, the other not, or far apart);
      * low heatmap score of the chosen blob, or a point with no blob behind it
        (filled gap), or blobs present but none chosen;
      * several candidate blobs in the frame;
      * a big jump to the neighbouring frames, or the shuttle appearing/vanishing.
    Combined as 1 - prod(1 - weight * sign), so one strong sign is enough.
    `candidates` should be the moving ones (moving_candidates): (k, >=3) x, y, peak per frame."""
    F = len(path)
    v = path[:, 0] > 0
    xy = path[:, 1:3]
    scale = frame_height / 1080

    disagree = np.zeros(F)
    if raw is not None:
        rv = raw[:F, 0] > 0
        d = np.hypot(*(xy - raw[:F, 1:3]).T)
        disagree = ((v != rv) | (v & rv & (d > AGREE_PX * scale))).astype(float)
        if raw_weight is not None:  # e.g. less when top-1 sits on a known static distractor
            disagree *= raw_weight[:F]

    low = np.zeros(F)
    many = np.zeros(F)
    if candidates is not None:
        for f in range(min(F, len(candidates))):
            c = np.asarray(candidates[f])
            n = len(c)
            many[f] = min(1.0, max(0, n - 1) / 3)
            if v[f]:
                near = c[np.hypot(c[:, 0] - xy[f, 0], c[:, 1] - xy[f, 1]) <= NEAR_PX * scale] if n else c
                low[f] = float(np.clip((1 - near[:, 2].max()) / 0.7, 0, 1)) if len(near) else 0.6
            elif n:
                low[f] = 0.7

    jump = np.zeros(F)
    if F > 1:
        step = np.hypot(*(xy[1:] - xy[:-1]).T)
        both = v[1:] & v[:-1]
        s = np.where(both, np.clip((step - JUMP_FREE * scale) / (JUMP_SCALE * scale), 0, 1), 0.0)
        s = np.maximum(s, 0.5 * (v[1:] != v[:-1]))
        jump[1:] = s
        jump[:-1] = np.maximum(jump[:-1], s)

    return 1 - (1 - W_DISAGREE * disagree) * (1 - W_LOW * low) * (1 - W_MANY * many) * (1 - W_JUMP * jump)


def rallies_from_predictions(path: np.ndarray, fps: float = 30.0, min_s: float = 2.0, gap_s: float = 1.5) -> list:
    """Fallback when no rally list is given: stretches where the shuttle is seen,
    joined across gaps shorter than `gap_s`, at least `min_s` long. [(start, end)] inclusive frames."""
    seen = np.flatnonzero(path[:, 0] > 0)
    if not len(seen):
        return []
    out = []
    a = b = int(seen[0])
    for f in seen[1:]:
        if f - b > gap_s * fps:
            out.append((a, b))
            a = int(f)
        b = int(f)
    out.append((a, b))
    return [(a, b) for a, b in out if b - a + 1 >= min_s * fps]


def load_rallies(movement_json: str | Path, fps: float, n_frames: int) -> list:
    """Rallies of `badminton movement` (start_s/end_s) as inclusive frame ranges."""
    data = json.loads(Path(movement_json).read_text())
    out = []
    for r in data.get("rallies", []):
        a, b = int(round(r["start_s"] * fps)), min(n_frames - 1, int(round(r["end_s"] * fps)))
        if b > a:
            out.append((max(0, a), b))
    return out


def rank_rallies(rallies: list, uncertainty: np.ndarray) -> list[int]:
    """Rally indices, the one the model is least sure about first (mean frame uncertainty)."""
    score = [float(uncertainty[a:b + 1].mean()) if b >= a else 0.0 for a, b in rallies]
    return sorted(range(len(rallies)), key=lambda i: (-score[i], i))


def rank_frames(rally: tuple[int, int], uncertainty: np.ndarray) -> list[int]:
    """Frames of one rally, most uncertain first."""
    a, b = rally
    return sorted(range(a, b + 1), key=lambda f: (-uncertainty[f], f))


def prefill(store: LabelStore, rally: tuple[int, int], path: np.ndarray) -> int:
    """Write the model's prediction as an unreviewed ("machine") label for every frame
    of the rally that has no label yet. Returns how many were written."""
    n = 0
    for f in range(rally[0], rally[1] + 1):
        if f not in store.labels and f < len(path):
            store.set(f, int(path[f, 0] > 0), path[f, 1], path[f, 2], "machine")
            n += 1
    if n:
        store.meta.setdefault("prefill", []).append({
            "when": _now(), "frames": [int(rally[0]), int(rally[1])], "written": n,
            "source": "TrackNetV3 prediction after path selection (badminton shuttle), not reviewed"})
    return n


def estimate_minutes(n_frames: int, uncertainty: np.ndarray | None = None, seconds_per_frame: float | None = None) -> float:
    """Minutes to label `n_frames` more frames: the session's measured pace if known,
    otherwise confirm/click times weighted by how many predictions look doubtful."""
    if seconds_per_frame is None:
        wrong = float(np.mean(uncertainty > DOUBTFUL)) if uncertainty is not None and len(uncertainty) else 0.4
        seconds_per_frame = (1 - wrong) * SECONDS_CONFIRM + wrong * SECONDS_CLICK
    return n_frames * seconds_per_frame / 60


# --- web server --------------------------------------------------------------------


class FrameSource:
    """JPEG frames of the video by number: one decoder, reads on when the next frame
    is asked for (the usual case), seeks otherwise; the last frames stay cached."""

    def __init__(self, video: str | Path, cache: int = 96, quality: int = 92):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(video))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = 0  # frame the next read() returns
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.cache_size = cache
        self.quality = quality
        self.lock = threading.Lock()

    def jpeg(self, frame: int) -> bytes | None:
        with self.lock:
            if frame in self.cache:
                self.cache.move_to_end(frame)
                return self.cache[frame]
            if not 0 <= frame < self.n:
                return None
            if not self.pos <= frame <= self.pos + 8:  # a few grabs are cheaper than a seek
                self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame)
                self.pos = frame
            while self.pos < frame:
                if not self.cap.grab():
                    return None
                self.pos += 1
            ok, img = self.cap.read()
            self.pos += 1
            if not ok:
                return None
            ok, buf = self.cv2.imencode(".jpg", img, [self.cv2.IMWRITE_JPEG_QUALITY, self.quality])
            data = buf.tobytes()
            self.cache[frame] = data
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return data


class AnnotationApp:
    """Everything the page needs; the HTTP handler only translates requests."""

    def __init__(self, video: str | Path, store: LabelStore, path: np.ndarray, raw: np.ndarray | None,
                 candidates: list | None, rallies: list, fps: float, size: tuple[int, int]):
        self.video = Path(video)
        self.store = store
        self.path, self.raw = path, raw
        raw_weight = None
        if candidates is not None:
            candidates, grid = moving_candidates(candidates, size)
            if raw is not None:
                raw_weight = np.where(on_static(grid, raw[:, 1:3], size) & (raw[:, 0] > 0), RAW_ON_STATIC, 1.0)
        self.candidates = candidates
        self.rallies = [(int(a), int(b)) for a, b in rallies]
        self.fps, self.size = fps, size
        self.uncertainty = frame_uncertainty(path, raw, candidates, size[1], raw_weight)
        self.order = rank_rallies(self.rallies, self.uncertainty)
        self.frames = FrameSource(video)
        self.times: list[float] = []  # seconds between consecutive label actions in this session
        self.last_action: float | None = None
        # the server answers requests on several threads: labels must not change while
        # /api/state walks them or while another request is being applied
        self.lock = threading.RLock()

    def state(self) -> dict:
        with self.lock:
            return self._state()

    def _state(self) -> dict:
        r3 = lambda a: [[int(v), round(float(x), 1), round(float(y), 1)] for v, x, y in a]  # noqa: E731
        cands = None
        if self.candidates is not None:
            cands = [[[round(float(x), 1), round(float(y), 1), round(float(p), 2)] for x, y, p in np.asarray(c)[:, :3]]
                     for c in self.candidates]
        return {
            "video": self.video.name, "frames": len(self.path), "fps": self.fps, "size": list(self.size),
            "tool_version": __version__, "annotator": self.store.annotator, "label_file": str(self.store.csv_path),
            "rallies": [{"index": i, "start": a, "end": b, "uncertainty": round(float(self.uncertainty[a:b + 1].mean()), 3)}
                        for i, (a, b) in enumerate(self.rallies)],
            "order": self.order, "confident": CONFIDENT, "doubtful": DOUBTFUL,
            "pred": r3(self.path), "raw": r3(self.raw) if self.raw is not None else None, "candidates": cands,
            "uncertainty": [round(float(u), 3) for u in self.uncertainty],
            "labels": {str(f): [lab.visibility, lab.x, lab.y, lab.status] for f, lab in self.store.labels.items()},
            "progress": self.progress(),
        }

    def progress(self) -> dict:
        labels = self.store.labels
        in_rally = np.zeros(len(self.path), bool)
        for a, b in self.rallies:
            in_rally[a:b + 1] = True
        done = np.zeros(len(self.path), bool)
        for f, lab in labels.items():
            if lab.status != "machine" and f < len(done):
                done[f] = True
        todo = in_rally & ~done
        pace = float(np.median(self.times)) if len(self.times) >= 20 else None
        return {"counts": self.store.counts(), "rally_frames": int(in_rally.sum()), "done": int((in_rally & done).sum()),
                "left": int(todo.sum()), "seconds_per_frame": pace,
                "minutes_left": round(estimate_minutes(int(todo.sum()), self.uncertainty[todo], pace), 1)}

    def apply(self, items: list) -> dict:
        """items: [{"frame": n, "visibility": 0/1, "x":, "y":, "status":}] or {"frame": n, "delete": true}.
        All items are checked before any is applied: a bad request changes nothing."""
        w, h = self.size
        ops = []
        for it in items:
            f = int(it["frame"])
            if not 0 <= f < len(self.path):
                raise ValueError(f"frame {f} out of range")
            if it.get("delete"):
                ops.append((f, None))
                continue
            status = it.get("status", "human")
            if status not in ("human", "bulk", "skip"):  # the page never writes machine labels
                raise ValueError(f"status {status!r} not allowed")
            x, y = float(it.get("x", 0)), float(it.get("y", 0))
            if not (np.isfinite(x) and np.isfinite(y) and 0 <= x <= w and 0 <= y <= h):
                raise ValueError(f"point ({x}, {y}) outside the frame")
            ops.append((f, (int(it.get("visibility", 0)), x, y, status)))
        with self.lock:
            return self._apply(ops, len(items))

    def _apply(self, ops: list, n_items: int) -> dict:
        for f, lab in ops:
            if lab is None:
                self.store.remove(f)
            else:
                self.store.set(f, *lab)
        now = time.monotonic()
        if n_items == 1 and self.last_action is not None and now - self.last_action < 30:
            self.times.append(now - self.last_action)  # pauses and bulk actions do not count
        self.last_action = now
        self.store.save()
        return {"ok": True, "progress": self.progress()}


def _handler(app: AnnotationApp, port: int):
    page = (Path(__file__).with_name("annotate.html")).read_bytes()
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json", cache: str = "no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _local(self) -> bool:
            # only this machine's browser, addressed as localhost: refuses a page of
            # another site that points a host name at 127.0.0.1 (DNS rebinding) and cross-site POSTs
            if self.headers.get("Host") not in hosts:
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin in {f"http://{h}" for h in hosts}

        def do_GET(self):
            if not self._local():
                return self._send(403, b'{"error": "forbidden"}')
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/api/state":
                return self._send(200, json.dumps(app.state()).encode())
            if path.startswith("/frame/") and path.endswith(".jpg"):
                try:
                    data = app.frames.jpeg(int(path[7:-4]))
                except ValueError:
                    data = None
                if data is None:
                    return self._send(404, b'{"error": "no such frame"}')
                return self._send(200, data, "image/jpeg", "max-age=3600")
            self._send(404, b'{"error": "not found"}')

        def do_POST(self):
            if not self._local():
                return self._send(403, b'{"error": "forbidden"}')
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/label":
                    return self._send(200, json.dumps(app.apply(body["items"])).encode())
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, json.dumps({"error": str(e)}).encode())
            self._send(404, b'{"error": "not found"}')

    return Handler


def make_server(app: AnnotationApp, port: int = 8765) -> ThreadingHTTPServer:
    """Bound to 127.0.0.1 only. port 0 picks a free one (tests)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), BaseHTTPRequestHandler)
    server.RequestHandlerClass = _handler(app, server.server_address[1])
    server.daemon_threads = True
    return server
