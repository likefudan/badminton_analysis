"""Stroke-type labelling on the phone: clips of every hit, and the labels written back as CSV."""

import csv
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from badminton import stroke_pack as sp

W, H, N, FPS = 640, 360, 120, 30


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "game.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(N):
        img = np.full((H, W, 3), 90, np.uint8)
        cv2.putText(img, str(f), (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        wr.write(img)
    wr.release()
    court = tmp_path / "game.court.json"
    court.write_text("{}")
    return path


def fake_run(calls):
    def run(shuttle, data, calib, weights, hits, per_side):
        calls.append(per_side)
        results = [
            {"frame": 10, "time_s": 0.33, "player_id": 1, "side": "near", "type": "发短球", "group": "发球", "prob": 0.9,
             "alternatives": [], "attributes": {}, "needs_review": False, "ok_share": 1.0},
            {"frame": 60, "time_s": 2.0, "player_id": 3, "side": "far", "type": "吊球", "group": "吊球", "prob": 0.4,
             "alternatives": [{"type": "过渡吊球", "prob": 0.3}], "attributes": {"direction": {"value": "对角"}},
             "needs_review": True, "ok_share": 0.5},
            {"frame": 115, "time_s": 3.83, "player_id": 1, "side": "near", "type": "杀球", "group": "杀球", "prob": 0.8,
             "alternatives": [], "attributes": {}, "needs_review": False, "ok_share": 1.0},
        ]
        frames = np.arange(0, N, 2)
        bbox = np.zeros((len(frames), 1, 4))
        bbox[:, 0] = (100, 100, 200, 300)
        track = SimpleNamespace(id=1, side=0, det_index=np.zeros(len(frames), int))
        return results, {"fps": FPS, "data": {"frame": frames, "bbox": bbox}, "tracking": SimpleNamespace(tracks=[track])}
    return run


@pytest.fixture
def analysis(clip, tmp_path, monkeypatch):
    d = tmp_path / "analysis"
    d.mkdir()
    (d / "game.skeleton.npz").write_bytes(b"x")
    (d / "game.shuttle.csv").write_text("Frame,Visibility,X,Y\n")
    (d / "game.players.json").write_text(json.dumps({"1": "JayX", "3": "CDL"}))
    weights = tmp_path / "bst.pt"
    weights.write_bytes(b"w")
    from badminton import court as C
    monkeypatch.setattr(C.CourtCalibration, "load", staticmethod(lambda p: object()))
    return d, weights


def test_pack_cuts_one_clip_per_hit_with_a_key_frame_at_each_start(clip, analysis, tmp_path):
    d, weights = analysis
    calls = []
    out = tmp_path / "out"
    seen = []

    def progress(n, total, elapsed):  # the CLI passes pipeline.print_progress: (n, total, elapsed)
        from badminton.pipeline import print_progress

        print_progress(n, total, elapsed)
        seen.append((n, total))

    state = sp.pack(clip, out, clip.with_suffix(".court.json"), d, singles=False, weights=[weights], run=fake_run(calls),
                    progress=progress)
    assert seen and seen[-1] == (129, 129)
    assert calls == [2]
    assert (d / "game.strokes.json").exists()  # kept for the next time
    c = state["clip"]
    assert c["frames"] == round(0.8 * FPS) + round(0.6 * FPS) + 1 == 43 and c["hit"] == 24
    assert [s["frame"] for s in state["strokes"]] == [10, 60, 115]
    assert state["strokes"][1]["needs_review"] and state["strokes"][1]["direction"] == "对角"
    assert {p["id"] for p in state["players"]} == {1, 3}
    assert state["types"][-2:] == ["未知", "不是击球"] and "劈杀" in state["tags"]
    cap = cv2.VideoCapture(str(out / "strokes.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3 * 43
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    hit = frames[43 + 24]  # the hit of the second clip: red edge
    assert hit[2, W // 2, 2] > 150 and hit[2, W // 2, 0] < 120
    assert frames[43 + 5][2, W // 2, 2] < 150  # not on the other frames
    first = frames[24]  # the hitter of the first stroke (player 1) has a yellow box
    assert first[96, 150, 1] > 170 and first[96, 150, 0] < 100
    # the second stroke's hitter is player 3 (no track here): player 1 still has a box, white, with its number
    other = frames[43 + 5][96, 150]
    assert other.min() > 180

    # strokes.json is reused when newer than the shuttle track
    calls.clear()
    sp.pack(clip, out, clip.with_suffix(".court.json"), d, singles=False, weights=[weights], run=fake_run(calls))
    assert calls == []


def test_missing_weights_say_where_to_get_them(clip, analysis, tmp_path):
    d, _ = analysis
    with pytest.raises(SystemExit, match="BST"):
        sp.pack(clip, tmp_path / "out", clip.with_suffix(".court.json"), d, weights=[tmp_path / "none.pt"], run=fake_run([]))


def test_labels_csv_has_the_agreed_columns(tmp_path):
    out = tmp_path / "game.strokes.labels.csv"
    n = sp.write_labels_csv(out, {"60": {"player_id": 3, "type": "吊球", "tags": ["劈吊", "对角"], "note": "很快"},
                                  "10": {"player_id": None, "type": "发短球", "tags": [], "note": ""}})
    assert n == 2
    rows = list(csv.reader(out.open()))
    assert rows == [["frame", "player_id", "type", "tags", "note"], ["10", "", "发短球", "", ""],
                    ["60", "3", "吊球", "劈吊;对角", "很快"]]


def test_node_prepares_strokes_and_writes_the_labels_csv(clip, analysis, tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from badminton import node

    vid = "11111111-2222-3333-4444-555555555555"
    video = {"id": vid, "filename": "game.mp4", "size": 1, "options": {"mode": "doubles"}}
    job = tmp_path / "work" / vid
    job.mkdir(parents=True)
    (job / "game.mp4").write_bytes(clip.read_bytes())
    (job / "game.court.json").write_text("{}")
    d, weights = analysis
    (job / "analysis-doubles").mkdir()
    seen, posts, files = {}, [], {}
    tasks = [("strokes", video), ("stroke-labels", video)]

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.endswith("/stroke-labels"):
                return self.send({"labels": {"60": {"player_id": 3, "type": "吊球", "tags": ["劈吊"], "note": ""}},
                                  "changedAt": "T1"})
            self.send({})

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            if self.path.endswith("/node/claim"):
                task, v = tasks.pop(0) if tasks else (None, None)
                return self.send({"video": v, "task": task})
            posts.append((self.path.split(vid)[-1], data))
            self.send({})

        def do_PUT(self):
            files[self.path.split("/files/")[1]] = self.rfile.read(int(self.headers.get("content-length") or 0))
            self.send({})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def strokes_fn(work, out_dir, court, analysis_dir, singles, on_frames, log):
        seen.update(work=work, analysis_dir=analysis_dir, singles=singles)
        sp.pack(work, out_dir, court, d, singles=singles, weights=[weights], run=fake_run([]))
        return 0

    try:
        cfg = node.Config(url=f"http://127.0.0.1:{srv.server_port}", token="t", work_dir=tmp_path / "work", name="m")
        node.run(cfg, once=True, strokes_fn=strokes_fn)
    finally:
        srv.shutdown()
    assert seen == {"work": job / "game.mp4", "analysis_dir": job / "analysis-doubles", "singles": False}
    assert list(files) == ["strokes.mp4", "strokes.state.json"]
    jobs = [p for path, p in posts if path == "/strokes/job"]
    assert jobs[-1]["status"] == "ready" and "3 拍，其中 1 拍" in jobs[-1]["stage"]
    rows = list(csv.reader((job / "game.strokes.labels.csv").open()))
    assert rows == [["frame", "player_id", "type", "tags", "note"], ["60", "3", "吊球", "劈吊", ""]]
    assert ("/stroke-labels/synced", {"changedAt": "T1"}) in posts
