"""Labelling on the phone: packing frames for the page, and writing the page's labels back next to the video."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from badminton import annotate as an
from badminton import annotate_pack as ap
from badminton import node

W, H, N, FPS = 640, 360, 90, 30


def xy(f):
    return 30 + 6 * f, 40 + 2 * f


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(N):
        img = np.full((H, W, 3), 60, np.uint8)
        cv2.putText(img, str(f), (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.circle(img, xy(f), 4, (255, 255, 255), -1)
        wr.write(img)
    wr.release()
    # rallies 0.5-1.0 s, 1.5-2.5 s and 2.6-2.9 s
    (tmp_path / "clip.movement.json").write_text(json.dumps({"rallies": [
        {"start_s": 0.5, "end_s": 1.0}, {"start_s": 1.5, "end_s": 2.5}, {"start_s": 2.6, "end_s": 2.9}]}))
    return path


def fake_predict(calls):
    def predict(video, court, tracknet_dir, progress):
        calls.append(video)
        path = np.zeros((N, 3))
        for f in range(N):
            path[f] = (1, *xy(f))
        path[50:60, 1] += 80  # frames 50-59 (second rally) are wrong: the model is unsure there
        raw = path.copy()
        raw[50:60] = 0
        cands = [np.array([[*xy(f), 0.9], [600.0, 20.0, 0.5]]) for f in range(N)]
        return path, raw, cands
    return predict


def test_pack_picks_the_unsure_rallies_and_cuts_their_frames(clip, tmp_path):
    calls = []
    out = tmp_path / "annot"
    state = ap.pack(clip, out, movement=clip.with_suffix(".movement.json"), count=2, predict=fake_predict(calls), width=320)
    assert len(calls) == 1
    assert state["video"]["size"] == [W, H] and state["image"] == [320, 180]
    starts = [r["start"] for r in state["rallies"]]
    assert starts[0] == 45  # the rally with the wrong frames comes first
    assert len(state["rallies"]) == 2
    r = state["rallies"][0]
    assert len(r["offsets"]) == r["end"] - r["start"] + 1 == len(r["pred"]) == len(r["unc"]) == len(r["cand"])
    data = (out / r["pack"]).read_bytes()
    o, n = r["offsets"][10]
    img = cv2.imdecode(np.frombuffer(data[o:o + n], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (180, 320)
    # the frame really is frame start+10: the dot sits where it should (scaled by half)
    f = r["start"] + 10
    x, y = xy(f)
    assert img[y // 2, x // 2].mean() > 150
    assert r["pred"][10] == [1, round(x + 80.0, 1), float(y)]  # predictions stay in video pixels
    assert max(r["unc"][5:15]) > state["confident"]
    # the static blob at (600, 20) is in every frame: not offered as a candidate
    assert all(len(c) == 1 for c in r["cand"])

    # packed again: same rallies, cached predictions
    state2 = ap.pack(clip, out, movement=clip.with_suffix(".movement.json"), count=2, predict=fake_predict(calls), width=320)
    assert [x["start"] for x in state2["rallies"]] == starts and len(calls) == 1
    ap.pack(clip, out, movement=clip.with_suffix(".movement.json"), fresh=True, predict=fake_predict(calls), width=320)
    assert len(calls) == 2


def test_choose_rallies_prefers_short_ones_within_budget():
    unc = np.zeros(1000)
    unc[0:690] = 0.9  # the long rally (23 s) is the least sure
    unc[700:760] = 0.5
    rallies = [(0, 689), (700, 759), (800, 829)]
    assert ap.choose_rallies(rallies, unc, fps=30, count=2) == [1, 2]
    assert ap.choose_rallies(rallies, unc, fps=30, count=3) == [1, 2, 0]
    assert ap.choose_rallies(rallies, unc, fps=30, count=3, max_frames=100) == [1, 2]


def test_labels_from_the_site_replace_the_web_rallies_and_keep_the_rest(clip, tmp_path):
    state = ap.pack(clip, tmp_path / "annot", movement=clip.with_suffix(".movement.json"), count=1,
                    predict=fake_predict([]), width=320)
    a = state["rallies"][0]["start"]
    store = an.LabelStore(clip.with_suffix(".labels.csv"), {"file": clip.name}, "desk")
    store.set(5, 1, 10, 10, "human")       # outside the web rally: kept
    store.set(a + 1, 1, 10, 10, "human")   # inside, not on the site any more: dropped
    store.save()
    got = ap.write_labels_from_cloud(clip, state, {str(a): [1, 111.5, 22, "human"], str(a + 2): [0, 0, 0, "skip"],
                                                   str(a + 3): [1, 5, 6, "bulk"]})
    labels = an.read_labels(clip.with_suffix(".labels.csv"))
    assert labels[5].status == "human" and a + 1 not in labels
    assert labels[a] == an.Label(1, 111.5, 22.0, "human")
    assert labels[a + 2] == an.Label(0, 0.0, 0.0, "skip") and labels[a + 3].status == "bulk"
    assert got["frames"] == 4 and got["changed"] == 4
    meta = json.loads(clip.with_suffix(".labels.json").read_text())
    assert meta["rallies"] == state["all_rallies"]  # annotate-export splits by these
    assert meta["sessions"][-1]["source"] == "ba.llmat.dev"


class AnnotCloud:
    """The cloud side of the annotate and labels tasks."""

    def __init__(self, video, labels):
        self.video, self.labels = video, labels
        self.tasks = [("annotate", video), ("labels", video)]
        self.annot_jobs, self.files, self.synced, self.claims = [], {}, [], []
        cloud = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def send(self, obj, status=200):
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def body(self):
                return self.rfile.read(int(self.headers.get("content-length") or 0))

            def do_GET(self):
                if self.path == "/api/v1/me":
                    return self.send({"user": "owner"})
                if self.path.endswith("/labels"):
                    return self.send({"labels": cloud.labels, "changedAt": "2026-11-13T10:00:00.000Z"})
                self.send({"error": "?"}, 404)

            def do_POST(self):
                data = json.loads(self.body() or b"{}")
                if self.path == "/api/v1/node/claim":
                    cloud.claims.append(data)
                    if not cloud.tasks:
                        return self.send({"video": None})
                    task, video = cloud.tasks.pop(0)
                    return self.send({"video": video, "task": task})
                if self.path.endswith("/annot/job"):
                    cloud.annot_jobs.append(data)
                    return self.send({})
                if self.path.endswith("/labels/synced"):
                    cloud.synced.append(data)
                    return self.send({})
                self.send({"error": "?"}, 404)

            def do_PUT(self):
                cloud.files[self.path.split("/annot/files/")[1]] = self.body()
                self.send({})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


def test_node_packs_uploads_and_syncs_labels(clip, tmp_path):
    vid = "11111111-2222-3333-4444-555555555555"
    video = {"id": vid, "filename": "clip.mp4", "size": 1, "options": {"mode": "singles"},
             "annot": {"status": "preparing", "options": {"fresh": True}}}
    work = tmp_path / "work"
    job = work / vid
    (job / "analysis").mkdir(parents=True)
    (job / "clip.mp4").write_bytes(clip.read_bytes())
    (job / "analysis" / "clip.movement.json").write_text(clip.with_suffix(".movement.json").read_text())
    seen = {}

    def pack_fn(work_video, out_dir, court, movement, fresh, on_frames, log):
        seen.update(work=work_video, movement=movement, fresh=fresh)
        on_frames(10, 90, 1.0)
        ap.pack(work_video, out_dir, court, movement, fresh, count=2, predict=fake_predict([]), width=320)
        return 0

    cloud = AnnotCloud(video, {})
    try:
        cfg = node.Config(url=cloud.url, token="secret", work_dir=work, name="test-mac")
        assert node.run(cfg, once=True, pack_fn=pack_fn) == 0  # annotate; once stops after the labels task
        assert set(cloud.claims[0]["tasks"]) == {"process", "annotate", "labels"}
        assert seen == {"work": job / "clip.mp4", "movement": job / "analysis" / "clip.movement.json", "fresh": True}
        assert list(cloud.files)[-1] == "state.json" and {"rally-0.bin", "rally-1.bin"} <= set(cloud.files)
        state = json.loads(cloud.files["state.json"])
        assert cloud.annot_jobs[-1]["status"] == "ready" and "2 个回合" in cloud.annot_jobs[-1]["stage"]
        # labels task (the second claim) ran too: nothing labelled yet, an empty CSV and synced
        a = state["rallies"][0]["start"]
        cloud.labels = {str(a): [1, 100, 50, "human"]}
        cloud.tasks = [("labels", video)]
        node.run(cfg, once=True, pack_fn=pack_fn)
        labels = an.read_labels(job / "clip.labels.csv")
        assert labels == {a: an.Label(1, 100.0, 50.0, "human")}
        assert cloud.synced[-1] == {"changedAt": "2026-11-13T10:00:00.000Z"}
    finally:
        cloud.server.shutdown()


def test_node_reports_a_missing_recording(tmp_path):
    video = {"id": "11111111-2222-3333-4444-555555555555", "filename": "gone.mov", "size": 1, "annot": {}}
    cloud = AnnotCloud(video, {})
    cloud.tasks = [("annotate", video)]
    try:
        node.run(node.Config(url=cloud.url, token="secret", work_dir=tmp_path, name="m"), once=True)
        assert cloud.annot_jobs[-1]["status"] == "failed" and "找不到" in cloud.annot_jobs[-1]["error"]
    finally:
        cloud.server.shutdown()


@pytest.mark.parametrize("measured", [None, np.array([True, False] * 45)])
def test_predict_passes_what_tracknet_measured_to_the_path_selection(clip, monkeypatch, measured):
    """The tracker reports `measured` as a numpy array (or not at all): both must work."""
    from types import SimpleNamespace

    from badminton import shuttle, shuttle_path

    def track(video, tracknet_dir=None, progress=None, debug=None, **kw):
        debug["candidates"] = [np.array([[10.0, 10.0, 0.9]]) for _ in range(N)]
        return [(f, 1, 10.0, 10.0) for f in range(N)], SimpleNamespace(measured=measured)

    got = {}

    def rows_from_candidates(cands, size, calib=None, **kw):
        got.update(kw)
        return [(f, 1, 10.0, 10.0) for f in range(N)]

    monkeypatch.setattr(shuttle, "track_shuttle", track)
    monkeypatch.setattr(shuttle_path, "rows_from_candidates", rows_from_candidates)
    path, raw, cands = an.predict(clip)
    assert len(path) == N and len(cands) == N
    assert got["measured"] == (None if measured is None else list(measured))
    assert got["fps"] == FPS
