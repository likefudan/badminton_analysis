import csv
import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from badminton import annotate as an
from badminton.annotate_export import export_dataset, split_rallies, usable_runs

W, H, N = 320, 180, 60


def shuttle_xy(f):
    return 20 + 4 * f, 30 + f


@pytest.fixture
def tiny_video(tmp_path):
    """60 frames, grey background with a fixed pattern, a white dot flying across."""
    path = tmp_path / "clip.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for f in range(N):
        img = np.full((H, W, 3), 60, np.uint8)
        cv2.rectangle(img, (10, 120), (310, 170), (40, 110, 40), -1)
        cv2.circle(img, shuttle_xy(f), 4, (255, 255, 255), -1)
        wr.write(img)
    wr.release()
    return path


def test_label_csv_round_trip(tmp_path):
    store = an.LabelStore(tmp_path / "clip.labels.csv", {"file": "clip.mp4", "frames": N}, "tester", [(0, 29)])
    store.set(3, 1, 101.5, 40, "human")
    store.set(4, 0, 55, 66, "human")  # no shuttle: coordinates are dropped
    store.set(5, 1, 7, 8, "skip")  # occluded: never visible
    store.set(6, 1, 300, 20, "machine")
    store.set(7, 1, 1, 2, "bulk")
    store.remove(7)
    store.save()

    with open(tmp_path / "clip.labels.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Frame", "Visibility", "X", "Y", "Status"]  # TrackNetV3's four columns come first
    assert rows[1:] == [["3", "1", "101.50", "40", "human"], ["4", "0", "0", "0", "human"],
                        ["5", "0", "0", "0", "skip"], ["6", "1", "300", "20", "machine"]]

    again = an.LabelStore(tmp_path / "clip.labels.csv")
    assert again.labels == store.labels
    meta = json.loads((tmp_path / "clip.labels.json").read_text())
    assert meta["tool_version"] and meta["sessions"][-1]["annotator"] == "tester"
    assert meta["counts"] == {"human": 2, "bulk": 0, "machine": 1, "skip": 1}
    assert meta["rallies"] == [[0, 29]]
    with pytest.raises(ValueError):
        store.set(1, 1, 0, 0, "guess")


def test_plain_tracknet_csv_reads_as_unreviewed(tmp_path):
    p = tmp_path / "x_ball.csv"
    p.write_text("Frame,Visibility,X,Y\n0,1,10,20\n1,0,0,0\n")
    labels = an.read_labels(p)
    assert labels[0] == an.Label(1, 10.0, 20.0, "machine") and labels[1].visibility == 0


def _clean_and_messy():
    """Frames 0-49: a smooth track both outputs agree on. Frames 50-99: top-1 jumps
    to a wall light every other frame, two blobs, weaker scores."""
    F = 100
    path = np.zeros((F, 3))
    raw = np.zeros((F, 3))
    cands = []
    for f in range(F):
        x, y = 200 + 10 * f, 300 + 2 * f
        path[f] = raw[f] = (1, x, y)
        if f < 50:
            cands.append(np.array([[x, y, 0.95]]))
        else:
            cands.append(np.array([[x, y, 0.55], [1500, 100, 0.9]]))
            if f % 2:
                raw[f] = (1, 1500, 100)
    return path, raw, cands


def test_uncertainty_ranking():
    path, raw, cands = _clean_and_messy()
    u = an.frame_uncertainty(path, raw, cands)
    assert ((u >= 0) & (u <= 1)).all()
    assert u[:49].max() < an.CONFIDENT < u[50:].min()
    assert u[51] > u[52]  # disagreement on top of the weak score
    rallies = [(0, 49), (50, 99)]
    assert an.rank_rallies(rallies, u) == [1, 0]
    assert an.rank_frames((40, 60), u)[0] > 50 and an.rank_frames((40, 60), u)[-1] < 50

    # each sign alone raises the score
    base = an.frame_uncertainty(path[:50], raw[:50], cands[:50])
    jumpy = path[:50].copy()
    jumpy[20, 1:] += 600
    assert an.frame_uncertainty(jumpy, None, None)[20] > 0.5 > base[20]
    gone = path[:50].copy()
    gone[20] = 0
    assert an.frame_uncertainty(gone, None, cands[:50])[20] > an.CONFIDENT  # a blob is there but was not chosen
    # works without the optional inputs
    assert an.frame_uncertainty(path, None, None).shape == (100,)


def test_prefill_only_touches_empty_frames_and_estimate(tmp_path):
    path, raw, cands = _clean_and_messy()
    store = an.LabelStore(tmp_path / "v.labels.csv")
    store.set(52, 0, status="human")
    assert an.prefill(store, (50, 59), path) == 9
    assert store.labels[52].status == "human" and store.labels[53] == an.Label(1, 730.0, 406.0, "machine")
    u = an.frame_uncertainty(path, raw, cands)
    assert an.estimate_minutes(600, u[:50]) < an.estimate_minutes(600, u[50:])
    assert an.estimate_minutes(600, None, 1.0) == 10


def test_rallies_from_predictions_and_movement_json(tmp_path):
    path = np.zeros((600, 3))
    path[100:250, 0] = 1
    path[260:300, 0] = 1  # short gap: same rally
    path[450:460, 0] = 1  # too short
    assert an.rallies_from_predictions(path, 30) == [(100, 299)]
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"rallies": [{"start_s": 1.0, "end_s": 3.0}, {"start_s": 19.0, "end_s": 30.0}]}))
    assert an.load_rallies(p, 30, 600) == [(30, 90), (570, 599)]


def test_split_and_runs():
    assert split_rallies(1) == ["train"]
    assert split_rallies(2) == ["train", "val"]
    s = split_rallies(10, 0.2)
    assert s.count("val") == 2 and s[0] == "train"
    labels = {f: an.Label(1, 1, 1, "human") for f in range(0, 30)}
    labels[10] = an.Label(0, 0, 0, "skip")
    labels[25] = an.Label(1, 1, 1, "machine")
    assert usable_runs(labels, 0, 40) == [(0, 9), (11, 24)]  # 26-29 is shorter than a window
    assert usable_runs(labels, 0, 40, ("human", "bulk", "machine")) == [(0, 9), (11, 29)]


def test_export_layout(tiny_video, tmp_path):
    store = an.LabelStore(tiny_video.with_suffix(".labels.csv"))
    for f in range(N):
        x, y = shuttle_xy(f)
        if f == 12:
            store.set(f, 0, status="skip")  # cuts rally 0 in two
        elif f == 20:
            store.set(f, 0, status="human")  # no shuttle
        elif f < 50:
            store.set(f, 1, x, y, "human" if f % 2 else "bulk")
        else:
            store.set(f, 1, x, y, "machine")  # unreviewed: not exported
    store.save()
    out = tmp_path / "ds"
    (out / "train").mkdir(parents=True)
    (out / "img_config_288x512_train.npz").write_bytes(b"stale")  # dataset.py's cache from an earlier export
    rallies = [(0, 29), (30, 59)]
    manifest = export_dataset([(tiny_video, store.csv_path, rallies)], out)

    m = manifest["matches"][0]
    assert [(s["id"], s["split"], s["frames"]) for s in m["segments"]] == [
        ("r000_f000000", "train", 12), ("r000_f000013", "train", 17), ("r001_f000030", "val", 20)]
    assert not (out / "img_config_288x512_train.npz").exists()
    for split, rid, n, first in (("train", "r000_f000000", 12, 0), ("train", "r000_f000013", 17, 13), ("val", "r001_f000030", 20, 30)):
        match = out / split / "match1"
        frames = match / "frame" / rid
        assert sorted(p.name for p in frames.iterdir()) == sorted([f"{k}.png" for k in range(n)] + ["median.npz"])
        with open(match / "csv" / f"{rid}_ball.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["Frame", "Visibility", "X", "Y"]
        assert [int(r[0]) for r in rows[1:]] == list(range(n))
        img = cv2.imread(str(frames / "0.png"))
        assert img.shape == (288, 512, 3)
        # the label, in pixels of the exported image, sits on the white dot
        x, y = float(rows[1][2]), float(rows[1][3])
        assert abs(x - shuttle_xy(first)[0] * 512 / W) < 0.01 and abs(y - shuttle_xy(first)[1] * 288 / H) < 0.01
        assert img[int(round(y)), int(round(x))].min() > 180
        for med in (match / "median.npz", frames / "median.npz"):
            md = np.load(med)["median"]
            assert md.shape == (288, 512, 3) and md.dtype == np.uint8
            assert md[int(round(y)), int(round(x))].max() < 120  # the moving dot is not in the background
    with open(out / "train" / "match1" / "csv" / "r000_f000013_ball.csv") as f:
        rows = list(csv.reader(f))
    assert rows[1 + 7] == ["7", "0", "0", "0"]  # source frame 20: no shuttle

    # exporting again keeps the match number and replaces the folders; unreviewed labels on request
    manifest = export_dataset([(tiny_video, store.csv_path, rallies)], out, include_unreviewed=True, full_size=True)
    segs = manifest["matches"][0]["segments"]
    assert len(manifest["matches"]) == 1 and segs[-1]["frames"] == 30 and segs[-1]["status_counts"]["machine"] == 10
    assert cv2.imread(str(out / "val" / "match1" / "frame" / "r001_f000030" / "29.png")).shape == (H, W, 3)
    with open(out / "val" / "match1" / "csv" / "r001_f000030_ball.csv") as f:
        assert list(csv.reader(f))[1] == ["0", "1", str(shuttle_xy(30)[0]), str(shuttle_xy(30)[1])]

    store2 = an.LabelStore(tmp_path / "empty.labels.csv")
    store2.set(0, 1, 5, 5, "machine")
    store2.save()
    with pytest.raises(ValueError):
        export_dataset([(tiny_video, store2.csv_path, None)], tmp_path / "ds2")


def test_server_serves_frames_and_saves_labels(tiny_video):
    path = np.array([(1, *shuttle_xy(f)) for f in range(N)], float)
    store = an.LabelStore(tiny_video.with_suffix(".labels.csv"), annotator="tester")
    app = an.AnnotationApp(tiny_video, store, path, None, None, [(0, 29), (30, 59)], 30.0, (W, H))
    server = an.make_server(app, 0)
    host, port = server.server_address
    assert host == "127.0.0.1"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(items, headers=None):
        req = urllib.request.Request(base + "/api/label", json.dumps({"items": items}).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
        return json.loads(urllib.request.urlopen(req).read())

    try:
        assert b"<canvas" in urllib.request.urlopen(base + "/").read()
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert state["frames"] == N and len(state["pred"]) == N and state["progress"]["left"] == N
        for f in (5, 6, 40, 5):  # forward, then a seek back
            img = cv2.imdecode(np.frombuffer(urllib.request.urlopen(f"{base}/frame/{f}.jpg").read(), np.uint8), 1)
            assert img.shape == (H, W, 3) and img[shuttle_xy(f)[1], shuttle_xy(f)[0]].min() > 150
        res = post([{"frame": 5, "visibility": 1, "x": 40, "y": 35, "status": "human"},
                    {"frame": 6, "visibility": 0, "status": "skip"}])
        assert res["progress"]["done"] == 2 and res["progress"]["left"] == N - 2
        assert an.read_labels(store.csv_path)[5] == an.Label(1, 40.0, 35.0, "human")  # already on disk
        post([{"frame": 6, "delete": True}])
        assert 6 not in an.read_labels(store.csv_path)
        for bad, headers in (([{"frame": 5, "visibility": 1, "x": 1, "y": 1, "status": "machine"}], None),
                             ([{"frame": 9999, "visibility": 0}], None),
                             ([{"frame": 5, "visibility": 0}], {"Origin": "http://evil.example"}),
                             ([{"frame": 5, "visibility": 0}], {"Host": "evil.example"})):
            with pytest.raises(urllib.error.HTTPError):
                post(bad, headers)
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/frame/9999.jpg")
    finally:
        server.shutdown()
        server.server_close()
