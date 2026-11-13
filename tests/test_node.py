"""The processing node against a small in-process stand-in for the cloud API (no network, no models)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from badminton import node


class FakeCloud:
    """Just enough of cloud/src/index.ts for the node: one queued recording, results kept in memory."""

    def __init__(self, original: bytes, options=None, drop_first_download=False):
        self.original = original
        self.video = {"id": "11111111-2222-3333-4444-555555555555", "filename": "IMG 0964.mov", "size": len(original),
                      "options": options or {"mode": "singles"}, "originalDeleted": False}
        self.queued = True
        self.released = False
        self.jobs: list[dict] = []
        self.results: dict[str, bytes] = {}
        self.multipart: dict[int, bytes] = {}
        self.drop = drop_first_download
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
                if self.headers.get("authorization") != "Bearer secret":
                    return self.send({"error": "口令不对"}, 401)
                if self.path == "/api/v1/me":
                    return self.send({"user": "owner"})
                if self.path.endswith("/file"):
                    start = int(self.headers["range"].split("=")[1].split("-")[0])
                    data = cloud.original[start:]
                    self.send_response(206)
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    if cloud.drop:  # send half, then cut the connection
                        cloud.drop = False
                        self.wfile.write(data[: len(data) // 2])
                        self.wfile.flush()
                        self.connection.close()
                        return
                    self.wfile.write(data)
                    return
                self.send({"error": "?"}, 404)

            def do_POST(self):
                data = self.body()
                if self.path == "/api/v1/node/claim":
                    video, cloud.queued = (cloud.video if cloud.queued else None), False
                    return self.send({"video": video})
                if self.path.endswith("/original/release"):
                    cloud.released = True
                    return self.send({})
                if self.path.endswith("/job"):
                    cloud.jobs.append(json.loads(data))
                    return self.send({})
                if self.path.endswith("/multipart"):
                    return self.send({"uploadId": "up/1+2"}, 201)
                if "/multipart?uploadId=up%2F1%2B2" in self.path:
                    parts = json.loads(data)["parts"]
                    name = self.path.split("/results/")[1].split("/")[0]
                    cloud.results[name] = b"".join(cloud.multipart[p["partNumber"]] for p in parts)
                    return self.send({})
                self.send({"error": "?"}, 404)

            def do_PUT(self):
                data = self.body()
                if "/multipart?" in self.path:
                    n = int(self.path.split("part=")[1])
                    cloud.multipart[n] = data
                    return self.send({"partNumber": n, "etag": f"e{n}"})
                cloud.results[self.path.split("/results/")[1]] = data
                self.send({})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


@pytest.fixture
def cloud(request):
    c = FakeCloud(b"x" * 300_000, **getattr(request, "param", {}))
    yield c
    c.server.shutdown()


def config(cloud, tmp_path):
    return node.Config(url=cloud.url, token="secret", work_dir=tmp_path / "work", name="test-mac")


def fake_analyse(video: Path, out_dir: Path, options, on_stage, log):
    out_dir.mkdir(parents=True, exist_ok=True)
    on_stage("骨架 50%")
    (out_dir / f"{video.stem}.movement.json").write_text(json.dumps({"mode": options.get("mode")}))
    (out_dir / f"{video.stem}.card.P1.png").write_bytes(b"png")
    (out_dir / f"{video.stem}.skeleton.npz").write_bytes(b"not uploaded")
    return sorted(p for p in out_dir.iterdir() if p.suffix in node.RESULT_SUFFIXES)


@pytest.mark.parametrize("cloud", [{}, {"drop_first_download": True}], indirect=True)
def test_downloads_releases_uploads_and_finishes(cloud, tmp_path):
    assert node.run(config(cloud, tmp_path), once=True, analyse_fn=fake_analyse) == 0
    local = tmp_path / "work" / cloud.video["id"] / "IMG_0964.mov"
    assert local.read_bytes() == cloud.original          # also after a dropped connection
    assert cloud.released                                # the cloud copy is deleted once the Mac has it
    assert set(cloud.results) == {"movement.json", "card.P1.png"}   # stem stripped, .npz stays local
    assert json.loads(cloud.results["movement.json"]) == {"mode": "singles"}
    assert cloud.jobs[-1] == {"status": "done", "stage": "完成"}
    assert any(j.get("stage") == "下载原片" for j in cloud.jobs)


def test_failure_is_reported_with_the_log_tail(cloud, tmp_path):
    def broken(video, out_dir, options, on_stage, log):
        log.append("自动标定失败：只找到 3 条线")
        raise node.StepFailed("自动标定球场失败")

    node.run(config(cloud, tmp_path), once=True, analyse_fn=broken)
    last = cloud.jobs[-1]
    assert last["status"] == "failed"
    assert "自动标定球场失败" in last["error"] and "只找到 3 条线" in last["error"]
    assert not cloud.results


def test_large_results_go_up_in_parts(cloud, tmp_path, monkeypatch):
    monkeypatch.setattr(node, "SINGLE_PUT_LIMIT", 1000)
    monkeypatch.setattr(node, "PART_SIZE", 400)
    monkeypatch.setattr(node, "web_video", lambda src, dst: src)
    payload = bytes(range(256)) * 5  # 1280 bytes: 4 parts

    def big(video, out_dir, options, on_stage, log):
        out_dir.mkdir(parents=True, exist_ok=True)
        f = out_dir / f"{video.stem}.rallies.mp4"
        f.write_bytes(payload)
        return [f]

    node.run(config(cloud, tmp_path), once=True, analyse_fn=big)
    assert cloud.results == {"rallies.mp4": payload}
    assert sorted(cloud.multipart) == [1, 2, 3, 4]
    assert cloud.jobs[-1]["status"] == "done"


def test_wrong_token_stops_with_a_clear_message(cloud, tmp_path):
    cfg = config(cloud, tmp_path)
    cfg.token = "nope"
    with pytest.raises(SystemExit, match="口令不对"):
        node.run(cfg, once=True, analyse_fn=fake_analyse)


def test_result_names_drop_the_recording_stem():
    assert node.result_name(Path("IMG_1_sdr.movement.json"), "IMG_1") == "movement.json"
    assert node.result_name(Path("IMG_1.court.png"), "IMG_1") == "court.png"
    assert node.result_name(Path("别的 文件.png"), "IMG_1") == "_____.png"


def calib_runner(outcomes):
    """Stand-in for run_step: outcomes[i] is a confidence in percent (success) or None (failure)."""
    calls = []

    def run(args, on_stage, label, log):
        t, out = float(args[args.index("--time") + 1]), Path(args[args.index("-o") + 1])
        result = outcomes[len(calls)]
        calls.append(t)
        if result is None:
            log.append("自动标定失败：结果不可靠（线条重合度 54%，只对上 6/12 条线）")
            out.with_suffix(".rejected.png").write_bytes(b"red")
            return 2
        out.write_text(json.dumps({"confidence": result}))
        out.with_suffix(".png").write_bytes(b"green %d" % result)
        log.append(f"置信度 {result}%（投影的球场线落在白线上的比例），对上 10/12 条线")
        return 0

    return run, calls


def test_calibration_tries_several_moments_and_keeps_the_best(tmp_path):
    court = tmp_path / "IMG_sdr.court.json"
    run, calls = calib_runner([None, 73, None, 78, None, 71])
    log = []
    node.auto_calibrate_video(tmp_path / "IMG_sdr.mp4", court, 20.0, lambda s: None, log, run=run)
    assert calls == [1.0, 10.0, 5.0, 15.0, 18.0]            # 10% of 20 s is too close to the start
    assert json.loads(court.read_text()) == {"confidence": 78}
    assert court.with_suffix(".png").read_bytes() == b"green 78"
    assert "15.0 秒" in log[-1] and "78%" in log[-1]


def test_calibration_stops_at_a_good_fit(tmp_path):
    run, calls = calib_runner([90])
    node.auto_calibrate_video(tmp_path / "v.mp4", tmp_path / "v.court.json", 600.0, lambda s: None, [], run=run)
    assert calls == [1.0]


def test_calibration_failure_names_the_reason_and_keeps_the_rejected_picture(tmp_path):
    run, calls = calib_runner([None] * 6)
    with pytest.raises(node.StepFailedWithFiles) as e:
        node.auto_calibrate_video(tmp_path / "v.mp4", tmp_path / "v.court.json", 60.0, lambda s: None, [], run=run)
    assert len(calls) == 6
    assert "只对上 6/12 条线" in str(e.value)
    assert [node.result_name(f, "v") for f in e.value.files] == ["court.rejected.png"]
    assert not (tmp_path / "v.court.json").exists()
