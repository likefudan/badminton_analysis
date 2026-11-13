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
