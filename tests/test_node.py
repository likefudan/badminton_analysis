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
        self.reviews = 0
        self.review_error = ""
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
                if self.path.endswith("/review"):
                    cloud.reviews += 1
                    if cloud.review_error:
                        return self.send({"error": cloud.review_error}, 503)
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


def fake_analyse(video: Path, out_dir: Path, options, progress, log):
    out_dir.mkdir(parents=True, exist_ok=True)
    progress.video_seconds = 20.0
    progress.skip("sdr", "不是 HDR，不需要")
    progress.skip("calibrate", "沿用已有的标定")
    progress.start("skeleton")
    progress.update("skeleton", percent=50, detail="600/1200 帧", eta_seconds=30)
    progress.start("stats")
    progress.finish_current()
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
    last = cloud.jobs[-1]
    assert (last["status"], last["stage"]) == ("done", "完成")
    steps = {s["key"]: s for s in last["progress"]["steps"]}
    assert [s["key"] for s in last["progress"]["steps"]] == [k for k, _ in node.STEPS]
    assert steps["download"]["status"] == "done" and steps["download"]["percent"] == 100
    assert steps["sdr"]["status"] == "skipped" and steps["skeleton"]["status"] == "done"
    assert steps["upload"]["detail"] == "2 个文件"
    assert steps["review"]["status"] == "skipped" and cloud.reviews == 0   # no statistics summary, nothing to ask
    assert any(j["stage"].startswith("下载原片") for j in cloud.jobs)
    # every report carries the whole step list, so the page can draw it from any one of them
    assert all(len(j["progress"]["steps"]) == len(node.STEPS) for j in cloud.jobs)


def test_failure_is_reported_with_the_log_tail(cloud, tmp_path):
    def broken(video, out_dir, options, progress, log):
        progress.start("calibrate")
        log.append("自动标定失败：只找到 3 条线")
        raise node.StepFailed("自动标定球场失败")

    node.run(config(cloud, tmp_path), once=True, analyse_fn=broken)
    last = cloud.jobs[-1]
    assert last["status"] == "failed"
    assert "自动标定球场失败" in last["error"] and "只找到 3 条线" in last["error"]
    assert last["stage"] == "标定球场"
    steps = {s["key"]: s["status"] for s in last["progress"]["steps"]}
    assert steps["calibrate"] == "failed" and steps["skeleton"] == "pending"
    assert not cloud.results


def test_large_results_go_up_in_parts(cloud, tmp_path, monkeypatch):
    monkeypatch.setattr(node, "SINGLE_PUT_LIMIT", 1000)
    monkeypatch.setattr(node, "PART_SIZE", 400)
    monkeypatch.setattr(node, "web_video", lambda src, dst: src)
    payload = bytes(range(256)) * 5  # 1280 bytes: 4 parts

    def big(video, out_dir, options, progress, log):
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


def quiet_progress(history=None, megabytes=100.0, clock=None):
    return node.Progress(lambda stage, data: None, history or node.History(None), megabytes, **({"clock": clock} if clock else {}))


def calib_runner(outcomes):
    """Stand-in for run_step: outcomes[i] is a confidence in percent (success) or None (failure)."""
    calls = []

    def run(args, log):
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
    progress = quiet_progress()
    progress.start("calibrate")
    chosen = node.auto_calibrate_video(tmp_path / "IMG_sdr.mp4", court, 20.0, progress, log, run=run)
    assert calls == [1.0, 10.0, 5.0, 15.0, 18.0]            # 10% of 20 s is too close to the start
    assert json.loads(court.read_text()) == {"confidence": 78}
    assert court.with_suffix(".png").read_bytes() == b"green 78"
    assert chosen == "采用第 15 秒的画面，线条重合度 78%" and chosen in log[-1]
    assert progress.steps["calibrate"]["detail"] == "第 5/5 次，用第 18 秒的画面；目前最好 78%"


def test_calibration_stops_at_a_good_fit(tmp_path):
    run, calls = calib_runner([90])
    node.auto_calibrate_video(tmp_path / "v.mp4", tmp_path / "v.court.json", 600.0, quiet_progress(), [], run=run)
    assert calls == [1.0]


def test_calibration_failure_names_the_reason_and_keeps_the_rejected_picture(tmp_path):
    run, calls = calib_runner([None] * 6)
    with pytest.raises(node.StepFailedWithFiles) as e:
        node.auto_calibrate_video(tmp_path / "v.mp4", tmp_path / "v.court.json", 60.0, quiet_progress(), [], run=run)
    assert len(calls) == 6
    assert "只对上 6/12 条线" in str(e.value)
    assert [node.result_name(f, "v") for f in e.value.files] == ["court.rejected.png"]
    assert not (tmp_path / "v.court.json").exists()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_progress_times_steps_and_estimates_the_rest_from_earlier_runs(tmp_path):
    clock = Clock()
    history = node.History(tmp_path / "timings.json")
    # first recording on this machine: nothing to estimate from
    p = quiet_progress(history, megabytes=200.0, clock=clock)
    p.video_seconds = 20.0
    p.start("download")
    clock.t += 10
    first = p.to_json()
    assert first["etaSeconds"] is None and first["elapsedSeconds"] == 10
    assert first["steps"][0]["seconds"] == 10 and "estimateSeconds" not in first["steps"][3]
    for key, seconds in (("sdr", 8), ("calibrate", 6), ("skeleton", 40), ("label", 2), ("stats", 4), ("cut", 6), ("shuttle", 10), ("upload", 12), ("review", 30)):
        p.start(key)          # starting the next step finishes the one before
        clock.t += seconds
    p.done("review")
    done = p.to_json()
    assert [s["status"] for s in done["steps"]] == ["done"] * 10
    assert [s["seconds"] for s in done["steps"]] == [10, 8, 6, 40, 2, 4, 6, 10, 12, 30]
    assert done["etaSeconds"] == 0

    # second recording, twice as long and twice as big: estimates scale with it
    p2 = quiet_progress(node.History(tmp_path / "timings.json"), megabytes=400.0, clock=clock)
    p2.video_seconds = 40.0
    p2.start("download")
    clock.t += 5
    p2.update("download", percent=30, eta_seconds=12)       # the step's own estimate wins over history
    j = p2.to_json()
    assert j["steps"][0]["etaSeconds"] == 12
    assert j["steps"][3]["estimateSeconds"] == 80           # skeleton: 40 s for 20 s of video -> 80 s for 40 s
    assert j["etaSeconds"] == 12 + 16 + 12 + 80 + 4 + 8 + 12 + 20 + 24 + 60
    p2.start("sdr")
    clock.t += 6
    assert p2.to_json()["steps"][1]["etaSeconds"] == 10     # 16 s expected, 6 s gone
    assert p2.stage_text() == "HDR 转 SDR"
    p2.fail()
    assert p2.to_json()["steps"][1]["status"] == "failed"


def test_big_pictures_are_shrunk_for_the_phone_and_cards_are_not(cloud, tmp_path, monkeypatch):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    monkeypatch.setattr(node, "WEB_IMAGE_BYTES", 20_000)
    frame = tmp_path / "frame.png"   # noise does not compress: a "4K frame"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "nullsrc=s=2560x1440,geq=random(1)*255:128:128",
                    "-frames:v", "1", str(frame)], check=True)
    card = tmp_path / "card.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=navy:s=900x1200",
                    "-frames:v", "1", str(card)], check=True)

    def pictures(video, out_dir, options, progress, log):
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(frame, out_dir / f"{video.stem}.players.png")
        shutil.copy(card, out_dir / f"{video.stem}.movement.1.png")
        return sorted(out_dir.iterdir())

    node.run(config(cloud, tmp_path), once=True, analyse_fn=pictures)
    assert set(cloud.results) == {"players.jpg", "movement.1.png"}
    assert cloud.results["players.jpg"][:2] == b"\xff\xd8"                 # a JPEG
    assert cloud.results["movement.1.png"] == card.read_bytes()             # untouched
    out = tmp_path / "work" / cloud.video["id"] / "web" / "players.jpg"
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True)
    assert probe.stdout.strip() == "1920"


def shuttle_runner(fail: str | None = None):
    calls = []

    def run(args, log, on_line=None, on_frames=None):
        calls.append(args[0])
        if args[0] == fail:
            log.append("找不到模型")
            return 1
        out = Path(args[args.index("-o") + 1])
        out.write_bytes(b"x")
        if args[0] == "shuttle":
            on_frames(50, 100, 0.1)
            on_line(f"已写入 {out}：100 帧，70 帧有球（70.0%），其中 3 帧是插值补出来的（Source 列）")
        return 0

    return run, calls


def test_shuttle_track_is_drawn_on_the_skeleton_video(tmp_path):
    (tmp_path / "v.skeleton.mp4").write_bytes(b"s")
    run, calls = shuttle_runner()
    progress = quiet_progress()
    out = node.track_shuttle(tmp_path / "v.mp4", tmp_path / "v.court.json", tmp_path, progress, [], run=run)
    assert calls == ["shuttle", "overlay"] and out == tmp_path / "v.combined.mp4"
    step = progress.steps["shuttle"]
    assert step["status"] == "done" and step["detail"] == "70.0% 的画面里找到了球"
    # a second run of the same recording does not track again
    assert node.track_shuttle(tmp_path / "v.mp4", tmp_path / "v.court.json", tmp_path, progress, [], run=run) == out
    assert calls == ["shuttle", "overlay"]
    # ... unless the tracking code has changed since
    (tmp_path / "v.shuttle.version").write_text("0")
    node.track_shuttle(tmp_path / "v.mp4", tmp_path / "v.court.json", tmp_path, progress, [], run=run)
    assert calls == ["shuttle", "overlay"] * 2


@pytest.mark.parametrize("fail", ["shuttle", "overlay"])
def test_shuttle_failure_does_not_fail_the_job(tmp_path, fail):
    (tmp_path / "v.skeleton.mp4").write_bytes(b"s")
    run, _ = shuttle_runner(fail)
    progress = quiet_progress()
    assert node.track_shuttle(tmp_path / "v.mp4", tmp_path / "v.court.json", tmp_path, progress, [], run=run) is None
    step = progress.steps["shuttle"]
    assert step["status"] == "skipped" and "找不到模型" in step["detail"]


def with_review_request(video: Path, out_dir: Path, options, progress, log):
    files = fake_analyse(video, out_dir, options, progress, log)
    request = out_dir / f"{video.stem}.review.request.json"
    request.write_text(json.dumps({"system": "s", "user": "u"}))
    return files + [request]


def test_the_site_is_asked_for_the_ai_report_after_the_upload(cloud, tmp_path):
    node.run(config(cloud, tmp_path), once=True, analyse_fn=with_review_request)
    assert "review.request.json" in cloud.results and cloud.reviews == 1
    last = cloud.jobs[-1]
    assert last["status"] == "done" and last["progress"]["steps"][-1]["status"] == "done"


def test_no_api_key_on_the_site_skips_the_report_and_keeps_the_results(cloud, tmp_path):
    cloud.review_error = "网站还没有配置 ANTHROPIC_API_KEY，出不了 AI 报告"
    node.run(config(cloud, tmp_path), once=True, analyse_fn=with_review_request)
    last = cloud.jobs[-1]
    step = last["progress"]["steps"][-1]
    assert last["status"] == "done" and step["status"] == "skipped" and "ANTHROPIC_API_KEY" in step["detail"]


def test_review_inputs_use_what_the_analysis_produced(tmp_path):
    calls = []

    def run(args, log, on_line=None, on_frames=None):
        calls.append(args)
        return 0

    work, court = tmp_path / "v.mp4", tmp_path / "v.court.json"
    assert not node.review_inputs(work, court, tmp_path, [], run=run) and not calls   # no movement statistics
    for name in ("v.movement.json", "v.positioning.json", "v.shuttle.csv"):
        (tmp_path / name).write_text("{}")
    assert node.review_inputs(work, court, tmp_path, [], run=run)
    assert calls == [["review", str(tmp_path / "v.movement.json"), str(tmp_path / "v.positioning.json"),
                      "--court", str(court), "-o", str(tmp_path), "--shuttle", str(tmp_path / "v.shuttle.csv"),
                      "--video", str(work)]]
