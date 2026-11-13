"""Processing node: pull uploaded recordings from the cloud, analyse them here, push the results back.

The cloud (cloud/ in this repo, deployed at ba.llmat.dev) only stores files and
job state. This node runs on the Mac that has the models: it polls for a queued
recording, downloads the original, tells the cloud to delete its copy, runs the
normal ``badminton`` commands as subprocesses and uploads what they produced.

Only the standard library is used for the network side, so the node starts
without the analysis dependencies being imported.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_URL = "https://ba.llmat.dev"
CONFIG_PATH = Path.home() / ".config" / "badminton" / "node.json"
RESULT_SUFFIXES = {".png", ".json", ".mp4"}
SINGLE_PUT_LIMIT = 64 * 1024 * 1024   # larger results go up in parts
PART_SIZE = 16 * 1024 * 1024
DOWNLOAD_CHUNK = 4 * 1024 * 1024
WEB_HEIGHT = 1080                      # result videos are re-encoded for the phone
WEB_BITRATE = "6M"
WEB_IMAGE_BYTES = 1_500_000            # pictures larger than this (4K frames) are shrunk for the phone
WEB_IMAGE_WIDTH = 1920


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class StepFailed(RuntimeError):
    """An analysis step failed; the message is shown to the user on the web page."""


class StepFailedWithFiles(StepFailed):
    def __init__(self, message: str, files: list[Path]):
        super().__init__(message)
        self.files = files


@dataclass
class Config:
    url: str
    token: str
    work_dir: Path
    name: str

    @classmethod
    def load(cls, url: str | None = None, work_dir: str | None = None, config: Path = CONFIG_PATH) -> "Config":
        saved = json.loads(config.read_text()) if config.exists() else {}
        token = os.environ.get("BA_ACCESS_TOKEN") or saved.get("token") or ""
        if not token:
            raise SystemExit(
                f"没有登录口令。把网站的口令写进 {config}：\n"
                f'  mkdir -p {config.parent} && echo \'{{"token": "你的口令"}}\' > {config} && chmod 600 {config}\n'
                "或者设置环境变量 BA_ACCESS_TOKEN。"
            )
        return cls(
            url=(url or saved.get("url") or DEFAULT_URL).rstrip("/"),
            token=token,
            work_dir=Path(work_dir or saved.get("work_dir") or Path.home() / "badminton_uploads").expanduser(),
            name=saved.get("name") or platform.node() or "node",
        )


class Api:
    def __init__(self, cfg: Config):
        self.base = cfg.url + "/api/v1"
        self.token = cfg.token

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None,
                timeout: float = 120):
        req = urllib.request.Request(self.base + path, data=body, method=method)
        req.add_header("authorization", "Bearer " + self.token)
        req.add_header("user-agent", "badminton-node")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            try:
                text = json.loads(text).get("error", text)
            except ValueError:
                pass
            raise ApiError(e.code, text) from None

    def json(self, method: str, path: str, data: dict | None = None, timeout: float = 120) -> dict:
        body = json.dumps(data).encode() if data is not None else None
        headers = {"content-type": "application/json"} if data is not None else None
        with self.request(method, path, body, headers, timeout) as res:
            return json.loads(res.read() or b"{}")

    def retrying(self, fn, what: str, tries: int = 6):
        """Network hiccups are normal on a home connection: retry with backoff, give up on real errors."""
        for attempt in range(1, tries + 1):
            try:
                return fn()
            except ApiError as e:
                if e.status < 500 or attempt == tries:
                    raise
                err = e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if attempt == tries:
                    raise
                err = e
            wait = min(60, 2 ** attempt)
            print(f"  {what} 失败（{err}），{wait} 秒后重试")
            time.sleep(wait)

    # ---- originals ----

    def download(self, video: dict, dest: Path, on_bytes=None) -> None:
        size = video["size"]
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        with open(part, "ab") as f:
            while have < size:
                def chunk(start=have):
                    with self.request("GET", f"/videos/{video['id']}/file",
                                      headers={"range": f"bytes={start}-{size - 1}"}, timeout=300) as res:
                        got = 0
                        while True:
                            block = res.read(DOWNLOAD_CHUNK)
                            if not block:
                                break
                            f.write(block)
                            got += len(block)
                            if on_bytes:
                                on_bytes(start + got, size)
                        return got
                try:
                    have += chunk()
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                    f.flush()
                    have = part.stat().st_size  # keep what arrived, continue from there
                    print(f"  下载中断（{e}），从 {have / 1e6:.0f} MB 继续")
                    time.sleep(5)
        if part.stat().st_size != size:
            raise StepFailed(f"下载的大小不对：{part.stat().st_size} / {size}")
        part.rename(dest)

    # ---- results ----

    def upload_result(self, video_id: str, path: Path, name: str) -> None:
        size = path.stat().st_size
        base = f"/videos/{video_id}/results/{name}"
        if size <= SINGLE_PUT_LIMIT:
            data = path.read_bytes()
            self.retrying(lambda: self.request("PUT", base, data, {"content-type": "application/octet-stream"},
                                               timeout=600).close(), f"上传 {name}")
            return
        upload_id = self.json("POST", base + "/multipart")["uploadId"]
        parts = []
        with open(path, "rb") as f:
            n = 0
            while True:
                data = f.read(PART_SIZE)
                if not data:
                    break
                n += 1

                def put(n=n, data=data):
                    with self.request("PUT", f"{base}/multipart?uploadId={urllib.request.quote(upload_id, safe='')}&part={n}",
                                      data, {"content-type": "application/octet-stream"}, timeout=600) as res:
                        return json.loads(res.read())

                parts.append(self.retrying(put, f"上传 {name} 第 {n} 片"))
        self.json("POST", f"{base}/multipart?uploadId={urllib.request.quote(upload_id, safe='')}", {"parts": parts})


# ---------- progress ----------

STEPS = [
    ("download", "下载原片"),
    ("sdr", "HDR 转 SDR"),
    ("calibrate", "标定球场"),
    ("skeleton", "骨架识别"),
    ("label", "球员编号"),
    ("stats", "跑动和站位统计"),
    ("cut", "剪辑回合视频"),
    ("shuttle", "羽毛球追踪"),
    ("upload", "转码并上传结果"),
    ("review", "AI 报告"),
]
HISTORY_KEEP = 8


class History:
    """How long each step took on this machine before, per second of recording (download: per megabyte).

    Used only to estimate the time left; with no history a step simply has no estimate.
    """

    def __init__(self, path: Path | None):
        self.path = path
        try:
            self.data = json.loads(path.read_text()) if path and path.exists() else {}
        except (OSError, ValueError):
            self.data = {}

    def rate(self, key: str) -> float | None:
        values = sorted(self.data.get(key) or [])
        return values[len(values) // 2] if values else None

    def add(self, key: str, rate: float) -> None:
        self.data[key] = ((self.data.get(key) or []) + [round(rate, 4)])[-HISTORY_KEEP:]
        if self.path:
            try:
                self.path.write_text(json.dumps(self.data))
            except OSError:
                pass


class Progress:
    """The step list shown on the web page: what is done, what is running, how long it took, what is left."""

    def __init__(self, send, history: History, megabytes: float, clock=time.time):
        self.send = send              # send(stage_text, progress_dict)
        self.history = history
        self.megabytes = megabytes
        self.video_seconds: float | None = None
        self.clock = clock
        self.started = clock()
        self.steps = {k: {"key": k, "label": label, "status": "pending"} for k, label in STEPS}
        self.current: str | None = None
        self._sent = 0.0
        self._lock = threading.Lock()

    # -- step transitions --

    def start(self, key: str, detail: str = "") -> None:
        with self._lock:
            if self.current and self.current != key:
                self._finish(self.current, "done")
            step = self.steps[key]
            if step["status"] != "running":
                step.update(status="running", startedAt=self.clock(), detail=detail)
                step.pop("percent", None)
                step.pop("etaSeconds", None)
            self.current = key
        self.flush(force=True)

    def update(self, key: str, percent: float | None = None, detail: str | None = None,
               eta_seconds: float | None = None) -> None:
        with self._lock:
            step = self.steps[key]
            if percent is not None:
                step["percent"] = max(0, min(100, int(percent)))
            if detail is not None:
                step["detail"] = detail
            if eta_seconds is not None:
                step["etaSeconds"] = max(0, int(eta_seconds))
        self.flush()

    def done(self, key: str, detail: str | None = None) -> None:
        with self._lock:
            self._finish(key, "done", detail)
        self.flush(force=True)

    def skip(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.steps[key].update(status="skipped", detail=detail)
            if self.current == key:
                self.current = None
        self.flush(force=True)

    def fail(self) -> None:
        with self._lock:
            if self.current:
                self._finish(self.current, "failed")

    def finish_current(self) -> None:
        with self._lock:
            if self.current:
                self._finish(self.current, "done")

    def _finish(self, key: str, status: str, detail: str | None = None) -> None:
        step = self.steps[key]
        if step["status"] != "running":
            if status == "done" and step["status"] == "pending":  # finished without a visible start
                step.update(status="done", seconds=0)
            return
        seconds = self.clock() - step["startedAt"]
        step.update(status=status, seconds=round(seconds, 1))
        step.pop("etaSeconds", None)
        if detail is not None:
            step["detail"] = detail
        if status == "done":
            step["percent"] = 100
            unit = self.megabytes if key == "download" else self.video_seconds
            if unit and seconds > 0.5:
                self.history.add(key, seconds / unit)
        if self.current == key:
            self.current = None

    # -- estimates --

    def _estimate(self, key: str) -> float | None:
        rate = self.history.rate(key)
        unit = self.megabytes if key == "download" else self.video_seconds
        return rate * unit if rate is not None and unit else None

    def to_json(self) -> dict:
        now = self.clock()
        steps, eta, known = [], 0.0, True
        for key, _ in STEPS:
            step = dict(self.steps[key])
            if step["status"] == "running":
                step["seconds"] = round(now - step["startedAt"], 1)
                left = step.get("etaSeconds")
                if left is None:
                    est = self._estimate(key)
                    if est is not None:
                        left = max(0.0, est - step["seconds"])
                        step["etaSeconds"] = int(left)
                if left is None:
                    known = False
                else:
                    eta += left
            elif step["status"] == "pending":
                est = self._estimate(key)
                if est is None:
                    known = False
                else:
                    step["estimateSeconds"] = int(est)
                    eta += est
            step.pop("startedAt", None)
            steps.append(step)
        return {
            "steps": steps,
            "elapsedSeconds": int(now - self.started),
            # the total is only given when every remaining step has an estimate
            "etaSeconds": int(eta) if known else None,
            "videoSeconds": round(self.video_seconds, 1) if self.video_seconds else None,
        }

    def stage_text(self) -> str:
        step = self.steps.get(self.current or "")
        if not step:
            return "处理中"
        return step["label"] + (f" {step['percent']}%" if "percent" in step else "")

    def flush(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self._sent < 2:  # at most one request every couple of seconds
            return
        self._sent = now
        self.send(self.stage_text(), self.to_json())


# ---------- analysis ----------

_FRAMES = re.compile(r"(\d+)/(\d+) 帧")
_LEFT = re.compile(r"剩余约\s*([\d.]+) 分钟")


def run_step(args: list[str], log: list[str], on_line=None, on_frames=None) -> int:
    """Run one ``badminton`` command. Output lines go to ``log`` (for the error message) and ``on_line``;
    the frame counter that long steps print goes to ``on_frames(done, total, minutes_left)``."""
    cmd = [sys.executable, "-m", "badminton.cli", *args]
    print("  $ badminton " + " ".join(args))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    buf = b""
    while True:
        block = proc.stdout.read1(4096)
        if not block:
            break
        buf += block
        *lines, buf = re.split(rb"[\r\n]", buf)
        for raw in lines:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            m = _FRAMES.search(line)
            if m and int(m.group(2)):
                left = _LEFT.search(line)
                if on_frames:
                    on_frames(int(m.group(1)), int(m.group(2)), float(left.group(1)) if left else None)
                continue
            log.append(line)
            del log[:-60]
            if on_line:
                on_line(line)
    return proc.wait()


def web_video(src: Path, dst: Path) -> Path:
    """Re-encode a result video so a phone plays it: H.264, at most 1080p, moov atom first."""
    if not shutil.which("ffmpeg"):
        return src
    encoders = [["-c:v", "libx264", "-preset", "veryfast", "-crf", "24"]]
    if sys.platform == "darwin":  # hardware encoder first; fall back when it is busy or missing
        encoders.insert(0, ["-c:v", "h264_videotoolbox", "-b:v", WEB_BITRATE])
    for encoder in encoders:
        done = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-vf", f"scale=-2:'min({WEB_HEIGHT},ih)'", *encoder, "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst)],
            capture_output=True, text=True)
        if done.returncode == 0:
            break
    return dst if done.returncode == 0 and dst.exists() and dst.stat().st_size else src


def cover_image(video: Path, dst: Path, seconds: float) -> Path | None:
    """A small picture of the recording for the list on the web page."""
    if dst.exists():
        return dst
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    done = subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", f"{max(0.0, seconds):.2f}", "-i", str(video), "-frames:v", "1",
                           "-vf", "scale=640:-2", "-q:v", "5", str(dst)], stdin=subprocess.DEVNULL)
    return dst if done.returncode == 0 and dst.exists() else None


def web_image(src: Path, dst: Path) -> Path:
    """A 4K frame saved as PNG is 8-9 MB; the phone gets a 1920-wide JPEG instead. Cards stay as they are."""
    if src.stat().st_size <= WEB_IMAGE_BYTES or not shutil.which("ffmpeg"):
        return src
    done = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", f"scale='min({WEB_IMAGE_WIDTH},iw)':-2", "-q:v", "3", str(dst)],
        capture_output=True, text=True)
    return dst if done.returncode == 0 and dst.exists() and dst.stat().st_size else src


CALIBRATION_FRACTIONS = (0.5, 0.25, 0.75, 0.1, 0.9)  # where in the recording to look for a clear view of the court
GOOD_CONFIDENCE = 85                                 # stop looking once a fit is this good (percent)
_CONFIDENCE = re.compile(r"置信度 (\d+)%")


def auto_calibrate_video(work: Path, court: Path, duration: float, progress: Progress, log: list[str], run=None) -> str:
    """Automatic court calibration, tried at several moments of the recording.

    Players stand on different lines at different times, so one frame can fail
    where another passes. Every attempt goes to its own file; the fit with the
    highest confidence becomes ``court`` (with its preview picture). Returns a
    one-line description of what was chosen.
    """
    run = run or run_step
    times = [1.0] + [round(duration * f, 1) for f in CALIBRATION_FRACTIONS if duration * f > 2.0]
    best: tuple[int, Path, float] | None = None
    reason = ""
    rejected: Path | None = None
    for i, t in enumerate(times, 1):
        note = f"；目前最好 {best[0]}%" if best else ""
        progress.update("calibrate", percent=(i - 1) * 100 // len(times),
                        detail=f"第 {i}/{len(times)} 次，用第 {t:g} 秒的画面{note}")
        attempt = court.with_name(f"{court.stem}.try{i}.json")
        lines: list[str] = []  # this attempt's own output (the shared log is trimmed as it grows)
        code = run(["calibrate", str(work), "--auto", "--time", str(t), "-o", str(attempt)], lines)
        log.extend(lines)
        if code == 0 and attempt.exists():
            m = next((m for m in map(_CONFIDENCE.search, lines) if m), None)
            confidence = int(m.group(1)) if m else 0
            if best is None or confidence > best[0]:
                best = (confidence, attempt, t)
            if confidence >= GOOD_CONFIDENCE:
                break
        else:
            reason = next((x for x in lines if x.startswith("自动标定失败")), reason)
            if attempt.with_suffix(".rejected.png").exists():
                rejected = attempt.with_suffix(".rejected.png")
    if best is None:
        raise StepFailedWithFiles(
            f"自动标定球场失败（试了录像里的 {len(times)} 个时刻）。{reason}\n"
            "请确认整片球场（四条边线和两条底线）都在画面里；或者在 Mac 上手动标定后，在网页上点“重新处理”。",
            [rejected] if rejected else [])
    shutil.copyfile(best[1], court)
    if best[1].with_suffix(".png").exists():
        shutil.copyfile(best[1].with_suffix(".png"), court.with_suffix(".png"))
    chosen = f"采用第 {best[2]:g} 秒的画面，线条重合度 {best[0]}%"
    log.append("球场标定" + chosen)
    return chosen


# `badminton analyze` announces its stages with lines like "== 2/4 骨架 =="
_ANALYZE_STAGES = (("== 2/4", "skeleton"), ("== 3/4", "label"), ("== 4/4", "stats"), ("== 剪掉死球时间", "cut"))


SHUTTLE_VERSION = "3"  # raise when shuttle tracking changes, so recordings already on this Mac are tracked again
_SEEN = re.compile(r"(\d+) 帧有球（([\d.]+)%）")


def track_shuttle(work: Path, court: Path, out_dir: Path, progress: Progress, log: list[str], run=None) -> Path | None:
    """Shuttle track drawn on the skeleton video. Returns the combined video, or None when it could not be
    made: the shuttle model is the least reliable part, so a failure here does not fail the whole job."""
    run = run or run_step
    skeleton = out_dir / f"{work.stem}.skeleton.mp4"
    csv = out_dir / f"{work.stem}.shuttle.csv"
    combined = out_dir / f"{work.stem}.combined.mp4"
    if not skeleton.exists():
        progress.skip("shuttle", "没有骨架视频，画不了球的轨迹")
        return None
    stamp = out_dir / f"{work.stem}.shuttle.version"  # which tracking code made the files: a re-run after an update tracks again
    fresh = stamp.exists() and stamp.read_text().strip() == SHUTTLE_VERSION
    if fresh and combined.exists() and csv.exists() and combined.stat().st_mtime >= skeleton.stat().st_mtime:
        progress.skip("shuttle", "沿用上次的结果")
        return combined
    progress.start("shuttle", "逐帧找球")
    seen = ""

    def on_line(line: str) -> None:
        nonlocal seen
        m = _SEEN.search(line)
        if m:
            seen = f"{m.group(2)}% 的画面里找到了球"

    def on_frames(done: int, total: int, minutes_left: float | None) -> None:
        progress.update("shuttle", percent=done * 80 // total, detail=f"逐帧找球 {done}/{total} 帧")

    mark = len(log)
    ok = run(["shuttle", str(work), "--court", str(court), "-o", str(csv)], log, on_line, on_frames) == 0 and csv.exists()
    if ok:
        progress.update("shuttle", percent=80, detail="把球的轨迹画到骨架视频上")
        ok = run(["overlay", str(skeleton), "--shuttle", str(csv), "-o", str(combined)], log, None, None) == 0
    if not ok or not combined.exists():
        reason = log[-1] if len(log) > mark else "没有输出"
        print(f"  羽毛球追踪没有成功：{reason}")
        progress.skip("shuttle", f"这次没有成功，其它结果不受影响（{reason[:120]}）")
        return None
    stamp.write_text(SHUTTLE_VERSION)
    progress.done("shuttle", seen or None)
    return combined


def review_inputs(work: Path, court: Path, out_dir: Path, log: list[str], run=None) -> bool:
    """Write <stem>.review.request.json: the statistics as text plus the prompt. No model is called here;
    the web site asks the model with its own key once the file is uploaded."""
    movement = out_dir / f"{work.stem}.movement.json"
    if not movement.exists():
        return False
    args = ["review", str(movement)]
    positioning = out_dir / f"{work.stem}.positioning.json"
    if positioning.exists():
        args.append(str(positioning))
    args += ["--court", str(court), "-o", str(out_dir)]
    csv = out_dir / f"{work.stem}.shuttle.csv"
    if csv.exists():
        args += ["--shuttle", str(csv), "--video", str(work)]
    mine: list[str] = []  # its chatter (how to paste the prompt) is of no use in the job's error log
    if (run or run_step)(args, mine, None, None) != 0:
        print("  生成 AI 报告的统计摘要失败：" + (mine[-1] if mine else ""))
        return False
    return True


def ask_for_review(api: "Api", vid: str, progress: Progress, names: set[str]) -> None:
    """Have the web site write the AI report. Never fails the job: the results are already up."""
    if "review.request.json" not in names:
        progress.skip("review", "这次没有生成统计摘要")
        return
    progress.start("review", "把统计数字交给大模型写点评（不发视频）")
    try:
        api.json("POST", f"/videos/{vid}/review", {}, timeout=600)
    except ApiError as e:
        progress.skip("review", str(e).split(": ", 1)[-1][:200])
        return
    except Exception as e:
        progress.skip("review", f"没连上网站：{type(e).__name__}")
        return
    progress.done("review")


def analyse(video: Path, out_dir: Path, options: dict, progress: Progress, log: list[str]) -> list[Path]:
    """The normal one-click pipeline, without anything that needs a person at the keyboard."""
    from .video import probe  # needs OpenCV; imported late so the node itself starts without it

    info = probe(video)
    if info.frames and info.fps:
        progress.video_seconds = info.frames / info.fps
    work = video
    if info.is_hdr:
        work = video.with_name(video.stem + "_sdr.mp4")
        if work.exists():
            progress.skip("sdr", "沿用上次转好的")
        else:
            progress.start("sdr", "iPhone 的 HDR 颜色转成普通颜色")
            if run_step(["sdr", str(video), str(work)], log) != 0:
                raise StepFailed("HDR 转 SDR 失败")
            progress.done("sdr")
    else:
        progress.skip("sdr", "不是 HDR，不需要")
    court = work.with_suffix(".court.json")
    extra: list[Path] = []
    if court.exists():
        progress.skip("calibrate", "沿用已有的标定")
    else:
        progress.start("calibrate")
        chosen = auto_calibrate_video(work, court, progress.video_seconds or 0.0, progress, log)
        progress.done("calibrate", chosen)
    if court.with_suffix(".png").exists():
        extra.append(court.with_suffix(".png"))
    args = ["analyze", str(work), "--court", str(court), "-o", str(out_dir)]
    if options.get("mode") != "doubles":
        args.append("--singles")
    if options.get("names"):
        args += ["--names", str(options["names"])]

    def on_line(line: str) -> None:
        for prefix, key in _ANALYZE_STAGES:
            if line.startswith(prefix):
                if "沿用" in line:
                    progress.finish_current()
                    progress.skip(key, "沿用上次的结果")
                else:
                    progress.start(key)

    def on_frames(done: int, total: int, minutes_left: float | None) -> None:
        if progress.current == "skeleton":
            progress.update("skeleton", percent=done * 100 // total, detail=f"{done}/{total} 帧",
                            eta_seconds=minutes_left * 60 if minutes_left is not None else None)

    code = run_step(args, log, on_line, on_frames)
    if code != 0:
        raise StepFailed("分析失败")
    progress.finish_current()
    for key in ("skeleton", "label", "stats", "cut"):  # anything analyze did not announce
        if progress.steps[key]["status"] == "pending":
            progress.skip(key, "这次没有运行")
    combined = track_shuttle(work, court, out_dir, progress, log)
    review_inputs(work, court, out_dir, log)
    files = sorted(p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in RESULT_SUFFIXES)
    if combined:  # the combined video is the skeleton video plus the shuttle: no need to send both
        files = [p for p in files if p.name != f"{work.stem}.skeleton.mp4"]
    else:
        files = [p for p in files if p.name != f"{work.stem}.combined.mp4"]
    return extra + files


def result_name(path: Path, stem: str) -> str:
    """Names on the web are the same for every recording: strip the recording's own file stem."""
    name = path.name
    for prefix in (stem + "_sdr.", stem + "."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = re.sub(r"^court\.try\d+\.", "court.", name)  # calibration attempts: one name on the web
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120]


# ---------- the loop ----------

def process(api: Api, cfg: Config, video: dict, analyse_fn=analyse) -> None:
    vid = video["id"]
    job_dir = cfg.work_dir / vid
    job_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", video["filename"]) or "video.mov"
    local = job_dir / safe
    stop = threading.Event()

    def send(stage: str, data: dict) -> None:
        try:
            api.json("POST", f"/videos/{vid}/job", {"status": "processing", "stage": stage, "progress": data})
        except Exception as e:  # a missed progress report is not worth failing the job for
            print(f"  汇报进度失败：{e}")

    megabytes = video["size"] / 1e6
    progress = Progress(send, History(cfg.work_dir / "timings.json"), megabytes)

    def heartbeat() -> None:  # keeps the lease alive and the clock on the web page moving
        while not stop.wait(20):
            progress.flush(force=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    log: list[str] = []
    try:
        if local.exists():
            progress.skip("download", "原片已经在这台 Mac 上")
        else:
            if video.get("originalDeleted"):
                raise StepFailed("云端的原片已经删除，这台 Mac 上也找不到，请重新上传")
            progress.start("download", f"0 / {megabytes:.0f} MB")
            print(f"  下载 {video['filename']}（{megabytes:.0f} MB）")
            t0 = time.time()

            def on_bytes(have: int, size: int) -> None:
                speed = have / max(time.time() - t0, 0.5)
                progress.update("download", percent=have * 100 // size,
                                detail=f"{have / 1e6:.0f} / {size / 1e6:.0f} MB，{speed * 8 / 1e6:.0f} Mbps",
                                eta_seconds=(size - have) / speed if speed > 0 else None)

            api.download(video, local, on_bytes)
            progress.done("download", f"{megabytes:.0f} MB")
        if not video.get("originalDeleted"):
            api.retrying(lambda: api.json("POST", f"/videos/{vid}/original/release"), "删除云端原片")
            print("  云端原片已删除，本地保留在 " + str(local))
        # one folder per mode: the skeleton data of a singles run (one player per side) is no use for doubles
        options = video.get("options") or {}
        out_dir = job_dir / ("analysis-doubles" if options.get("mode") == "doubles" else "analysis")
        try:
            files = analyse_fn(local, out_dir, options, progress, log)
        except StepFailedWithFiles as e:
            (job_dir / "web").mkdir(exist_ok=True)
            for f in e.files:
                name = result_name(f, local.stem)
                small = web_image(f, job_dir / "web" / (name[:-4] + ".jpg"))
                api.upload_result(vid, small, name[:-4] + small.suffix.lower())
            raise
        web_dir = job_dir / "web"
        web_dir.mkdir(exist_ok=True)
        progress.start("upload")
        for i, f in enumerate(files, 1):
            name = result_name(f, local.stem)
            progress.update("upload", percent=(i - 1) * 100 // len(files), detail=f"第 {i}/{len(files)} 个：{name}")
            if f.suffix.lower() == ".mp4":
                f = web_video(f, web_dir / name)
            elif f.suffix.lower() == ".png":
                f = web_image(f, web_dir / (name[:-4] + ".jpg"))
                name = name[:-4] + f.suffix.lower()
            print(f"  上传 {name}（{f.stat().st_size / 1e6:.1f} MB）")
            api.upload_result(vid, f, name)
        cover = cover_image(local.with_name(local.stem + "_sdr.mp4") if local.with_name(local.stem + "_sdr.mp4").exists() else local,
                            web_dir / "cover.jpg", (progress.video_seconds or 0) * 0.1)
        if cover:
            api.upload_result(vid, cover, "cover.jpg")
        progress.done("upload", f"{len(files)} 个文件")
        ask_for_review(api, vid, progress, {result_name(f, local.stem) for f in files})
        stop.set()
        api.retrying(lambda: api.json("POST", f"/videos/{vid}/job",
                                      {"status": "done", "stage": "完成", "progress": progress.to_json()}), "汇报完成")
        print("  完成")
    except Exception as e:
        stop.set()
        stage = progress.stage_text()
        progress.fail()
        detail = str(e) if isinstance(e, StepFailed) else f"{type(e).__name__}: {e}"
        tail = "\n".join(log[-12:])
        print(f"  失败：{detail}")
        try:
            api.json("POST", f"/videos/{vid}/job", {"status": "failed", "stage": stage, "progress": progress.to_json(),
                                                    "error": (detail + "\n\n" + tail).strip()})
        except Exception as e2:
            print(f"  汇报失败状态时出错：{e2}")
    finally:
        stop.set()


def run(cfg: Config, once: bool = False, interval: float = 15.0, analyse_fn=analyse) -> int:
    api = Api(cfg)
    try:
        api.json("GET", "/me")
    except ApiError as e:
        raise SystemExit("口令不对，连不上 " + cfg.url if e.status == 401 else f"连不上 {cfg.url}：{e}") from None
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    try:  # when the output goes to a log file, write each line at once instead of in 8 KB blocks
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    print(f"处理节点 {cfg.name} 已连接 {cfg.url}，原片和结果放在 {cfg.work_dir}。Ctrl+C 退出。")
    while True:
        try:
            video = api.json("POST", "/node/claim", {"node": cfg.name}).get("video")
        except (ApiError, urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"查询任务失败：{e}")
            video = None
        if video:
            print(f"{time.strftime('%H:%M:%S')} 开始处理 {video['filename']}（{video['id']}）")
            process(api, cfg, video, analyse_fn)
            continue
        if once:
            return 0
        time.sleep(interval)
