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

    def json(self, method: str, path: str, data: dict | None = None) -> dict:
        body = json.dumps(data).encode() if data is not None else None
        headers = {"content-type": "application/json"} if data is not None else None
        with self.request(method, path, body, headers) as res:
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

    def download(self, video: dict, dest: Path) -> None:
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


# ---------- analysis ----------

_PROGRESS = re.compile(r"(\d+)/(\d+) 帧")


def run_step(args: list[str], on_stage, label: str, log: list[str]) -> int:
    """Run one ``badminton`` command, keep its output for the error message, report progress."""
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
            m = _PROGRESS.search(line)
            if m and int(m.group(2)):
                on_stage(f"{label} {int(m.group(1)) * 100 // int(m.group(2))}%")
                continue
            log.append(line)
            del log[:-60]
            if line.startswith("== "):
                on_stage(line.strip("= ").split("（")[0])
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


def analyse(video: Path, out_dir: Path, options: dict, on_stage, log: list[str]) -> list[Path]:
    """The normal one-click pipeline, without anything that needs a person at the keyboard."""
    from .video import probe  # needs OpenCV; imported late so the node itself starts without it

    work = video
    if probe(video).is_hdr:
        work = video.with_name(video.stem + "_sdr.mp4")
        if not work.exists():
            on_stage("HDR 转 SDR")
            if run_step(["sdr", str(video), str(work)], on_stage, "HDR 转 SDR", log) != 0:
                raise StepFailed("HDR 转 SDR 失败")
    court = work.with_suffix(".court.json")
    extra: list[Path] = []
    if not court.exists():
        on_stage("自动标定球场")
        code = run_step(["calibrate", str(work), "--auto", "-o", str(court)], on_stage, "自动标定球场", log)
        if code != 0:
            rejected = court.with_suffix(".rejected.png")
            if rejected.exists():
                extra.append(rejected)
            raise StepFailedWithFiles(
                "自动标定球场失败：画面里找不到完整、清楚的球场线。请确认整片球场都在画面里，或者在 Mac 上手动标定后重新处理。",
                extra)
    if court.with_suffix(".png").exists():
        extra.append(court.with_suffix(".png"))
    args = ["analyze", str(work), "--court", str(court), "-o", str(out_dir)]
    if options.get("mode") != "doubles":
        args.append("--singles")
    if options.get("names"):
        args += ["--names", str(options["names"])]
    on_stage("骨架")
    if run_step(args, on_stage, "骨架", log) != 0:
        raise StepFailed("分析失败")
    files = sorted(p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in RESULT_SUFFIXES)
    return extra + files


def result_name(path: Path, stem: str) -> str:
    """Names on the web are the same for every recording: strip the recording's own file stem."""
    name = path.name
    for prefix in (stem + "_sdr.", stem + "."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120]


# ---------- the loop ----------

def process(api: Api, cfg: Config, video: dict, analyse_fn=analyse) -> None:
    vid = video["id"]
    job_dir = cfg.work_dir / vid
    job_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", video["filename"]) or "video.mov"
    local = job_dir / safe
    stage = {"text": "已领取"}
    stop = threading.Event()

    def report(text: str | None = None) -> None:
        if text:
            stage["text"] = text
        try:
            api.json("POST", f"/videos/{vid}/job", {"status": "processing", "stage": stage["text"]})
        except Exception as e:  # a missed heartbeat is not worth failing the job for
            print(f"  汇报进度失败：{e}")

    last = {"t": 0.0, "text": ""}

    def on_stage(text: str) -> None:
        stage["text"] = text
        now = time.time()
        if text != last["text"] and now - last["t"] >= 3:  # at most one request every few seconds
            last.update(t=now, text=text)
            report()

    def heartbeat() -> None:
        while not stop.wait(60):
            report()

    threading.Thread(target=heartbeat, daemon=True).start()
    log: list[str] = []
    try:
        if not local.exists():
            if video.get("originalDeleted"):
                raise StepFailed("云端的原片已经删除，这台 Mac 上也找不到，请重新上传")
            report("下载原片")
            print(f"  下载 {video['filename']}（{video['size'] / 1e6:.0f} MB）")
            api.download(video, local)
        if not video.get("originalDeleted"):
            api.retrying(lambda: api.json("POST", f"/videos/{vid}/original/release"), "删除云端原片")
            print("  云端原片已删除，本地保留在 " + str(local))
        out_dir = job_dir / "analysis"
        try:
            files = analyse_fn(local, out_dir, video.get("options") or {}, on_stage, log)
        except StepFailedWithFiles as e:
            for f in e.files:
                api.upload_result(vid, f, result_name(f, local.stem))
            raise
        web_dir = job_dir / "web"
        web_dir.mkdir(exist_ok=True)
        for i, f in enumerate(files, 1):
            name = result_name(f, local.stem)
            on_stage(f"上传结果 {i}/{len(files)}")
            if f.suffix.lower() == ".mp4":
                f = web_video(f, web_dir / name)
            print(f"  上传 {name}（{f.stat().st_size / 1e6:.1f} MB）")
            api.upload_result(vid, f, name)
        stop.set()
        api.retrying(lambda: api.json("POST", f"/videos/{vid}/job", {"status": "done", "stage": "完成"}), "汇报完成")
        print("  完成")
    except Exception as e:
        stop.set()
        detail = str(e) if isinstance(e, StepFailed) else f"{type(e).__name__}: {e}"
        tail = "\n".join(log[-12:])
        print(f"  失败：{detail}")
        try:
            api.json("POST", f"/videos/{vid}/job",
                     {"status": "failed", "stage": stage["text"], "error": (detail + "\n\n" + tail).strip()})
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
