"""What this Mac is doing, reported to the cloud every half minute so the web page can show it.

The node knows its own job. Everything else that runs here in the background (a
fine-tuning run started by hand, a long ``badminton`` command) is found without
being registered: Python processes whose working directory is inside a watched
directory (this repository by default), with the file their output goes to. The
last lines of that file are sent along, and a progress is read from them where
they look like "Epoch [3 / 10]" or a tqdm bar.

Only the standard library and the ``ps``, ``lsof``, ``vm_stat`` commands of macOS
are used. Every probe is optional: whatever fails is left out of the report.
"""

from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

INTERVAL = 30.0
LOG_LINES = 40                 # lines of the node's own log in a report
TAIL_LINES = 25                # lines of a background task's log
TAIL_BYTES = 256 * 1024        # how far back a log is read (progress bars fill it quickly)
LINE_CHARS = 300
MIN_AGE = 20.0                 # seconds a process must have lived to be listed (not every short command)
KEEP_ENDED = 24 * 3600.0       # ended tasks stay in the report this long
MAX_TASKS = 12
REPO = Path(__file__).resolve().parents[2]

# the node's own recent output: what it printed, and the output of the step it is running
LOG: collections.deque[str] = collections.deque(maxlen=200)

_LOCK = threading.Lock()       # LOG is written by the job's threads and read by the reporter's

# values that look like credentials are not sent: "token=…", "Authorization: Bearer …", "--api-key …"
_SECRET = re.compile(
    r"(?i)((?:token|secret|password|passwd|api[-_]?key|authorization)\w*[\"']?\s*[=:]\s*[\"']?(?:bearer\s+)?"
    r"|--?[\w-]*(?:token|secret|password|passwd|api[-_]?key)[\w-]*\s+"
    r"|\bbearer\s+)([^\s\"']+)")


def clean(line: str) -> str:
    return _SECRET.sub(r"\1***", line.rstrip())[:LINE_CHARS]


def note(line: str) -> None:
    line = line.rstrip()
    if line.strip():
        with _LOCK:
            LOG.append(time.strftime("%H:%M:%S ") + clean(line))


def recent(n: int = LOG_LINES) -> list[str]:
    with _LOCK:
        return list(LOG)[-n:]


class Tee:
    """Stands in for sys.stdout: writes through, and keeps the finished lines in LOG."""

    def __init__(self, stream):
        self.stream = stream
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            note(line)
        return self.stream.write(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)


# ---------- reading a log ----------

def screen_lines(text: str) -> list[str]:
    """The lines as a terminal would show them: a progress bar redrawn with \\r is one line, its last state."""
    out = []
    for line in text.split("\n"):
        parts = [p for p in line.split("\r") if p.strip()]
        if parts:
            out.append(parts[-1].rstrip())
    return out


def read_tail(path: str, size: int = TAIL_BYTES) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            f.seek(max(0, end - size))
            data = f.read()
    except OSError:
        return []
    lines = screen_lines(data.decode("utf-8", "replace"))
    return lines[1:] if end > size and len(lines) > 1 else lines  # the first one is cut somewhere in the middle


_EPOCH = re.compile(r"(?i)epoch\W{0,3}(\d+)\s*(?:/|of)\s*(\d+)")
_BAR = re.compile(r"(\d{1,3})%\|.*?\|\s*(\d+)/(\d+)\s*\[[^<\]]*<([\d:]+|\?)")
_FRAMES = re.compile(r"(\d+)/(\d+) 帧")
_ERROR = re.compile(r"Traceback \(most recent call last\)|^\w*(Error|Exception): |Killed|out of memory", re.M)


def _clock(text: str) -> float | None:
    try:
        seconds = 0.0
        for part in text.split(":"):
            seconds = seconds * 60 + int(part)
        return seconds
    except ValueError:
        return None


def read_progress(lines: list[str]) -> dict:
    """{"label", "percent", "etaSeconds"} from the last lines of a log; empty when nothing is recognised."""
    epoch = bar = frames = None
    epoch_at = bar_at = -1
    for i, line in enumerate(lines):
        m = _EPOCH.search(line)
        if m and 0 < int(m.group(2)) and int(m.group(1)) <= int(m.group(2)):
            epoch, epoch_at = (int(m.group(1)), int(m.group(2))), i
        m = _BAR.search(line)
        if m and int(m.group(3)):
            bar, bar_at = (int(m.group(2)), int(m.group(3)), m.group(4), line.split(":")[0].strip() if ":" in line.split("%|")[0] else ""), i
        m = _FRAMES.search(line)
        if m and int(m.group(2)):
            frames = (int(m.group(1)), int(m.group(2)))
    out: dict = {}
    if epoch:
        n, total = epoch
        out["label"] = f"第 {n}/{total} 轮"
        inside = 0.0
        if bar and bar_at > epoch_at:  # a bar after the epoch line is the progress inside this epoch
            inside = min(1.0, bar[0] / bar[1])
            out["label"] += f"，{bar[3] + ' ' if bar[3] else ''}{bar[0]}/{bar[1]}"
        # "Epoch [3 / 10]" is printed when epoch 3 starts in some trainers and when it ends in others:
        # without knowing which, count the epochs before it as done
        out["percent"] = round(min(100.0, (n - 1 + inside) * 100 / total), 1)
    elif bar:
        out["label"] = f"{bar[3] + ' ' if bar[3] else ''}{bar[0]}/{bar[1]}"
        out["percent"] = round(min(100.0, bar[0] * 100 / bar[1]), 1)
        left = _clock(bar[2])
        if left is not None:
            out["etaSeconds"] = left
    elif frames:
        out["label"] = f"{frames[0]}/{frames[1]} 帧"
        out["percent"] = round(min(100.0, frames[0] * 100 / frames[1]), 1)
    return out


# ---------- processes ----------

def _etime(text: str) -> float:
    """ps etime, [[dd-]hh:]mm:ss, in seconds."""
    days, _, rest = text.rpartition("-")
    return (int(days) if days else 0) * 86400 + (_clock(rest) or 0.0)


def list_processes(run=subprocess.run) -> list[dict]:
    res = run(["ps", "-axo", "pid=,ppid=,etime=,pcpu=,rss=,command="], capture_output=True, text=True, timeout=20)
    out = []
    for line in res.stdout.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        try:
            out.append({"pid": int(parts[0]), "ppid": int(parts[1]), "age": _etime(parts[2]), "cpu": float(parts[3]),
                        "rssMb": int(parts[4]) // 1024, "command": parts[5]})
        except ValueError:
            continue
    return out


def open_files(pids: list[int], run=subprocess.run) -> dict[int, dict]:
    """{pid: {"cwd": path, "out": file its standard output goes to, if that is a file}}."""
    if not pids:
        return {}
    res = run(["lsof", "-a", "-p", ",".join(map(str, pids)), "-d", "cwd,1", "-F", "pftn"],
              capture_output=True, text=True, timeout=20)
    out: dict[int, dict] = {}
    pid, fd, kind = None, "", ""
    for line in res.stdout.splitlines():
        tag, value = line[:1], line[1:]
        if tag == "p":
            pid = int(value)
            out[pid] = {}
        elif tag == "f":
            fd, kind = value, ""
        elif tag == "t":
            kind = value
        elif tag == "n" and pid is not None:
            if fd == "cwd":
                out[pid]["cwd"] = value
            elif fd == "1" and kind == "REG":
                out[pid]["out"] = value
    return out


_PYTHON = re.compile(r"(?i)(^|/)python[\d.]*(\s|$)|\.app/Contents/MacOS/Python\s|(^|/)badminton(\s|$)")
_WORKER = re.compile(r"multiprocessing\.(spawn|resource_tracker|forkserver)|joblib|loky")


def short_command(command: str) -> str:
    """The command without the interpreter's path: 'scripts/tracknet_finetune.py --epochs 10 …'."""
    words = command.split()
    if words and _PYTHON.search(words[0] + " "):
        words = words[1:]
    if words:
        words[0] = re.sub(r"^.*/\.venv/bin/", "", words[0])
    home = str(Path.home())
    return clean(" ".join(words).replace(home, "~")) or clean(command)


def find_tasks(watch: list[Path], own_pid: int, processes: list[dict] | None = None, files=None) -> list[dict]:
    """Background Python jobs in the watched directories, other than the node and what it started itself."""
    processes = list_processes() if processes is None else processes
    parent = {p["pid"]: p["ppid"] for p in processes}

    def descends(pid: int, ancestor: int) -> bool:
        for _ in range(40):
            pid = parent.get(pid, 0)
            if pid == ancestor:
                return True
            if pid <= 1:
                return False
        return False

    wanted = [p for p in processes
              if _PYTHON.search(p["command"]) and not _WORKER.search(p["command"]) and p["age"] >= MIN_AGE
              and p["pid"] != own_pid and not descends(p["pid"], own_pid)]
    info = (files or open_files)([p["pid"] for p in wanted])
    roots = [str(w) for w in watch]
    kept = []
    for p in wanted:
        cwd = info.get(p["pid"], {}).get("cwd", "")
        if not any(cwd == r or cwd.startswith(r + "/") or r + "/" in p["command"] for r in roots):
            continue
        kept.append({**p, "cwd": cwd, "log": info.get(p["pid"], {}).get("out", "")})
    ids = {p["pid"] for p in kept}
    tasks = []
    for p in kept:
        if any(descends(p["pid"], other) for other in ids if other != p["pid"]):
            continue  # a helper of another task (data loader, a step of a script): shown with its parent
        family = [q for q in processes if q["pid"] == p["pid"] or descends(q["pid"], p["pid"])]
        tasks.append({"pid": p["pid"], "command": short_command(p["command"]), "cwd": p["cwd"], "log": p["log"],
                      "age": p["age"], "cpu": round(sum(q["cpu"] for q in family), 1),
                      "rssMb": sum(q["rssMb"] for q in family), "processes": len(family)})
    return sorted(tasks, key=lambda t: -t["age"])


# ---------- the machine ----------

def system_info(work_dir: Path, run=subprocess.run) -> dict:
    out: dict = {}
    try:
        out["load"] = [round(x, 2) for x in os.getloadavg()]
        out["cpus"] = os.cpu_count()
    except OSError:
        pass
    try:
        disk = shutil.disk_usage(work_dir)
        out["diskFreeGb"] = round(disk.free / 1e9, 1)
        out["diskTotalGb"] = round(disk.total / 1e9, 1)
    except OSError:
        pass
    try:
        total = int(run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=10).stdout)
        text = run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        page = int(re.search(r"page size of (\d+) bytes", text).group(1))
        pages = {k.strip(): int(v) for k, v in re.findall(r"^(.+?):\s+(\d+)\.", text, re.M)}
        used = (pages.get("Pages active", 0) + pages.get("Pages wired down", 0)
                + pages.get("Pages occupied by compressor", 0)) * page
        out["memTotalGb"] = round(total / 2**30, 1)
        out["memUsedGb"] = round(used / 2**30, 1)
    except Exception:
        pass
    return out


def code_version(repo: Path = REPO) -> str:
    try:
        res = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%h %cs %s"], capture_output=True, text=True,
                             timeout=10)
        return res.stdout.strip()[:120]
    except Exception:
        return ""


# ---------- the reporter ----------

def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


class Reporter:
    """Collects the status and posts it; ``job`` is set by the node's loop around each job."""

    def __init__(self, send, name: str, work_dir: Path, watch: list[Path] | None = None, clock=time.time,
                 tasks_fn=None, system_fn=system_info):
        self.send = send
        self.name = name
        self.work_dir = work_dir
        self.watch = [Path(w).expanduser().resolve() for w in (watch or [REPO])]
        self.clock = clock
        self.tasks_fn = tasks_fn or (lambda: find_tasks(self.watch, os.getpid()))
        self.system_fn = system_fn
        self.started = clock()
        self.version = code_version()
        self.job: dict | None = None
        self.last_error = ""
        self._seen_path = work_dir / "tasks_seen.json"
        self._seen: dict[str, dict] = {}
        self._stop = threading.Event()
        try:
            self._seen = json.loads(self._seen_path.read_text())
        except (OSError, ValueError):
            pass

    def set_job(self, task: str | None, video: dict | None = None) -> None:
        self.job = None if task is None else {
            "task": task, "videoId": (video or {}).get("id", ""),
            "name": (video or {}).get("title") or (video or {}).get("filename", ""), "since": _iso(self.clock())}

    def tasks(self) -> list[dict]:
        now = self.clock()
        running = {}
        for t in self.tasks_fn():
            started = now - t["age"]
            # a pid can be used again after a restart of the Mac: the start time tells them apart
            key = next((k for k, s in self._seen.items() if s["pid"] == t["pid"] and not s.get("endedAt")
                        and abs(s["started"] - started) < 120), f"{t['pid']}-{int(started)}")
            running[key] = {**self._seen.get(key, {}), "pid": t["pid"], "command": t["command"], "cwd": t["cwd"],
                            "log": t["log"] or self._seen.get(key, {}).get("log", ""),
                            "started": self._seen.get(key, {}).get("started", started),
                            "cpu": t["cpu"], "rssMb": t["rssMb"], "processes": t["processes"]}
        for key, s in self._seen.items():
            if key not in running and not s.get("endedAt"):
                s["endedAt"] = now
        self._seen.update(running)
        self._seen = {k: s for k, s in self._seen.items() if not s.get("endedAt") or now - s["endedAt"] < KEEP_ENDED}
        try:
            self._seen_path.parent.mkdir(parents=True, exist_ok=True)
            self._seen_path.write_text(json.dumps(self._seen))
        except OSError:
            pass
        out = []
        for key, s in sorted(self._seen.items(), key=lambda kv: (bool(kv[1].get("endedAt")), -kv[1]["started"])):
            lines = read_tail(s["log"]) if s.get("log") else []
            ended = s.get("endedAt")
            item = {"id": key, "pid": s["pid"], "command": s["command"], "cwd": s["cwd"].replace(str(Path.home()), "~"),
                    "logFile": s.get("log", "").replace(str(Path.home()), "~"), "startedAt": _iso(s["started"]),
                    "running": not ended, "log": [clean(x) for x in lines[-TAIL_LINES:]]}
            if ended:
                item["endedAt"] = _iso(ended)
                item["failed"] = bool(_ERROR.search("\n".join(lines[-TAIL_LINES:])))
            else:
                item.update(cpu=s["cpu"], rssMb=s["rssMb"], processes=s["processes"])
            progress = read_progress(lines)
            if progress:
                item["progress"] = progress
            out.append(item)
        return out[:MAX_TASKS]

    def collect(self) -> dict:
        status: dict = {"name": self.name, "at": _iso(self.clock()), "startedAt": _iso(self.started),
                        "version": self.version, "job": self.job, "log": recent(),
                        "watch": [str(w).replace(str(Path.home()), "~") for w in self.watch]}
        for key, fn in (("system", lambda: self.system_fn(self.work_dir)), ("tasks", self.tasks)):
            try:
                status[key] = fn()
            except Exception as e:  # a probe that breaks must not stop the others, or the node
                status[key + "Error"] = f"{type(e).__name__}: {e}"[:200]
        return status

    def report(self) -> None:
        try:
            self.send(self.collect())
            self.last_error = ""
        except Exception as e:
            if str(e) != self.last_error:  # say it once, not every half minute
                print(f"上报 Mac 状态失败：{e}")
            self.last_error = str(e)

    def start(self, interval: float = INTERVAL) -> None:
        def loop() -> None:
            while True:
                self.report()
                if self._stop.wait(interval):
                    return

        threading.Thread(target=loop, daemon=True, name="status").start()

    def stop(self) -> None:
        self._stop.set()
