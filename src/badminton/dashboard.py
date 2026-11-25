"""The dashboard of the processing node: a web page served to this Mac only (http://localhost:8787).

It shows whether the node runs and what it is doing (read from the file the node writes next to its
work, so every second and without the cloud), the queue (asked from the cloud every few seconds),
how busy the machine is, background jobs and the node's log. It starts and stops the node through
launchd (service.py).

Only the standard library is used. The server listens on 127.0.0.1 and refuses requests whose Host
is not this machine, and commands that do not carry the page's own header, so that another web page
open in the browser cannot press the buttons.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import metrics, node, node_status, service

PORT = 8787
HEADER = "x-ba-dashboard"
LOG_LINES = 200
REPO = service.REPO


class Cached:
    """A value that is expensive to get: asked again at most every ``seconds``, by one caller at a time."""

    def __init__(self, fn, seconds: float, empty=None):
        self.fn = fn
        self.seconds = seconds
        self.value = empty
        self.at = 0.0
        self._lock = threading.Lock()

    def get(self):
        if time.time() - self.at >= self.seconds and self._lock.acquire(blocking=False):
            try:
                self.value = self.fn()
            except Exception as e:  # shown on the page, never fatal
                self.value = {"error": f"{type(e).__name__}: {e}"[:300]}
            finally:
                self.at = time.time()
                self._lock.release()
        return self.value

    def expire(self) -> None:
        self.at = 0.0


def alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


_NODE_CMD = re.compile(r"badminton(\.cli)?\s+node(\s+(--\S+(\s+[^-]\S*)?))*\s*$")


def find_node(run=subprocess.run) -> int | None:
    """A node that runs without a state file: one started by hand from a version that did not write it yet."""
    try:
        out = run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if _NODE_CMD.search(cmd) and pid.isdigit() and int(pid) != os.getpid():
            return int(pid)
    return None


def is_node(pid: int, run=subprocess.run) -> bool:
    """A pid from an old state file may belong to something else by now."""
    try:
        cmd = run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "badminton" in cmd and " node" in cmd


class Dashboard:
    def __init__(self, cfg_fn=node.Config.load, work_dir: Path | None = None, sampler: metrics.Sampler | None = None,
                 svc=service, label: str = service.NODE):
        self.cfg_fn = cfg_fn
        self.svc = svc
        self.label = label
        self.sampler = sampler or metrics.Sampler()
        self._work_dir = work_dir
        self.busy = ""                      # the command that is being carried out, shown on the page
        self.last_error = ""
        self._lock = threading.Lock()
        self.service = Cached(lambda: self.svc.state(self.label), 1.5, {})
        self.cloud = Cached(self._cloud, 8.0, {})
        self.tasks = Cached(self._tasks, 10.0, [])
        self.system = Cached(lambda: node_status.system_info(self.work_dir()), 10.0, {})
        self.log = Cached(self._log, 1.0, {"lines": [], "file": ""})
        self.stray = Cached(find_node, 3.0, None)
        self._node_pid: int | None = None

    # -- where things are --

    def cfg(self):
        try:
            return self.cfg_fn()
        except SystemExit:   # no token yet: the page still shows the machine and the service
            return None

    def work_dir(self) -> Path:
        if self._work_dir is None:
            cfg = self.cfg()
            self._work_dir = cfg.work_dir if cfg else Path.home() / "badminton_uploads"
        return self._work_dir

    def stop_after_file(self) -> Path:
        return self.work_dir() / "stop_after_job"

    # -- the pieces of the state --

    def node_state(self) -> dict:
        try:
            local = json.loads((self.work_dir() / node_status.STATE_FILE).read_text())
        except (OSError, ValueError):
            local = {}
        svc = self.service.get() or {}
        pid = svc.get("pid")
        if not pid and alive(local.get("pid")) and not local.get("stopped") and is_node(local["pid"]):
            pid = local["pid"]              # started by hand in a terminal, not by launchd
        if not pid:
            found = self.stray.get()
            pid = found if isinstance(found, int) else None
        running = bool(pid)
        self._node_pid = pid if running else None
        out = {"running": running, "pid": pid, "managed": bool(svc.get("pid")), "installed": bool(svc.get("installed")),
               "loaded": bool(svc.get("loaded")), "lastExit": svc.get("lastExit"),
               "stopAfter": running and self.stop_after_file().exists()}
        if running and local.get("pid") == pid:
            out.update({k: local.get(k) for k in ("name", "startedAt", "version", "job", "stage", "progress", "at")})
        elif running:
            out["starting"] = True          # launchd started it, it has not written its state yet
        return out

    def _cloud(self) -> dict:
        cfg = self.cfg()
        if not cfg:
            return {"error": f"没有登录口令（{node.CONFIG_PATH}），看不到队列"}
        d = node.Api(cfg).json("GET", "/status", timeout=20)
        return {"url": cfg.url, "active": d.get("active", []), "queued": d.get("queued", []), "failed": d.get("failed", []),
                "now": d.get("now")}

    def _tasks(self) -> list[dict]:
        mine = {os.getpid(), self._node_pid}
        out = []
        for t in node_status.find_tasks([REPO], self._node_pid or 0):
            if t["pid"] in mine:
                continue
            lines = node_status.read_tail(t["log"]) if t["log"] else []
            item = {"pid": t["pid"], "command": t["command"], "ageSeconds": int(t["age"]), "cpu": t["cpu"],
                    "rssMb": t["rssMb"], "logFile": t["log"].replace(str(Path.home()), "~"),
                    "log": [node_status.clean(x) for x in lines[-node_status.TAIL_LINES:]]}
            progress = node_status.read_progress(lines)
            if progress:
                item["progress"] = progress
            out.append(item)
        return out

    def _log(self) -> dict:
        path = str(self.svc.log_path(self.label))
        pid = self._node_pid
        svc = self.service.value or {}
        if pid and pid != svc.get("pid"):   # started by hand: its output may go to another file, or to a terminal
            path = node_status.open_files([pid]).get(pid, {}).get("out", "")
        lines = node_status.read_tail(path) if path else []
        return {"file": path.replace(str(Path.home()), "~"), "lines": [node_status.clean(x) for x in lines[-LOG_LINES:]]}

    def state(self) -> dict:
        n = self.node_state()
        return {"now": node_status._iso(time.time()), "node": n, "busy": self.busy, "error": self.last_error,
                "metrics": self.sampler.to_json(), "system": self.system.get(), "cloud": self.cloud.get(),
                "tasks": self.tasks.get(), "log": self.log.get(), "repo": str(REPO).replace(str(Path.home()), "~")}

    # -- commands --

    def _do(self, label: str, fn) -> None:
        """Run a command in the background (stopping can take half a minute); the page shows ``busy`` meanwhile."""
        with self._lock:
            if self.busy:
                raise RuntimeError(f"正在{self.busy}，稍等")
            self.busy = label
            self.last_error = ""

        def work() -> None:
            try:
                fn()
            except Exception as e:
                self.last_error = f"{label}失败：{e}"
            finally:
                self.busy = ""
                self.service.expire()
                self.stray.expire()
                self.cloud.expire()

        threading.Thread(target=work, daemon=True).start()

    def _start(self) -> None:
        self.stop_after_file().unlink(missing_ok=True)
        if self._node_pid:
            return
        if not self.svc.state(self.label)["installed"]:
            self.svc.install(self.label, ["node"])   # the first start from the page sets the service up
        else:
            self.svc.start(self.label)

    def _stop(self) -> None:
        pid = self._node_pid
        self.stop_after_file().unlink(missing_ok=True)
        if self.svc.state(self.label)["loaded"]:
            self.svc.stop(self.label)
        elif pid:                            # a node started by hand
            os.kill(pid, signal.SIGTERM)
            for _ in range(service.EXIT_TIMEOUT * 2):
                if not alive(pid):
                    return
                time.sleep(0.5)
            raise RuntimeError(f"进程 {pid} 在 {service.EXIT_TIMEOUT} 秒内没有退出")

    def _restart(self) -> None:
        self._stop()
        self._start()

    def _update(self) -> None:
        res = subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            raise RuntimeError("git pull 没成功，节点没有动：" + (res.stderr or res.stdout).strip()[-400:])
        self._restart()
        # the dashboard runs the old code too: when launchd looks after it, leave and be started again
        if self.svc.state(service.DASHBOARD)["pid"] == os.getpid():
            threading.Timer(1.0, lambda: os._exit(1)).start()

    def command(self, action: str, body: dict) -> dict:
        if action == "start":
            self._do("启动", self._start)
        elif action == "stop":
            self._do("停止", self._stop)
        elif action == "restart":
            self._do("重启", self._restart)
        elif action == "update":
            self._do("更新代码并重启", self._update)
        elif action == "stop-after":
            if not self._node_pid:
                raise RuntimeError("处理程序没有在运行")
            self.stop_after_file().parent.mkdir(parents=True, exist_ok=True)
            self.stop_after_file().write_text(node_status._iso(time.time()))
        elif action == "cancel-stop-after":
            self.stop_after_file().unlink(missing_ok=True)
        elif action == "retry":
            vid = str(body.get("videoId", ""))
            cfg = self.cfg()
            if not cfg or len(vid) != 36:
                raise RuntimeError("不知道要重试哪一个")
            node.Api(cfg).json("POST", f"/videos/{vid}/job/retry", {}, timeout=30)
            self.cloud.expire()
        else:
            raise RuntimeError("不认识的命令")
        return {"ok": True}


REMOTE_INTERVAL = 10.0
REMOTE_ACTIONS = ("start", "stop", "stop-after", "cancel-stop-after", "restart", "update")
REMOTE_POINTS = 60      # the ten minutes of curves, thinned out for the phone


def thin(history: dict, points: int = REMOTE_POINTS) -> dict:
    """Every n-th sample, always ending with the newest."""
    n = len(history.get("t", []))
    step = max(1, -(-n // points))
    keep = list(range(n - 1, -1, -step))[::-1]
    return {k: [v[i] for i in keep] for k, v in history.items()}


class Remote:
    """The phone's way in: every ten seconds the dashboard tells the site what it sees and asks for commands
    (the buttons of the 「Mac 状态」 page). The Mac only ever calls out; nothing listens for the internet.
    It runs here and not in the node, so that a stopped node can be started from the phone."""

    def __init__(self, dash: Dashboard, api=None, name: str = "", interval: float = REMOTE_INTERVAL):
        self.dash = dash
        self.api = api
        self.name = name
        self.interval = interval
        self.results: list[dict] = []
        self.last_error = ""
        self._stop = threading.Event()

    def state(self) -> dict:
        d = self.dash
        m = d.sampler.to_json()
        return {"at": node_status._iso(time.time()), "node": d.node_state(), "busy": d.busy, "error": d.last_error,
                "metrics": {"source": m["source"], "note": m["note"], "now": m["now"], "history": thin(m["history"])},
                "system": d.system.get()}

    def sync(self) -> int:
        """One exchange with the site. Returns how many commands were carried out."""
        sent, self.results = self.results, []
        try:
            got = self.api.json("POST", "/node/agent", {"node": self.name, "state": self.state(), "results": sent}, timeout=20)
        except Exception:
            self.results = sent + self.results   # the outcome of a command is not lost with one failed request
            raise
        commands = got.get("commands") or []
        for c in commands:
            action = str(c.get("action", ""))
            try:
                if action not in REMOTE_ACTIONS:   # never anything but the page's own buttons
                    raise RuntimeError("不认识的指令")
                self.dash.command(action, {})
                self.results.append({"id": c.get("id"), "ok": True, "message": "Mac 已收到，开始执行"})
            except Exception as e:
                self.results.append({"id": c.get("id"), "ok": False, "message": str(e)[:300]})
        return len(commands)

    def _loop(self) -> None:
        while not self._stop.is_set():
            wait = self.interval
            try:
                if self.api is None:
                    cfg = self.dash.cfg()
                    if cfg is None:          # no token yet: look again in a minute
                        self._stop.wait(60)
                        continue
                    self.api, self.name = node.Api(cfg), cfg.name
                if self.sync():
                    wait = 1.0               # say at once that the command arrived, and what it changed
                self.last_error = ""
            except Exception as e:           # the site is unreachable, or older than this: keep trying quietly
                if str(e) != self.last_error:
                    print(f"和网站同步面板状态失败：{e}")
                self.last_error = str(e)
            self._stop.wait(wait)

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="remote").start()

    def stop(self) -> None:
        self._stop.set()


def make_server(dash: Dashboard, port: int = PORT) -> ThreadingHTTPServer:
    page = (Path(__file__).with_name("dashboard.html")).read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def reply(self, body: bytes, kind: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("content-type", kind)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, status: int = 200) -> None:
            self.reply(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def local(self) -> bool:
            host = (self.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
            return host in ("localhost", "127.0.0.1", "::1")

        def do_GET(self):
            if not self.local():
                return self.json({"error": "只能在这台 Mac 上打开"}, 403)
            if self.path in ("/", "/index.html"):
                return self.reply(page, "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self.json(dash.state())
            self.json({"error": "没有这个地址"}, 404)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("content-length") or 0))
            if not self.local() or self.headers.get(HEADER) != "1":
                return self.json({"error": "只接受面板自己发的命令"}, 403)
            if not self.path.startswith("/api/control/"):
                return self.json({"error": "没有这个地址"}, 404)
            try:
                body = json.loads(raw or b"{}")
                self.json(dash.command(self.path.rsplit("/", 1)[1], body if isinstance(body, dict) else {}))
            except Exception as e:
                self.json({"error": str(e)[:400]}, 400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def run(port: int = PORT, open_browser: bool = False) -> int:
    dash = Dashboard()
    try:
        server = make_server(dash, port)
    except OSError as e:
        raise SystemExit(f"端口 {port} 用不了（{e}）。面板可能已经在运行：打开 http://localhost:{port}") from None
    dash.sampler.start()
    remote = Remote(dash)
    remote.start()
    url = f"http://localhost:{port}"
    print(f"面板在 {url}，只有这台 Mac 能打开。Ctrl+C 退出。")
    if open_browser:
        subprocess.run(["open", url], check=False)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        remote.stop()
        dash.sampler.stop()
    return 0
