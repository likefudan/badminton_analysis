"""The local dashboard: its state and its commands, with launchd and the cloud replaced."""

import json
import os
import urllib.error
import urllib.request

import pytest

from badminton import dashboard, metrics, node_status


class FakeService:
    NODE = "node"

    def __init__(self, tmp):
        self.tmp = tmp
        self.s = {"installed": False, "loaded": False, "pid": None, "lastExit": None}
        self.calls = []

    def state(self, label):
        return dict(self.s)

    def log_path(self, label):
        return self.tmp / "node.log"

    def install(self, label, args):
        self.calls.append(("install", args))
        self.s.update(installed=True, loaded=True, pid=os.getpid())

    def start(self, label):
        self.calls.append(("start",))
        self.s.update(loaded=True, pid=os.getpid())

    def stop(self, label):
        self.calls.append(("stop",))
        self.s.update(loaded=False, pid=None)


@pytest.fixture
def dash(tmp_path):
    d = dashboard.Dashboard(cfg_fn=lambda: None, work_dir=tmp_path, sampler=metrics.Sampler(), svc=FakeService(tmp_path))
    d.tasks = dashboard.Cached(lambda: [], 60, [])
    d.system = dashboard.Cached(lambda: {"diskFreeGb": 100.0}, 60, {})
    d.service = dashboard.Cached(lambda: d.svc.state("node"), 0, {})
    return d


def wait(d):
    for _ in range(200):
        if not d.busy:
            return
        import time
        time.sleep(0.01)
    raise AssertionError("command did not finish")


def test_state_of_a_stopped_node_and_of_one_that_works(dash, tmp_path):
    s = dash.state()
    assert s["node"]["running"] is False and s["node"]["installed"] is False
    assert "口令" in s["cloud"]["error"]
    dash.svc.s.update(installed=True, loaded=True, pid=os.getpid())
    (tmp_path / node_status.STATE_FILE).write_text(json.dumps({
        "pid": os.getpid(), "name": "mac", "job": {"task": "process", "name": "a.mov"}, "stage": "骨架 40%",
        "progress": {"steps": [], "elapsedSeconds": 12, "etaSeconds": 30}}))
    (tmp_path / "node.log").write_text("12:00:00 开始处理 a.mov\n  | token=abc123\n")
    n = dash.state()
    assert n["node"]["running"] and n["node"]["managed"] and n["node"]["stage"] == "骨架 40%"
    assert n["node"]["job"]["name"] == "a.mov"
    dash.log.expire()
    assert dash.log.get()["lines"] == ["12:00:00 开始处理 a.mov", "  | token=***"]


def test_a_state_file_left_by_a_dead_node_does_not_count(dash, tmp_path):
    (tmp_path / node_status.STATE_FILE).write_text(json.dumps({"pid": 2**22 + 12345, "job": {"task": "process"}}))
    n = dash.state()["node"]
    assert n["running"] is False and "job" not in n


def test_first_start_installs_the_service_and_stop_unloads_it(dash):
    dash.command("start", {})
    wait(dash)
    assert dash.svc.calls == [("install", ["node"])]
    assert dash.state()["node"]["running"]
    dash.command("stop", {})
    wait(dash)
    assert dash.svc.calls[-1] == ("stop",) and not dash.state()["node"]["running"]
    dash.command("start", {})
    wait(dash)
    assert dash.svc.calls[-1] == ("start",)


def test_stop_after_is_a_file_the_node_looks_for(dash, tmp_path):
    with pytest.raises(RuntimeError, match="没有在运行"):
        dash.command("stop-after", {})
    dash.svc.s.update(installed=True, loaded=True, pid=os.getpid())
    dash.state()
    dash.command("stop-after", {})
    assert (tmp_path / "stop_after_job").exists() and dash.state()["node"]["stopAfter"] is True
    dash.command("cancel-stop-after", {})
    assert not (tmp_path / "stop_after_job").exists()
    with pytest.raises(RuntimeError):
        dash.command("format-disk", {})


def test_a_failed_command_is_shown_not_raised(dash):
    dash.svc.install = lambda label, args: (_ for _ in ()).throw(RuntimeError("launchctl 说不行"))
    dash.command("start", {})
    wait(dash)
    assert "启动失败：launchctl 说不行" in dash.state()["error"]


def test_only_this_machine_and_only_the_page_may_command(dash):
    server = dashboard.make_server(dash, 0)
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path, headers=None, data=None):
        req = urllib.request.Request(base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        assert call("/")[0] == 200 and "处理节点面板".encode() in call("/")[1]
        assert json.loads(call("/api/state")[1])["node"]["running"] is False
        assert call("/api/state", {"Host": "evil.example"})[0] == 403                 # DNS rebinding
        assert call("/api/control/start", data=b"{}")[0] == 403                        # a form on another site
        assert not dash.svc.calls
        assert call("/api/control/cancel-stop-after", {"x-ba-dashboard": "1"}, b"{}")[0] == 200
        assert call("/api/control/nonsense", {"x-ba-dashboard": "1"}, b"{}")[0] == 400
    finally:
        server.shutdown()
