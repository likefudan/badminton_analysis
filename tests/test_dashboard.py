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
    d.stray = dashboard.Cached(lambda: None, 60, None)
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


def test_a_node_started_by_hand_from_an_older_version_is_found_by_its_command(dash):
    from types import SimpleNamespace

    ps = "  501 /sbin/launchd\n99765 /Users/kel/codebase/badminton_analysis/.venv/bin/python /Users/kel/x/.venv/bin/badminton node\n" \
         " 1677 /x/python -m badminton.cli shuttle /Users/kel/node/IMG_1191.mov\n 3319 /x/python -m badminton.cli dashboard --port 8787\n"
    assert dashboard.find_node(lambda *a, **k: SimpleNamespace(stdout=ps)) == 99765
    assert dashboard.find_node(lambda *a, **k: SimpleNamespace(stdout="77 python -m badminton.cli node --work-dir /tmp/w --once\n")) == 77
    assert dashboard.find_node(lambda *a, **k: SimpleNamespace(stdout="78 python -m badminton.cli node install\n")) is None
    dash.stray = dashboard.Cached(lambda: 99765, 60, None)
    n = dash.state()["node"]
    assert n["running"] and n["pid"] == 99765 and not n["managed"] and n["starting"]
    dash.command("start", {})                     # must not start a second node next to it
    wait(dash)
    assert dash.svc.calls == []


class FakeSite:
    def __init__(self, commands=()):
        self.commands = list(commands)
        self.posts = []
        self.down = False

    def json(self, method, path, data=None, timeout=0):
        if self.down:
            raise ConnectionError("offline")
        self.posts.append(data)
        out, self.commands = self.commands, []
        return {"commands": out}


def test_commands_from_the_phone_are_the_pages_buttons_and_nothing_else(dash, tmp_path):
    dash.svc.s.update(installed=True, loaded=True, pid=os.getpid())
    site = FakeSite([{"id": 7, "action": "stop-after"}, {"id": 8, "action": "retry"}, {"id": 9, "action": "rm -rf"}])
    remote = dashboard.Remote(dash, api=site, name="mac")
    assert remote.sync() == 3
    assert (tmp_path / "stop_after_job").exists()
    state = site.posts[0]["state"]
    assert site.posts[0]["node"] == "mac" and state["node"]["running"] and "history" in state["metrics"]
    site.down = True                                   # the answer must survive a failed request
    with pytest.raises(ConnectionError):
        remote.sync()
    site.down = False
    remote.sync()
    results = {r["id"]: r for r in site.posts[-1]["results"]}
    assert results[7]["ok"] is True
    assert results[8]["ok"] is False and results[9]["ok"] is False
    assert site.posts[-1]["state"]["node"]["stopAfter"] is True
    remote.sync()
    assert site.posts[-1]["results"] == []


def test_a_stopped_node_can_be_started_from_the_phone(dash):
    site = FakeSite([{"id": 1, "action": "start"}])
    remote = dashboard.Remote(dash, api=site, name="mac")
    remote.sync()
    wait(dash)
    assert dash.svc.calls == [("install", ["node"])]
    remote.sync()
    assert site.posts[-1]["state"]["node"]["running"] is True


def test_curves_are_thinned_for_the_phone_and_end_with_the_newest():
    h = {"t": list(range(300)), "cpu": list(range(300))}
    out = dashboard.thin(h, 60)
    assert len(out["t"]) == 60 and out["t"][-1] == 299 and out["cpu"] == out["t"]
    assert dashboard.thin({"t": [1, 2], "cpu": [5, 6]}) == {"t": [1, 2], "cpu": [5, 6]}
    assert dashboard.thin({"t": [], "cpu": []}) == {"t": [], "cpu": []}
