"""The launchd services, against a stand-in for launchctl."""

import plistlib
from types import SimpleNamespace

import pytest

from badminton import service


class Launchctl:
    def __init__(self):
        self.loaded = False
        self.pid = None
        self.calls: list[list[str]] = []

    def __call__(self, args, **kw):
        self.calls.append(args)
        verb = args[1]
        if verb == "print":
            if not self.loaded:
                return SimpleNamespace(returncode=113, stdout="", stderr="Could not find service")
            text = "\tstate = running\n\tpid = %d\n" % self.pid if self.pid else "\tstate = not running\n\tlast exit code = 0\n"
            return SimpleNamespace(returncode=0, stdout=text, stderr="")
        if verb == "bootstrap":
            self.loaded, self.pid = True, 4242
        elif verb == "bootout":
            self.loaded, self.pid = False, None
        elif verb == "kickstart":
            self.pid = 4243
        return SimpleNamespace(returncode=0, stdout="", stderr="")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(service.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(service, "_alive", lambda pid: False)   # the fake pids are not real processes
    return tmp_path


def test_stop_waits_until_the_process_is_gone(home, monkeypatch):
    # launchctl bootout returns at once; the node may still be handing its job back
    ctl = Launchctl()
    service.install(service.NODE, ["node"], run=ctl)
    asked = []
    monkeypatch.setattr(service, "_alive", lambda pid: asked.append(pid) or len(asked) < 3)
    monkeypatch.setattr(service.time, "sleep", lambda s: None)
    service.stop(service.NODE, ctl)
    assert asked == [4242, 4242, 4242]


def test_install_writes_the_definition_and_starts(home):
    ctl = Launchctl()
    path = service.install(service.NODE, ["node"], run=ctl)
    d = plistlib.loads(path.read_bytes())
    assert path == home / "Library/LaunchAgents/dev.llmat.badminton.node.plist"
    assert d["ProgramArguments"][1:] == ["-m", "badminton.cli", "node"]
    assert d["KeepAlive"] == {"SuccessfulExit": False} and d["RunAtLoad"] is True
    assert d["StandardOutPath"] == str(home / "Library/Logs/badminton/node.log")
    assert "/opt/homebrew/bin" in d["EnvironmentVariables"]["PATH"].split(":")
    assert service.state(service.NODE, ctl) == {"installed": True, "loaded": True, "pid": 4242, "lastExit": None}


def test_stop_unloads_and_start_loads_again(home):
    ctl = Launchctl()
    service.install(service.NODE, ["node"], run=ctl)
    service.stop(service.NODE, ctl)
    assert service.state(service.NODE, ctl)["loaded"] is False
    service.start(service.NODE, ctl)
    assert ctl.calls[-1][1] == "bootstrap" and service.state(service.NODE, ctl)["pid"] == 4242


def test_a_node_that_left_by_itself_is_kicked_not_loaded_twice(home):
    ctl = Launchctl()
    service.install(service.NODE, ["node"], run=ctl)
    ctl.pid = None                                # 「做完这个再停」: exited with 0, still loaded
    assert service.state(service.NODE, ctl) == {"installed": True, "loaded": True, "pid": None, "lastExit": 0}
    service.start(service.NODE, ctl)
    assert ctl.calls[-1][1] == "kickstart" and ctl.pid == 4243


def test_start_before_install_says_so_and_uninstall_removes_the_file(home):
    ctl = Launchctl()
    with pytest.raises(RuntimeError, match="还没有安装"):
        service.start(service.NODE, ctl)
    path = service.install(service.DASHBOARD, ["dashboard"], run=ctl)
    service.uninstall(service.DASHBOARD, ctl)
    assert not path.exists() and not ctl.loaded
