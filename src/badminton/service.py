"""The node and the dashboard as launchd services of the logged-in user (macOS LaunchAgents).

launchd starts them at login, restarts them when they crash, and writes their output to
~/Library/Logs/badminton/. Nothing here needs sudo: everything happens in the user's own
``gui/<uid>`` domain.

Stopping is ``launchctl bootout``: launchd sends SIGTERM (the node hands its job back to the queue
and exits), waits EXIT_TIMEOUT seconds, kills what is left, and forgets the service until the next
start or login. A node that exits by itself with code 0 (「做完这个再停」) is not restarted either.
"""

from __future__ import annotations

import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path

NODE = "dev.llmat.badminton.node"
DASHBOARD = "dev.llmat.badminton.dashboard"
REPO = Path(__file__).resolve().parents[2]
EXIT_TIMEOUT = 30
# launchd gives its services a bare PATH; ffmpeg and macmon come from Homebrew
EXTRA_PATH = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin")


def agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def log_dir() -> Path:
    return Path.home() / "Library" / "Logs" / "badminton"


def plist_path(label: str) -> Path:
    return agents_dir() / f"{label}.plist"


def log_path(label: str) -> Path:
    return log_dir() / (label.rsplit(".", 1)[-1] + ".log")


def domain() -> str:
    return f"gui/{os.getuid()}"


def service_path() -> str:
    """The PATH of the shell that installs the service, with the usual places added."""
    seen: list[str] = []
    for part in [*os.environ.get("PATH", "").split(":"), *EXTRA_PATH]:
        if part and part not in seen:
            seen.append(part)
    return ":".join(seen)


def definition(label: str, args: list[str], python: str = sys.executable) -> dict:
    return {
        "Label": label,
        "ProgramArguments": [python, "-m", "badminton.cli", *args],
        "WorkingDirectory": str(REPO),
        "RunAtLoad": True,
        # restarted after a crash, not after a clean exit (a stop that was asked for)
        "KeepAlive": {"SuccessfulExit": False},
        "ExitTimeOut": EXIT_TIMEOUT,
        "ProcessType": "Standard",   # not throttled like a "Background" job: the analysis needs the whole machine
        "StandardOutPath": str(log_path(label)),
        "StandardErrorPath": str(log_path(label)),
        "EnvironmentVariables": {"PATH": service_path(), "PYTHONUNBUFFERED": "1", "LANG": "zh_CN.UTF-8"},
    }


def _run(args: list[str], run=subprocess.run, timeout: float = EXIT_TIMEOUT + 15) -> subprocess.CompletedProcess:
    return run(args, capture_output=True, text=True, timeout=timeout)


def state(label: str, run=subprocess.run) -> dict:
    """{"installed", "loaded", "pid", "lastExit"}: what launchd knows about the service."""
    out = {"installed": plist_path(label).exists(), "loaded": False, "pid": None, "lastExit": None}
    try:
        res = _run(["launchctl", "print", f"{domain()}/{label}"], run, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return out
    if res.returncode != 0:
        return out
    out["loaded"] = True
    m = re.search(r"^\s*pid = (\d+)", res.stdout, re.M)
    if m:
        out["pid"] = int(m.group(1))
    m = re.search(r"^\s*last exit code = (-?\d+)", res.stdout, re.M)
    if m:
        out["lastExit"] = int(m.group(1))
    return out


def install(label: str, args: list[str], run=subprocess.run) -> Path:
    """Write the service's definition (again) and start it. A running one is restarted with the new definition."""
    path = plist_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_dir().mkdir(parents=True, exist_ok=True)
    if state(label, run)["loaded"]:
        _run(["launchctl", "bootout", f"{domain()}/{label}"], run)
    path.write_bytes(plistlib.dumps(definition(label, args)))
    start(label, run)
    return path


def uninstall(label: str, run=subprocess.run) -> None:
    if state(label, run)["loaded"]:
        _run(["launchctl", "bootout", f"{domain()}/{label}"], run)
    plist_path(label).unlink(missing_ok=True)


def start(label: str, run=subprocess.run) -> None:
    s = state(label, run)
    if not s["installed"]:
        raise RuntimeError("还没有安装成服务")
    if s["pid"]:
        return
    if s["loaded"]:  # loaded but not running: it exited cleanly by itself
        res = _run(["launchctl", "kickstart", f"{domain()}/{label}"], run)
    else:
        res = _run(["launchctl", "bootstrap", domain(), str(plist_path(label))], run)
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout).strip() or f"launchctl 返回 {res.returncode}")


def stop(label: str, run=subprocess.run) -> None:
    """Returns when the process is gone (at most EXIT_TIMEOUT seconds later)."""
    if state(label, run)["loaded"]:
        res = _run(["launchctl", "bootout", f"{domain()}/{label}"], run)
        if res.returncode != 0 and state(label, run)["loaded"]:
            raise RuntimeError((res.stderr or res.stdout).strip() or f"launchctl 返回 {res.returncode}")


def restart(label: str, run=subprocess.run) -> None:
    s = state(label, run)
    if s["loaded"] and s["pid"]:
        res = _run(["launchctl", "kickstart", "-k", f"{domain()}/{label}"], run)
        if res.returncode != 0:
            raise RuntimeError((res.stderr or res.stdout).strip() or f"launchctl 返回 {res.returncode}")
    else:
        start(label, run)
