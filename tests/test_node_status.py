"""What the node reports about this Mac: background jobs found from the process list, their logs and progress."""

import json
import sys
from pathlib import Path

from badminton import node, node_status
from test_node import FakeCloud, config, fake_analyse  # noqa: F401

REPO = "/Users/me/codebase/badminton_analysis"
PY = "/opt/homebrew/Cellar/python@3.12/3.12.16/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python"


def proc(pid, ppid, command, age=600.0, cpu=1.0, rss=100):
    return {"pid": pid, "ppid": ppid, "age": age, "cpu": cpu, "rssMb": rss, "command": command}


PROCESSES = [
    proc(100, 1, f"{PY} .venv/bin/badminton node"),                               # the node itself
    proc(101, 100, f"{PY} -m badminton.cli analyze x.mp4"),                       # a step the node started
    proc(200, 1, f"{PY} scripts/tracknet_finetune.py --epochs 10", cpu=80, rss=2000),
    proc(201, 200, f"{PY} -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5)", cpu=20, rss=300),
    proc(300, 1, f"{PY} -m http.server"),                                         # Python, but somewhere else
    proc(400, 1, "/usr/sbin/cfprefsd agent"),
    proc(500, 1, f"{PY} -m pytest tests", age=3.0),                               # too young to be worth listing
    proc(600, 1, "/bin/zsh -c source ~/.zshrc && nohup .venv/bin/badminton node >> node.log"),   # a shell, not a job
    proc(700, 1, "/Users/me/.venv/bin/python3.12 scripts/bst_trial.py"),
]
FILES = {
    200: {"cwd": REPO + "/data/exp/run", "out": REPO + "/data/exp/train.log"},
    201: {"cwd": REPO + "/data/exp/run"},
    300: {"cwd": "/Users/me/Downloads"},
    500: {"cwd": REPO},
    600: {"cwd": REPO, "out": REPO + "/node.log"},
    700: {"cwd": REPO},
}


def test_background_jobs_are_found_without_the_node_its_steps_and_helper_processes():
    asked = []

    def files(pids):
        asked.extend(pids)
        return FILES

    tasks = node_status.find_tasks([Path(REPO)], own_pid=100, processes=PROCESSES, files=files)
    assert [t["pid"] for t in tasks] == [200, 700]
    assert tasks[1]["command"] == "scripts/bst_trial.py"
    t = tasks[0]
    assert t["command"] == "scripts/tracknet_finetune.py --epochs 10"   # without the interpreter's long path
    assert t["log"] == REPO + "/data/exp/train.log"
    assert (t["cpu"], t["rssMb"], t["processes"]) == (100.0, 2300, 2)   # the data loader counts with its parent
    assert not {100, 101, 400, 600} & set(asked)


def test_ps_and_lsof_output_is_parsed():
    class Res:
        def __init__(self, stdout):
            self.stdout = stdout

    ps = "  200     1 01-02:03:04  14.3 2058240 /usr/bin/python3 train.py --lr 1e-4\n   7     1    00:09   0.0   1024 /sbin/launchd\n"
    got = node_status.list_processes(run=lambda *a, **k: Res(ps))
    assert got[0] == {"pid": 200, "ppid": 1, "age": 93784.0, "cpu": 14.3, "rssMb": 2010,
                      "command": "/usr/bin/python3 train.py --lr 1e-4"}
    assert got[1]["age"] == 9.0
    lsof = "p200\nfcwd\ntDIR\nn/repo/run\nf1\ntREG\nn/repo/train.log\np201\nfcwd\ntDIR\nn/repo\nf1\ntCHR\nn/dev/ttys001\n"
    assert node_status.open_files([200, 201], run=lambda *a, **k: Res(lsof)) == {
        200: {"cwd": "/repo/run", "out": "/repo/train.log"}, 201: {"cwd": "/repo"}}   # a terminal is not a log file


def test_log_tail_shows_a_redrawn_progress_bar_once_and_reads_the_epoch(tmp_path):
    log = tmp_path / "train.log"
    log.write_bytes(("Epoch runtime: 0.14 hrs\nEpoch [7 / 10]\n\n"
                     "  0%|          | 0/155 [00:00<?, ?it/s]\r 25%|██▌       | 39/155 [01:44<05:05,  2.64s/it]\r"
                     " 26%|██▌       | 40/155 [01:46<05:03,  2.64s/it]").encode())
    lines = node_status.read_tail(str(log))
    assert lines == ["Epoch runtime: 0.14 hrs", "Epoch [7 / 10]", " 26%|██▌       | 40/155 [01:46<05:03,  2.64s/it]"]
    assert node_status.read_progress(lines) == {"label": "第 7/10 轮，40/155", "percent": 62.6}
    assert node_status.read_progress(["Evaluation:  50%|█████     | 2/4 [00:03<00:03,  1.57s/it, TP=65]"]) == {
        "label": "Evaluation 2/4", "percent": 50.0, "etaSeconds": 3.0}
    assert node_status.read_progress(["骨架识别 600/1200 帧"]) == {"label": "600/1200 帧", "percent": 50.0}
    assert node_status.read_progress(["nothing to see"]) == {}


def test_secrets_do_not_leave_the_mac():
    assert node_status.clean("curl -H 'Authorization: Bearer abc123'") == "curl -H 'Authorization: Bearer ***'"
    assert node_status.clean("run --api-key sk-123 --epochs 3") == "run --api-key *** --epochs 3"
    assert node_status.clean("TOKEN=hunter2 python x.py") == "TOKEN=*** python x.py"
    assert node_status.clean('{"token": "abc", "n": 1}') == '{"token": "***", "n": 1}'
    assert node_status.clean("sent with Bearer abc.def") == "sent with Bearer ***"
    # ordinary log lines that only mention the words stay readable
    for line in ("input tokens 900, output tokens 40", "口令不对，连不上", "Epoch [3 / 10] loss=4.9e-5"):
        assert node_status.clean(line) == line


def test_report_keeps_a_task_after_it_ends_and_flags_an_error(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Epoch [2 / 10]\n")
    now = {"t": 1_000_000.0}
    found = [{"pid": 200, "command": "scripts/train.py", "cwd": str(tmp_path), "log": str(log), "age": 60.0, "cpu": 50.0,
              "rssMb": 900, "processes": 3}]
    sent = []
    r = node_status.Reporter(sent.append, "mac", tmp_path / "work", [tmp_path], clock=lambda: now["t"],
                             tasks_fn=lambda: list(found), system_fn=lambda d: {"cpus": 10})
    r.set_job("process", {"id": "v1", "filename": "IMG_1.mov", "title": "周六双打"})
    node_status.note("开始处理 IMG_1.mov")
    r.report()
    s = sent[-1]
    assert s["job"] == {"task": "process", "videoId": "v1", "name": "周六双打", "since": "1970-01-12T13:46:40Z"}
    assert s["system"] == {"cpus": 10} and s["log"][-1].endswith("开始处理 IMG_1.mov")
    t = s["tasks"][0]
    assert t["running"] and t["progress"]["label"] == "第 2/10 轮" and t["startedAt"] == "1970-01-12T13:45:40Z"
    json.dumps(s)   # goes over the wire as JSON

    now["t"] += 30
    found[0]["age"] = 90.0
    r.report()
    assert len(sent[-1]["tasks"]) == 1                    # the same process, not a second entry

    now["t"] += 30
    found.clear()
    log.write_text("Epoch [3 / 10]\nTraceback (most recent call last):\nRuntimeError: MPS backend out of memory\n")
    r.set_job(None)
    r.report()
    t = sent[-1]["tasks"][0]
    assert not t["running"] and t["failed"] and t["endedAt"] == "1970-01-12T13:47:40Z" and "RuntimeError" in t["log"][-1]
    assert sent[-1]["job"] is None

    # a restarted node still knows what ended before; a day later it is forgotten
    r2 = node_status.Reporter(sent.append, "mac", tmp_path / "work", [tmp_path], clock=lambda: now["t"],
                              tasks_fn=lambda: [], system_fn=lambda d: {})
    assert [t["pid"] for t in r2.collect()["tasks"]] == [200]
    now["t"] += 25 * 3600
    assert r2.collect()["tasks"] == []


def test_a_broken_probe_or_an_old_site_does_not_stop_the_node(tmp_path, capsys):
    def boom():
        raise OSError("ps is gone")

    def send(data):
        raise node.ApiError(404, "没有这个接口")

    r = node_status.Reporter(send, "mac", tmp_path, tasks_fn=boom, system_fn=lambda d: {})
    assert r.collect()["tasksError"] == "OSError: ps is gone"
    r.report()
    r.report()
    assert capsys.readouterr().out.count("上报 Mac 状态失败") == 1   # said once, not every half minute


def test_the_node_reports_its_status_while_it_waits(tmp_path, monkeypatch):
    cloud = FakeCloud(b"x" * 1000)
    try:
        monkeypatch.setattr(node_status, "find_tasks", lambda watch, own: [])
        monkeypatch.setattr(node_status, "INTERVAL", 0.05)

        class Stop(Exception):
            pass

        def sleep(seconds):   # the node's wait between two questions for work: by then a report has gone out
            import time as real
            deadline = real.monotonic() + 5
            while not cloud.statuses and real.monotonic() < deadline:
                real.sleep(0.01)
            raise Stop

        monkeypatch.setattr(node.time, "sleep", sleep)
        stdout = sys.stdout
        try:
            node.run(config(cloud, tmp_path), analyse_fn=fake_analyse)
        except Stop:
            pass
        assert sys.stdout is stdout
        body = cloud.statuses[0]
        assert body["node"] == "test-mac" and body["status"]["name"] == "test-mac"
        assert body["status"]["tasks"] == [] and "startedAt" in body["status"]
        assert cloud.jobs[-1]["status"] == "done"            # the queued recording was processed as before
    finally:
        cloud.server.shutdown()
