"""How busy this Mac is: CPU, GPU, the neural engine (NPU), memory. No sudo.

The first choice is ``macmon`` (``brew install macmon``), which reads the same counters as Apple's
powermetrics without root and prints one JSON object per sample. Without it the CPU comes from
``top`` and the GPU from ``ioreg``; the neural engine is then not available.

macOS does not publish a utilisation of the neural engine, only the power it draws. That is what is
shown, in watts: turning it into a percentage would need a made-up maximum.
"""

from __future__ import annotations

import collections
import json
import re
import shutil
import subprocess
import threading
import time

INTERVAL = 2.0
HISTORY = 300          # samples kept: ten minutes
FIELDS = ("cpu", "ecpu", "pcpu", "gpu", "anePowerW", "cpuPowerW", "gpuPowerW", "allPowerW", "memUsedGb", "memTotalGb",
          "cpuTemp", "gpuTemp")
CURVES = ("cpu", "gpu", "anePowerW", "memUsedGb")
_PLACES = ("/opt/homebrew/bin/macmon", "/usr/local/bin/macmon")


def find_macmon() -> str | None:
    import os

    return shutil.which("macmon") or next((p for p in _PLACES if os.path.exists(p)), None)


def _sysctl(name: str) -> int | None:
    try:
        return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def core_counts() -> tuple[int | None, int | None]:
    """(performance cores, efficiency cores) of an Apple Silicon Mac."""
    return _sysctl("hw.perflevel0.logicalcpu"), _sysctl("hw.perflevel1.logicalcpu")


def _percent(value) -> float | None:
    """macmon gives a usage as [frequency in MHz, share 0..1]."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        value = value[1]
    if not isinstance(value, (int, float)):
        return None
    return round(max(0.0, min(100.0, value * 100)), 1)


def _number(value, digits: int = 2) -> float | None:
    return round(float(value), digits) if isinstance(value, (int, float)) else None


def parse_macmon(line: str, cores: tuple[int | None, int | None] = (None, None)) -> dict | None:
    """One line of ``macmon pipe`` as a sample; None when it is not one."""
    try:
        d = json.loads(line)
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    out: dict = {}
    ecpu, pcpu = _percent(d.get("ecpu_usage")), _percent(d.get("pcpu_usage"))
    if ecpu is not None:
        out["ecpu"] = ecpu
    if pcpu is not None:
        out["pcpu"] = pcpu
    if ecpu is not None and pcpu is not None:
        p, e = cores[0] or 1, cores[1] or 1   # all cores together: the two groups weighted by their size
        out["cpu"] = round((pcpu * p + ecpu * e) / (p + e), 1)
    elif pcpu is not None or ecpu is not None:
        out["cpu"] = pcpu if pcpu is not None else ecpu
    gpu = _percent(d.get("gpu_usage"))
    if gpu is not None:
        out["gpu"] = gpu
    for key, name in (("ane_power", "anePowerW"), ("cpu_power", "cpuPowerW"), ("gpu_power", "gpuPowerW"),
                      ("all_power", "allPowerW")):
        v = _number(d.get(key))
        if v is not None:
            out[name] = v
    mem = d.get("memory") or {}
    if isinstance(mem.get("ram_usage"), (int, float)) and isinstance(mem.get("ram_total"), (int, float)):
        out["memUsedGb"] = round(mem["ram_usage"] / 2**30, 1)
        out["memTotalGb"] = round(mem["ram_total"] / 2**30, 1)
    temp = d.get("temp") or {}
    for key, name in (("cpu_temp_avg", "cpuTemp"), ("gpu_temp_avg", "gpuTemp")):
        v = _number(temp.get(key), 0)
        if v:
            out[name] = v
    return out or None


_TOP = re.compile(r"CPU usage:\s*([\d.]+)% user,\s*([\d.]+)% sys")
_GPU = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')


def parse_top(line: str) -> float | None:
    m = _TOP.search(line)
    return round(min(100.0, float(m.group(1)) + float(m.group(2))), 1) if m else None


def parse_ioreg(text: str) -> float | None:
    """GPU utilisation from ``ioreg -r -d 1 -w 0 -c IOAccelerator``."""
    values = [int(v) for v in _GPU.findall(text)]
    return float(max(values)) if values else None


def gpu_from_ioreg(run=subprocess.run) -> float | None:
    try:
        return parse_ioreg(run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True,
                               timeout=10).stdout)
    except (OSError, subprocess.SubprocessError):
        return None


def sample_once(run=subprocess.run) -> dict:
    """One sample for the node's half-minute report; {} when nothing can be read."""
    macmon = find_macmon()
    if macmon:
        try:
            res = run([macmon, "pipe", "-s", "1", "-i", "500"], capture_output=True, text=True, timeout=15)
            for line in reversed(res.stdout.splitlines()):
                sample = parse_macmon(line, core_counts())
                if sample:
                    return {"source": "macmon", **sample}
        except (OSError, subprocess.SubprocessError):
            pass
    gpu = gpu_from_ioreg(run)
    return {"source": "ioreg", "gpu": gpu} if gpu is not None else {}


class Sampler:
    """Keeps the last ten minutes of samples; a background thread fills them in."""

    def __init__(self, interval: float = INTERVAL, history: int = HISTORY):
        self.interval = interval
        self.samples: collections.deque[dict] = collections.deque(maxlen=history)
        self.source = ""
        self.note = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None

    def add(self, sample: dict) -> None:
        with self._lock:
            self.samples.append({"t": round(time.time(), 1), **sample})

    def to_json(self) -> dict:
        with self._lock:
            samples = list(self.samples)
        last = samples[-1] if samples else {}
        fresh = bool(last) and time.time() - last["t"] < self.interval * 5
        return {
            "source": self.source, "note": self.note, "interval": self.interval,
            "now": {k: last[k] for k in FIELDS if k in last} if fresh else {},
            "history": {"t": [s["t"] for s in samples], **{k: [s.get(k) for s in samples] for k in CURVES}},
        }

    # -- the two ways of reading --

    def _macmon(self, macmon: str) -> bool:
        """Reads until macmon ends. False when it never gave a sample (then the other way is used)."""
        cores = core_counts()
        got = False
        try:
            self._proc = subprocess.Popen([macmon, "pipe", "-i", str(int(self.interval * 1000))], stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True)
        except OSError:
            return False
        try:
            for line in self._proc.stdout:
                sample = parse_macmon(line, cores)
                if sample:
                    got = True
                    self.source = "macmon"
                    self.note = ""
                    self.add(sample)
                if self._stop.is_set():
                    break
        finally:
            self._proc.terminate()
        return got

    def _basic(self) -> None:
        self.source = "top + ioreg"
        self.note = "没有装 macmon：只有 CPU 和 GPU，没有 NPU 功耗。运行 brew install macmon 后重启面板。"
        try:
            self._proc = subprocess.Popen(["top", "-l", "0", "-n", "0", "-s", str(max(1, int(self.interval)))],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True)
        except OSError:
            self.note = "读不到利用率（这台机器上没有 macmon，也没有 top）"
            return
        first = True
        try:
            for line in self._proc.stdout:
                cpu = parse_top(line)
                if cpu is None:
                    continue
                if first:  # top's first sample is the average since boot
                    first = False
                    continue
                sample = {"cpu": cpu}
                gpu = gpu_from_ioreg()
                if gpu is not None:
                    sample["gpu"] = gpu
                self.add(sample)
                if self._stop.is_set():
                    break
        finally:
            self._proc.terminate()

    def _loop(self) -> None:
        while not self._stop.is_set():
            macmon = find_macmon()
            if not (macmon and self._macmon(macmon)):
                self._basic()
            self._stop.wait(5)  # whichever reader ended: try again shortly

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="metrics").start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
