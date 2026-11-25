"""Utilisation readings: the parsers, on output as the tools print it."""

import json
from types import SimpleNamespace

from badminton import metrics

MACMON = json.dumps({
    "timestamp": "2026-11-25T12:00:00Z", "temp": {"cpu_temp_avg": 51.4, "gpu_temp_avg": 44.0},
    "memory": {"ram_total": 34359738368, "ram_usage": 17179869184, "swap_total": 0, "swap_usage": 0},
    "ecpu_usage": [1181, 0.20], "pcpu_usage": [2360, 0.50], "gpu_usage": [461, 0.83],
    "cpu_power": 3.5, "gpu_power": 9.25, "ane_power": 0.0, "all_power": 12.75, "sys_power": 20.1})


def test_macmon_line_becomes_percentages_watts_and_gigabytes():
    s = metrics.parse_macmon(MACMON, cores=(6, 4))
    assert s["pcpu"] == 50.0 and s["ecpu"] == 20.0
    assert s["cpu"] == 38.0                      # (50 * 6 + 20 * 4) / 10
    assert s["gpu"] == 83.0
    assert s["anePowerW"] == 0.0 and s["gpuPowerW"] == 9.25 and s["allPowerW"] == 12.75
    assert s["memUsedGb"] == 16.0 and s["memTotalGb"] == 32.0
    assert s["cpuTemp"] == 51


def test_what_is_not_a_sample_is_ignored_and_missing_fields_are_left_out():
    assert metrics.parse_macmon("not json") is None
    assert metrics.parse_macmon("[1, 2]") is None
    assert metrics.parse_macmon("{}") is None
    assert metrics.parse_macmon(json.dumps({"gpu_usage": [0, 0.5]})) == {"gpu": 50.0}


def test_top_and_ioreg_are_the_fallback():
    assert metrics.parse_top("CPU usage: 12.50% user, 8.25% sys, 79.25% idle") == 20.8
    assert metrics.parse_top("Processes: 512 total") is None
    text = '| "PerformanceStatistics" = {"Tiled Scene Bytes"=0,"Device Utilization %"=37,"Renderer Utilization %"=35}'
    assert metrics.parse_ioreg(text) == 37.0
    assert metrics.parse_ioreg("nothing") is None


def test_one_sample_for_the_report(monkeypatch):
    monkeypatch.setattr(metrics, "find_macmon", lambda: "/opt/homebrew/bin/macmon")
    monkeypatch.setattr(metrics, "core_counts", lambda: (6, 4))
    run = lambda args, **kw: SimpleNamespace(stdout=MACMON + "\n", returncode=0)
    assert metrics.sample_once(run)["source"] == "macmon"
    monkeypatch.setattr(metrics, "find_macmon", lambda: None)
    run = lambda args, **kw: SimpleNamespace(stdout='"Device Utilization %"=12', returncode=0)
    assert metrics.sample_once(run) == {"source": "ioreg", "gpu": 12.0}
    assert metrics.sample_once(lambda args, **kw: SimpleNamespace(stdout="", returncode=1)) == {}


def test_sampler_keeps_a_bounded_history_and_drops_a_stale_reading():
    s = metrics.Sampler(interval=2.0, history=3)
    for i in range(5):
        s.add({"cpu": float(i), "gpu": 1.0})
    out = s.to_json()
    assert out["history"]["cpu"] == [2.0, 3.0, 4.0] and out["history"]["anePowerW"] == [None] * 3
    assert out["now"] == {"cpu": 4.0, "gpu": 1.0}
    s.samples[-1]["t"] -= 60                     # the reader stopped a minute ago
    assert s.to_json()["now"] == {}
