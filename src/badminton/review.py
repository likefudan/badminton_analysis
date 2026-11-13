"""Step 4: AI post-match review.

The program turns the analysis JSONs into a compact summary (numbers with
units, one line per rally, comparisons, data-quality caveats); a general chat
model then writes the review from that text. The model never sees video or
images, only the numbers below.

Three layers, each usable on its own:
  build_summary()  analysis JSON(s) -> summary dict (deterministic, saved as JSON)
  summary_text()   summary dict -> the text block that goes into the prompt
  build_prompt()   system + user prompt (Chinese output, fixed sections)
The prompt is always written to a file to paste into any chat model; the
Anthropic API is called only when the user asks for it (`badminton review --api`)
and ANTHROPIC_API_KEY is set. Only the prompt text is sent.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import os
import re
from pathlib import Path

import numpy as np

from .movement import ACTIVE_SPEED, PAUSE_MIN_S, SPRINT_MIN_S, SPRINT_SPEED

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 16000
SCHEMA = 1

TRACKED_WARN_PCT = 90.0  # a player tracked for less of the in-play time than this gets a caveat
LOW_CAMERA_M = 2.5  # below the recommended 2.5-4 m
SMALL_SAMPLE_RALLIES = 10
SMALL_SAMPLE_IN_PLAY_S = 300.0
SHUTTLE_SEPARATION_PCT = 15.0  # in-play vs dead "shuttle moving" must differ by this much to be useful

SECTIONS = ["比赛概况", "球员跑动画像", "三条观察", "三个练习建议", "数据不确定的地方"]

_SIDE = {"near": "近端", "far": "远端"}


# --- loading ---------------------------------------------------------------------


def is_movement(d: dict) -> bool:
    return isinstance(d.get("players"), list) and bool(d["players"]) and "distance_m" in d["players"][0]


_DROP_KEYS = {"heatmap", "counts", "x_edges", "y_edges"}


def compact(obj, max_list: int = 40):
    """Drop raw grids and long arrays from an analysis JSON, round floats."""
    if isinstance(obj, dict):
        return {k: compact(v, max_list) for k, v in obj.items() if k not in _DROP_KEYS and not k.startswith("_")}
    if isinstance(obj, list):
        if len(obj) > max_list:
            return f"（{len(obj)} 项，已省略）"
        return [compact(v, max_list) for v in obj]
    if isinstance(obj, float):
        return round(obj, 2)
    if isinstance(obj, str) and re.match(r"^(/|~|\.{1,2}/|[A-Za-z]:[\\/])", obj):
        return re.split(r"[\\/]", obj)[-1]  # file paths: the name only, not the folders
    return obj


def heatmap_centroid(hm: dict) -> tuple[float, float] | None:
    """Mean position (court metres) from a movement heatmap, before it is stripped."""
    try:
        c = np.asarray(hm["counts"], float)  # rows = y
        xe, ye = np.asarray(hm["x_edges"], float), np.asarray(hm["y_edges"], float)
    except (KeyError, TypeError, ValueError):
        return None
    if c.ndim != 2 or c.sum() <= 0 or c.shape != (len(ye) - 1, len(xe) - 1):
        return None
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    return float((c.sum(0) * xc).sum() / c.sum()), float((c.sum(1) * yc).sum() / c.sum())


def position_sigma(calib, xy: tuple[float, float]) -> dict:
    """1-sigma position error (metres) at a court point: the ankle keypoint error
    assumed by tracking.py pushed through the court homography."""
    from .tracking import KPT_SIGMA_PX, measurement_cov

    cov = measurement_cov(np.array([xy], float), np.array([True]), calib)[0]
    return {"along_m": round(float(np.sqrt(cov[1, 1])), 2), "across_m": round(float(np.sqrt(cov[0, 0])), 2),
            "assumed_keypoint_error_px": KPT_SIGMA_PX}


def shuttle_stats(csv_path: str | Path, fps: float, rallies: list[dict], offset_s: float = 0.0) -> dict:
    """Numbers from the per-frame shuttle CSV (Frame, Visibility, X, Y). Frame f is
    at f / fps + offset_s on the analysis clock. "Detected" means the tracker put a
    point there, not that the point is right."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows or fps <= 0:
        return {}
    n = max(int(r["Frame"]) for r in rows) + 1
    vis = np.zeros(n, bool)
    for r in rows:
        vis[int(r["Frame"])] = bool(int(r["Visibility"]))
    t = np.arange(n) / fps + offset_s
    gaps = [(b - a) / fps for a, b in _runs(~vis)]
    per_rally = []
    for r in rallies:
        m = (t >= r["start_s"]) & (t <= r["end_s"])
        if not m.any():
            per_rally.append(None)
            continue
        g = [(b - a) / fps for a, b in _runs(~vis[m])]
        per_rally.append({"detected_pct": round(100 * float(vis[m].mean()), 1),
                          "longest_gap_s": round(max(g), 2) if g else 0.0})
    return {"frames": int(n), "fps": round(float(fps), 3), "detected_pct": round(100 * float(vis.mean()), 1),
            "longest_gap_s": round(max(gaps), 2) if gaps else 0.0, "per_rally": per_rally}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


# --- summary -----------------------------------------------------------------------


def build_summary(movement: dict, *, source: str = "", extras: dict | None = None, calib=None,
                  shuttle: dict | None = None, shuttle_accuracy_pct: float | None = None) -> dict:
    """Compact, deterministic match summary from a movement JSON.

    extras: other analysis JSONs by name (e.g. positioning), passed through compacted.
    calib: CourtCalibration, for camera height and per-player position precision.
    shuttle: shuttle_stats() output. shuttle_accuracy_pct: hand-audited accuracy, if known.
    """
    in_play = float(movement.get("in_play_s", 0.0))
    duration = float(movement.get("duration_s", 0.0))
    raw_rallies = movement.get("rallies", [])
    lengths = [round(r["end_s"] - r["start_s"], 2) for r in raw_rallies]

    players = []
    for p in movement["players"]:
        mins = p["in_play_s"] / 60 if p.get("in_play_s") else 0.0
        q = {
            "id": p["id"], "name": p["name"], "side": p.get("side", ""),
            "distance_m": round(p["distance_m"], 1),
            "tracked_in_play_s": round(p["in_play_s"], 1),
            "tracked_pct": round(100 * p["in_play_s"] / in_play, 1) if in_play else 0.0,
            "mean_speed_mps": round(p["mean_speed_mps"], 2),
            "max_speed_mps": round(p["max_speed_mps"], 2),
            "sprints": int(p["sprints"]),
            "sprints_per_min": round(p["sprints"] / mins, 2) if mins else 0.0,
            "distance_per_min_m": round(p["distance_m"] / mins, 1) if mins else 0.0,
            "zones_pct": {k: round(v, 1) for k, v in p.get("zones_pct", {}).items()},
            "airborne_frames": int(p.get("airborne_frames", 0)),
        }
        if calib is not None:
            c = heatmap_centroid(p.get("heatmap", {}))
            if c is not None:
                q["position_sigma"] = position_sigma(calib, c)
        players.append(q)

    rallies = []
    for i, (r, length) in enumerate(zip(raw_rallies, lengths)):
        row = {"n": i + 1, "start_s": round(r["start_s"], 1), "end_s": round(r["end_s"], 1), "length_s": round(length, 1),
               "pause_before_s": round(r["start_s"] - raw_rallies[i - 1]["end_s"], 1) if i else None,
               "players": []}
        for p in movement["players"]:
            d = p.get("rally_distances_m", [])
            if i < len(d):
                row["players"].append({"name": p["name"], "distance_m": round(d[i], 1),
                                       "speed_mps": round(d[i] / length, 2) if length > 0 else 0.0})
        if shuttle and i < len(shuttle.get("per_rally", [])) and shuttle["per_rally"][i]:
            row["shuttle_detected_pct"] = shuttle["per_rally"][i]["detected_pct"]
        rallies.append(row)

    longest = int(np.argmax(lengths)) if lengths else None
    match = {
        "format": {2: "单打", 4: "双打"}.get(len(players), f"{len(players)} 人"),
        "duration_s": round(duration, 1), "in_play_s": round(in_play, 1),
        "in_play_pct": round(100 * in_play / duration, 1) if duration else 0.0,
        "rallies": len(rallies),
        "rally_mean_s": round(float(np.mean(lengths)), 1) if lengths else 0.0,
        "rally_median_s": round(float(np.median(lengths)), 1) if lengths else 0.0,
        "rally_longest": {"n": longest + 1, "length_s": round(lengths[longest], 1)} if lengths else None,
    }
    camera = None
    if calib is not None:
        try:
            from .camera import camera_from_calibration

            camera = {"height_m": round(float(camera_from_calibration(calib).center[2]), 2)}
            # an automatic calibration (or exactly four clicked points) fits its markers
            # exactly: an error of 0 there measures nothing, so it is left out
            if calib.reprojection_error_px >= 0.05:
                camera["reprojection_error_px"] = round(float(calib.reprojection_error_px), 1)
        except Exception:  # degenerate calibration: no camera numbers, the rest still works
            camera = None

    summary = {
        "schema": SCHEMA, "source": source, "match": match, "players": players, "rallies": rallies,
        "comparisons": _comparisons(players),
        "definitions": {"sprint_speed_mps": SPRINT_SPEED, "sprint_min_s": SPRINT_MIN_S,
                        "rally_active_speed_mps": ACTIVE_SPEED, "rally_pause_min_s": PAUSE_MIN_S},
    }
    if camera:
        summary["camera"] = camera
    if movement.get("shuttle_check"):
        summary["shuttle_check"] = movement["shuttle_check"]
    if shuttle:
        summary["shuttle"] = {k: v for k, v in shuttle.items() if k != "per_rally"}
        if shuttle_accuracy_pct is not None:
            summary["shuttle"]["audited_accuracy_pct"] = round(float(shuttle_accuracy_pct), 1)
    elif shuttle_accuracy_pct is not None:
        summary["shuttle"] = {"audited_accuracy_pct": round(float(shuttle_accuracy_pct), 1)}
    if extras:
        summary["extras"] = {k: compact(v) for k, v in extras.items()}
    summary["caveats"] = _caveats(summary)
    return summary


_METRICS = [("distance_m", "跑动距离", "米", 1), ("distance_per_min_m", "每分钟跑动", "米/分钟", 1),
            ("mean_speed_mps", "平均速度", "米/秒", 2), ("max_speed_mps", "最高速度", "米/秒", 2),
            ("sprints", "冲刺次数", "次", 0)]


def _comparisons(players: list[dict]) -> list[dict]:
    if len(players) < 2:
        return []
    out = []

    def add(label, unit, digits, values):
        hi, lo = max(values, key=values.get), min(values, key=values.get)
        diff = round(values[hi] - values[lo], digits)
        out.append({"metric": label, "unit": unit, "values": {k: round(v, digits) for k, v in values.items()},
                    "highest": hi if diff else None, "diff": diff,
                    "diff_pct": round(100 * diff / values[lo], 1) if values[lo] > 0 else None})

    for key, label, unit, digits in _METRICS:
        add(label, unit, digits, {p["name"]: p[key] for p in players})
    for z, label in (("front", "前场时间比例"), ("back", "后场时间比例")):
        if all(z in p["zones_pct"] for p in players):
            add(label, "%", 1, {p["name"]: p["zones_pct"][z] for p in players})
    return out


def _caveats(s: dict) -> list[str]:
    """Data-quality notes worked out from the numbers themselves (Chinese, for the prompt and the report)."""
    m, d = s["match"], s["definitions"]
    out = [
        "数据里只有跑动和站位：没有比分、没有每一拍的击球类型和落点、不知道每个回合谁赢。",
        f"回合是按球员动作切的（有人超过 {d['rally_active_speed_mps']:g} 米/秒算回合中，都慢于它超过 "
        f"{d['rally_pause_min_s']:g} 秒算死球）：回合开头常漏 1–2 秒，慢节奏的相持可能被切成两段，"
        "捡球时跑动快也可能被算进回合。回合数和回合长度是估计值。",
    ]
    if m["rallies"] < SMALL_SAMPLE_RALLIES or m["in_play_s"] < SMALL_SAMPLE_IN_PLAY_S:
        out.append(f"样本小：只有 {m['rallies']} 个回合、回合内 {m['in_play_s']:g} 秒，结论只代表这一段，不代表平时水平。")
    for p in s["players"]:
        if p["tracked_pct"] < TRACKED_WARN_PCT:
            out.append(f"{p['name']} 在回合内只有 {p['tracked_pct']:g}% 的时间被跟踪到（{p['tracked_in_play_s']:g} / "
                       f"{m['in_play_s']:g} 秒），丢失的时间不计距离，跑动距离偏小；和别人比较时看每分钟跑动和平均速度。")
    for p in s["players"]:
        sg = p.get("position_sigma")
        if sg:
            out.append(f"{p['name']}（{_SIDE.get(p['side'], p['side'])}）位置误差约：沿球场方向 {sg['along_m']:g} 米、"
                       f"横向 {sg['across_m']:g} 米（1 倍标准差，按脚踝关键点 {sg['assumed_keypoint_error_px']:g} 像素误差、"
                       "在其平均站位处换算）。")
    sig = [p for p in s["players"] if p.get("position_sigma")]
    if sig:
        worst = max(sig, key=lambda p: p["position_sigma"]["along_m"])
        if worst["position_sigma"]["along_m"] >= 0.3:
            out.append(f"{worst['name']} 的前后位置误差大，前中后场比例、最高速度和冲刺次数都会被误差抬高或搅乱，"
                       f"前后方向上小于约 {2 * worst['position_sigma']['along_m']:.1f} 米（2 倍标准差）的位置差别不要解读。")
    elif any(p["side"] == "far" for p in s["players"]):
        out.append("远端球员离相机远，位置精度低于近端（没有给球场标定，算不出具体数值）；远端的最高速度和冲刺次数可能偏高。")
    cam = s.get("camera")
    if cam and cam["height_m"] < LOW_CAMERA_M:
        out.append(f"相机高度约 {cam['height_m']:g} 米，低于建议的 2.5–4 米：远端一个像素对应的地面距离大，"
                   "跳起时脚的位置换算偏差也大。")
    air = [p for p in s["players"] if p["airborne_frames"]]
    if air:
        out.append("腾空或异常帧（" + "，".join(f"{p['name']} {p['airborne_frames']} 帧" for p in air) +
                   "）不用脚的位置，改用前后插值；这些帧附近的速度不可靠。腾空帧数不等于起跳次数。")
    c = s.get("shuttle_check")
    if c:
        txt = (f"和羽毛球轨迹对照：{c['agreement_pct']:g}% 的帧一致；回合内球在飞 {c['in_play_shuttle_moving_pct']:g}%，"
               f"死球时间球在飞 {c['dead_shuttle_moving_pct']:g}%。")
        if abs(c["in_play_shuttle_moving_pct"] - c["dead_shuttle_moving_pct"]) < SHUTTLE_SEPARATION_PCT:
            txt += "两者几乎一样，说明羽毛球轨迹目前分不出回合和死球，不能用来验证回合切分。"
        out.append(txt)
    sh = s.get("shuttle")
    if sh:
        parts = []
        if "detected_pct" in sh:
            parts.append(f"羽毛球在 {sh['detected_pct']:g}% 的帧里被检测到，最长连续丢失 {sh['longest_gap_s']:g} 秒")
        if "audited_accuracy_pct" in sh:
            parts.append(f"人工核对的位置正确率约 {sh['audited_accuracy_pct']:g}%")
        out.append("；".join(parts) + "。「检测到」不等于「位置正确」；轨迹有缺口，没有据此统计拍数、击球点或落点。")
    return out


# --- text for the prompt ---------------------------------------------------------------


def _g(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}"


def summary_text(s: dict) -> str:
    m = s["match"]
    lines = []
    head = (f"【比赛】{s.get('source') or '未命名'}｜{m['format']}｜视频 {_g(m['duration_s'])} 秒｜回合内 {_g(m['in_play_s'])} 秒"
            f"（{_g(m['in_play_pct'], 0)}%）｜{m['rallies']} 个回合｜平均 {_g(m['rally_mean_s'])} 秒，中位数 {_g(m['rally_median_s'])} 秒")
    if m.get("rally_longest"):
        head += f"｜最长是第 {m['rally_longest']['n']} 回合 {_g(m['rally_longest']['length_s'])} 秒"
    lines.append(head)
    d = s["definitions"]
    lines.append(f"【指标】所有统计只算回合内时间。冲刺 = 速度超过 {d['sprint_speed_mps']:g} 米/秒并持续 {d['sprint_min_s']:g} 秒以上。"
                 "前/中/后场 = 自己半场从球网到底线三等分；左/右按球员面向球网的方向。时间是视频里的秒数。"
                 "回合里每人的速度 = 该回合的距离 ÷ 回合长度，跟踪丢失时偏低。")
    if s.get("camera"):
        t = f"【相机】高度约 {s['camera']['height_m']:g} 米"
        if "reprojection_error_px" in s["camera"]:
            t += f"；球场标定重投影误差 {s['camera']['reprojection_error_px']:g} 像素"
        lines.append(t)
    lines.append("【球员】")
    for p in s["players"]:
        z = p["zones_pct"]
        t = (f"- {p['name']}（{p['id']} 号，{_SIDE.get(p['side'], p['side'])}）：跑动 {_g(p['distance_m'])} 米；"
             f"回合内被跟踪 {_g(p['tracked_in_play_s'])} 秒（覆盖 {_g(p['tracked_pct'], 0)}%）；每分钟 {_g(p['distance_per_min_m'])} 米；"
             f"平均 {_g(p['mean_speed_mps'], 2)} 米/秒；最高 {_g(p['max_speed_mps'], 2)} 米/秒；"
             f"冲刺 {p['sprints']} 次（{_g(p['sprints_per_min'], 2)} 次/分钟）")
        if z:
            t += (f"；站位 前 {_g(z.get('front', 0))}% / 中 {_g(z.get('mid', 0))}% / 后 {_g(z.get('back', 0))}%，"
                  f"左 {_g(z.get('left', 0))}% / 右 {_g(z.get('right', 0))}%")
        t += f"；腾空或异常帧 {p['airborne_frames']}"
        if p.get("position_sigma"):
            t += f"；位置误差 前后 {p['position_sigma']['along_m']:g} 米、左右 {p['position_sigma']['across_m']:g} 米"
        lines.append(t)
    lines.append("【回合】")
    for r in s["rallies"]:
        t = f"- 第 {r['n']} 回合 {_g(r['start_s'])}–{_g(r['end_s'])} 秒，长 {_g(r['length_s'])} 秒"
        if r["pause_before_s"] is not None:
            t += f"，之前停了 {_g(r['pause_before_s'])} 秒"
        if r["players"]:
            t += "：" + "，".join(f"{q['name']} {_g(q['distance_m'])} 米（{_g(q['speed_mps'], 2)} 米/秒）" for q in r["players"])
        if "shuttle_detected_pct" in r:
            t += f"；球被检测到 {_g(r['shuttle_detected_pct'], 0)}%"
        lines.append(t)
    if s["comparisons"]:
        lines.append("【对比】")
        for c in s["comparisons"]:
            digits = 0 if c["unit"] == "次" else (2 if c["unit"] == "米/秒" else 1)
            pct = c["unit"] == "%"
            unit = "%" if pct else f" {c['unit']}"
            t = f"- {c['metric']}：" + "，".join(f"{k} {_g(v, digits)}{unit}" for k, v in c["values"].items())
            if c["highest"]:
                t += f"；{c['highest']} 高 {_g(c['diff'], digits)}{' 个百分点' if pct else unit}"
                if c["diff_pct"] is not None and c["unit"] not in ("%", "次"):
                    t += f"（{_g(c['diff_pct'], 0)}%）"
            else:
                t += "；相同"
            lines.append(t)
    for name, extra in (s.get("extras") or {}).items():
        lines.append(f"【其他分析：{name}】")
        lines.append(json.dumps(extra, ensure_ascii=False, sort_keys=True))
    lines.append("【数据质量】")
    lines += [f"- {c}" for c in s["caveats"]]
    return "\n".join(lines)


def previous_text(prev: dict, cur: dict) -> str:
    """Comparison block against an earlier session's summary JSON. Sessions differ in
    length, so only rates and proportions are compared."""
    lines = [f"【上一次】{prev.get('source') or '未命名'}：{prev['match']['format']}，回合内 {_g(prev['match']['in_play_s'])} 秒，"
             f"{prev['match']['rallies']} 个回合，平均 {_g(prev['match']['rally_mean_s'])} 秒",
             "【和上一次比较】（这次 − 上次；两次时长不同，只比每分钟的量、速度和比例）",
             f"- 回合平均长度：{_g(cur['match']['rally_mean_s'])} 秒，上次 {_g(prev['match']['rally_mean_s'])} 秒，"
             f"变化 {cur['match']['rally_mean_s'] - prev['match']['rally_mean_s']:+.1f} 秒"]
    old = {p["name"]: p for p in prev["players"]}
    for p in cur["players"]:
        o = old.get(p["name"])
        if o is None:
            lines.append(f"- {p['name']}：上一次没有这个名字，无法比较")
            continue
        parts = [f"每分钟跑动 {_g(p['distance_per_min_m'])} 米（上次 {_g(o['distance_per_min_m'])}，{p['distance_per_min_m'] - o['distance_per_min_m']:+.1f}）",
                 f"平均速度 {_g(p['mean_speed_mps'], 2)} 米/秒（上次 {_g(o['mean_speed_mps'], 2)}，{p['mean_speed_mps'] - o['mean_speed_mps']:+.2f}）",
                 f"最高速度 {_g(p['max_speed_mps'], 2)} 米/秒（上次 {_g(o['max_speed_mps'], 2)}，{p['max_speed_mps'] - o['max_speed_mps']:+.2f}）",
                 f"冲刺 {_g(p['sprints_per_min'], 2)} 次/分钟（上次 {_g(o['sprints_per_min'], 2)}，{p['sprints_per_min'] - o['sprints_per_min']:+.2f}）"]
        for z, label in (("front", "前场"), ("back", "后场")):
            if z in p["zones_pct"] and z in o.get("zones_pct", {}):
                parts.append(f"{label} {_g(p['zones_pct'][z])}%（上次 {_g(o['zones_pct'][z])}%，{p['zones_pct'][z] - o['zones_pct'][z]:+.1f} 个百分点）")
        lines.append(f"- {p['name']}：" + "；".join(parts))
    if prev.get("caveats"):
        lines.append("上一次的数据质量说明：")
        lines += [f"- {c}" for c in prev["caveats"]]
    return "\n".join(lines)


# --- prompt ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一位羽毛球教练，帮业余球友复盘比赛。你看不到视频，手里只有程序从视频里算出来的数字（跑动、速度、站位、回合时间）。请只根据这些数字写点评，用简体中文，语气像球友之间说话，直接、具体。

必须遵守：
1. 只用给出的数字。数据里没有比分、没有每一拍的击球类型和落点、不知道回合谁赢，所以不要写出任何一次具体的击球（杀球、吊球、挑球等）、比分或胜负，也不要猜。可以说「这组数字常见于某种打法」，但要明说这是推测。
2. 每个判断都带上它依据的数字和单位；提到回合时写编号和时间，例如「第 2 回合（40.1–52.3 秒）」。
3. 「数据质量」里列出的问题要当真：受影响的数字要么不用，要么用的时候在同一句里说明不确定。差别小于误差的不要解读。
4. 不要重新计算或改动给出的数字，不要补出数据里没有的数字。
5. 不写客套话，不重复数据表。"""

REPORT_REQUEST = """请按下面的结构写点评，用 Markdown，五个二级标题的文字照抄，顺序不变：

## 比赛概况
三四句话：这一段打了多久、多少回合、节奏如何、两边（或各人）跑动上最大的差别。

## 球员跑动画像
每位球员一个三级标题（### 名字），写他的跑动特点：跑动量、速度、冲刺、前中后场和左右站位，各自对应的数字。

## 三条观察
正好三条，编号 1–3。每条：先一句结论，再给出依据的数字，至少一条要引用具体回合（编号和时间）。

## 三个练习建议
正好三条，编号 1–3。每条：练什么、怎么练（组数或时间）、为什么（对应上面哪条观察或哪个数字）。如果依据的数字不可靠，要说明这条建议的把握不大。

## 数据不确定的地方
逐条列出哪些数字不可靠、为什么、对上面哪些结论有影响；再用一句话说，要让下一次点评更准，录像或数据上最该补什么。"""

COMPARE_REQUEST = """另外数据里有上一次的摘要。在「比赛概况」之后加一个二级标题「## 和上一次比较」：说清哪些指标变了、变了多少，哪些变化小于误差不能算进步或退步。两次的对手、时长、机位可能不同，要提醒这一点。"""

ASK_REQUEST = """下面是球友的追问。根据上面的数据（以及你之前写的点评，如果有）回答，规则不变：只用给出的数字，带单位，引用回合编号和时间；数据答不了的就直说答不了，并说明需要什么数据。

追问：{question}"""


def build_prompt(summary: dict, previous: dict | None = None) -> tuple[str, str]:
    """(system, user) for the review."""
    parts = ["下面是这场比赛的统计摘要（程序计算，不含视频和图片）。", "", summary_text(summary)]
    if previous:
        parts += ["", previous_text(previous, summary)]
    parts += ["", REPORT_REQUEST]
    if previous:
        parts += ["", COMPARE_REQUEST]
    return SYSTEM_PROMPT, "\n".join(parts)


def build_messages(summary: dict, previous: dict | None = None, report: str | None = None,
                   question: str | None = None) -> tuple[str, list[dict]]:
    """Messages for the API. A follow-up question reuses the same context: the
    review prompt, the earlier report (if there is one) and then the question."""
    system, user = build_prompt(summary, previous)
    if question is None:
        return system, [{"role": "user", "content": user}]
    ask = ASK_REQUEST.format(question=question.strip())
    if report:
        return system, [{"role": "user", "content": user}, {"role": "assistant", "content": report},
                        {"role": "user", "content": ask}]
    context = "\n".join(["下面是这场比赛的统计摘要（程序计算，不含视频和图片）。", "", summary_text(summary)]
                        + (["", previous_text(previous, summary)] if previous else []))
    return system, [{"role": "user", "content": context + "\n\n" + ask}]


def request_json(summary: dict, previous: dict | None = None, model: str = DEFAULT_MODEL) -> dict:
    """Everything a server needs to ask for the review (and follow-up questions) without this code:
    the web site (cloud/) reads it, adds its own API key and calls the model. Text and numbers only."""
    system, user = build_prompt(summary, previous)
    return {"schema": SCHEMA, "model": model, "max_tokens": MAX_TOKENS, "system": system, "user": user,
            "ask": ASK_REQUEST, "title": summary.get("source", "")}


def prompt_file_text(system: str, messages: list[dict]) -> str:
    """One text file to paste into any chat model."""
    out = [system, ""]
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            out += ["———— 之前的点评 ————", msg["content"], "———— 点评结束 ————", ""]
        else:
            out += [msg["content"], ""]
    return "\n".join(out).rstrip() + "\n"


# --- backend ----------------------------------------------------------------------------


class ReviewError(RuntimeError):
    pass


def have_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def call_anthropic(system: str, messages: list[dict], model: str = DEFAULT_MODEL, client=None) -> str:
    """Send text only (no video, no images) to the Anthropic API and return the reply text.
    The key is read from ANTHROPIC_API_KEY by the SDK."""
    if client is None:
        try:
            import anthropic
        except ImportError as e:
            raise ReviewError('没有安装 anthropic 包：pip install -e ".[review]"') from e
        client = anthropic.Anthropic()
    try:
        with client.messages.stream(model=model, max_tokens=MAX_TOKENS, system=system, messages=messages) as stream:
            msg = stream.get_final_message()
    except Exception as e:  # SDK errors (auth, rate limit, network): report, the caller falls back to the prompt file
        if type(e).__module__.split(".")[0] in ("anthropic", "httpx", "httpx2"):
            raise ReviewError(f"调用 Anthropic API 失败（{type(e).__name__}）：{e}") from e
        raise
    if msg.stop_reason == "refusal":
        raise ReviewError("模型拒绝回答这个请求")
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if not text:
        raise ReviewError("模型没有返回文字")
    if msg.stop_reason == "max_tokens":
        text += "\n\n（回答到长度上限被截断）"
    return text


# --- report files --------------------------------------------------------------------------


def missing_sections(report: str) -> list[str]:
    heads = {h.strip() for h in re.findall(r"^##\s+(.+?)\s*$", report, flags=re.M)}
    return [s for s in SECTIONS if s not in heads]


def report_markdown(summary: dict, body: str, origin: str) -> str:
    m = summary["match"]
    names = "、".join(p["name"] for p in summary["players"])
    return (f"# 赛后点评：{summary.get('source') or '未命名'}\n\n"
            f"{names}｜{m['format']}｜{m['rallies']} 个回合｜回合内 {_g(m['in_play_s'])} 秒\n\n"
            f"> {origin}。点评只依据程序算出的数字，模型没有看视频。\n\n{body.strip()}\n")


def _inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def markdown_to_html(md: str) -> str:
    """The small part of Markdown the report uses: headings, lists, quotes, bold, paragraphs."""
    out, para, lst = [], [], None

    def flush():
        nonlocal para, lst
        if para:
            out.append("<p>" + "<br>".join(para) + "</p>")
            para = []
        if lst:
            out.append(f"</{lst}>")
            lst = None

    for line in md.splitlines():
        s = line.strip()
        h = re.match(r"^(#{1,4})\s+(.*)$", s)
        ol = re.match(r"^\d+[.、)]\s+(.*)$", s)
        ul = re.match(r"^[-*]\s+(.*)$", s)
        if not s:
            if para:
                out.append("<p>" + "<br>".join(para) + "</p>")
                para = []
            continue
        if h:
            flush()
            out.append(f"<h{len(h[1])}>{_inline(h[2])}</h{len(h[1])}>")
        elif ol or ul:
            kind = "ol" if ol else "ul"
            if para:
                out.append("<p>" + "<br>".join(para) + "</p>")
                para = []
            if lst != kind:
                if lst:
                    out.append(f"</{lst}>")
                out.append(f"<{kind}>")
                lst = kind
            out.append(f"<li>{_inline((ol or ul)[1])}</li>")
        elif s.startswith(">"):
            flush()
            out.append(f"<blockquote>{_inline(s.lstrip('> '))}</blockquote>")
        elif lst and line.startswith((" ", "\t")) and out and out[-1].endswith("</li>"):
            out[-1] = out[-1][:-5] + "<br>" + _inline(s) + "</li>"  # continuation of a list item
        else:
            if lst:
                out.append(f"</{lst}>")
                lst = None
            para.append(_inline(s))
    flush()
    return "\n".join(out)


_CSS = """body{font-family:-apple-system,"PingFang SC","Noto Sans SC","Microsoft YaHei",sans-serif;max-width:920px;
margin:0 auto;padding:24px 16px;line-height:1.7;color:#1c1c1e;background:#fff}
h1{font-size:1.6em}h2{margin-top:1.6em;border-bottom:1px solid #ddd;padding-bottom:.2em}
blockquote{margin:1em 0;padding:.5em 1em;background:#f4f4f6;border-left:4px solid #b0b0b8;color:#444}
.cards{display:flex;gap:12px;flex-wrap:wrap}.cards img{flex:1 1 280px;min-width:0;max-width:100%;border:1px solid #ddd;border-radius:6px}
li{margin:.3em 0}code{background:#f0f0f2;padding:0 .3em;border-radius:3px}
@media(prefers-color-scheme:dark){body{background:#1c1c1e;color:#e8e8ea}blockquote{background:#2a2a2e;color:#c8c8cc}
h2{border-color:#444}.cards img{border-color:#444}code{background:#333}}"""


def report_html(markdown: str, cards: list[Path], title: str) -> str:
    """Self-contained page: the movement cards side by side (embedded), then the report."""
    body = markdown_to_html(markdown)
    imgs = "".join(
        f'<img alt="{html.escape(c.name)}" src="data:image/png;base64,{base64.b64encode(c.read_bytes()).decode()}">'
        for c in cards)
    cards_html = f'<div class="cards">{imgs}</div>' if imgs else ""
    if cards_html and "</blockquote>" in body:  # cards go under the title block, above the first section
        body = body.replace("</blockquote>", "</blockquote>\n" + cards_html, 1)
    else:
        body = cards_html + body
    return (f'<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
            f"<style>{_CSS}</style></head><body>\n{body}\n</body></html>\n")


def find_cards(movement_json: Path, summary: dict) -> list[Path]:
    stem = stem_of(movement_json)
    cards = [movement_json.parent / f"{stem}.movement.{p['id']}.png" for p in summary["players"]]
    return [c for c in cards if c.exists()]


def stem_of(path: Path) -> str:
    name = path.name
    for suffix in (".movement.json", ".positioning.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem
