"""Step 4: summary builder, prompt, backends. No network: the API client is a fake."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from badminton import review as rv
from badminton.cli import main
from badminton.court import MARKERS, CourtCalibration


def movement_json():
    heat = {"x_edges": [0.0, 3.05, 6.1], "y_edges": [0.0, 3.35, 6.7], "counts": [[1, 1], [1, 1]]}
    far_heat = {"x_edges": [0.0, 3.05, 6.1], "y_edges": [6.7, 10.05, 13.4], "counts": [[0, 0], [2, 2]]}
    return {
        "duration_s": 100.0, "in_play_s": 60.0,
        "rallies": [{"start_s": 5.0, "end_s": 25.0}, {"start_s": 30.0, "end_s": 40.0}, {"start_s": 50.0, "end_s": 80.0}],
        "players": [
            {"id": 1, "name": "小王", "side": "near", "in_play_s": 60.0, "distance_m": 90.0, "mean_speed_mps": 1.5,
             "max_speed_mps": 5.0, "sprints": 3, "zones_pct": {"front": 20.0, "mid": 50.0, "back": 30.0, "left": 40.0, "right": 60.0},
             "airborne_frames": 12, "rally_distances_m": [30.0, 15.0, 45.0], "heatmap": heat},
            {"id": 2, "name": "小李", "side": "far", "in_play_s": 48.0, "distance_m": 96.0, "mean_speed_mps": 2.0,
             "max_speed_mps": 6.0, "sprints": 8, "zones_pct": {"front": 5.0, "mid": 25.0, "back": 70.0, "left": 50.0, "right": 50.0},
             "airborne_frames": 0, "rally_distances_m": [32.0, 16.0, 48.0], "heatmap": far_heat},
        ],
        "shuttle_check": {"agreement_pct": 60.0, "in_play_shuttle_moving_pct": 70.0, "dead_shuttle_moving_pct": 68.0},
    }


def calibration(camera):
    names = ["near_left_corner", "near_right_corner", "far_left_corner", "far_right_corner",
             "near_short_service_left", "far_short_service_right"]
    pts = camera.project(np.array([MARKERS[n] for n in names]))
    return CourtCalibration({n: tuple(p) for n, p in zip(names, pts)}, camera.size)


def test_summary_numbers_and_no_heatmap():
    s = rv.build_summary(movement_json(), source="match")
    assert "heatmap" not in json.dumps(s)
    m = s["match"]
    assert m["format"] == "单打" and m["rallies"] == 3 and m["in_play_pct"] == 60.0
    assert m["rally_mean_s"] == 20.0 and m["rally_longest"] == {"n": 3, "length_s": 30.0}
    wang, li = s["players"]
    assert wang["tracked_pct"] == 100.0 and li["tracked_pct"] == 80.0
    assert wang["distance_per_min_m"] == 90.0 and li["distance_per_min_m"] == 120.0
    assert li["sprints_per_min"] == 10.0
    r = s["rallies"]
    assert r[0]["pause_before_s"] is None and r[1]["pause_before_s"] == 5.0
    assert r[1]["players"][0] == {"name": "小王", "distance_m": 15.0, "speed_mps": 1.5}
    dist = next(c for c in s["comparisons"] if c["metric"] == "跑动距离")
    assert dist["highest"] == "小李" and dist["diff"] == 6.0


def test_summary_is_deterministic():
    a = rv.summary_text(rv.build_summary(movement_json(), source="m"))
    b = rv.summary_text(rv.build_summary(movement_json(), source="m"))
    assert a == b


def test_caveats_from_the_numbers():
    s = rv.build_summary(movement_json(), shuttle_accuracy_pct=73)
    text = "\n".join(s["caveats"])
    assert "没有比分" in text
    assert "样本小" in text and "3 个回合" in text
    assert "小李 在回合内只有 80% 的时间被跟踪到" in text
    assert "小王 在回合内只有" not in text
    assert "小王 12 帧" in text and "小李 0 帧" not in text
    assert "分不出回合和死球" in text  # 70 % vs 68 %
    assert "正确率约 73%" in text
    assert "远端球员离相机远" in text  # no calibration given


def test_shuttle_check_that_separates_is_not_flagged():
    mv = movement_json()
    mv["shuttle_check"]["dead_shuttle_moving_pct"] = 10.0
    assert "分不出" not in "\n".join(rv.build_summary(mv)["caveats"])


def test_precision_and_camera_from_calibration(camera):
    s = rv.build_summary(movement_json(), calib=calibration(camera))
    near, far = (p["position_sigma"] for p in s["players"])
    assert far["along_m"] > 3 * near["along_m"] > 0  # far from a 3 m camera: much worse along the court
    assert far["along_m"] > far["across_m"]
    assert abs(s["camera"]["height_m"] - 3.0) < 0.1
    text = "\n".join(s["caveats"])
    assert f"沿球场方向 {far['along_m']:g} 米" in text
    assert "低于建议" not in text  # 3 m is fine


def test_low_camera_caveat():
    from conftest import Camera

    s = rv.build_summary(movement_json(), calib=calibration(Camera(pos=(3.05, -3.0, 1.4))))
    assert any("低于建议的 2.5–4 米" in c for c in s["caveats"])


def test_heatmap_centroid():
    assert rv.heatmap_centroid(movement_json()["players"][1]["heatmap"]) == pytest.approx((3.05, 11.725))
    assert rv.heatmap_centroid({}) is None


def test_shuttle_stats(tmp_path):
    csv_path = tmp_path / "s.csv"
    rows = ["Frame,Visibility,X,Y"] + [f"{i},{0 if 20 <= i < 30 else 1},{i},{i}" for i in range(100)]
    csv_path.write_text("\n".join(rows))
    rallies = [{"start_s": 0.0, "end_s": 3.9}, {"start_s": 5.0, "end_s": 9.0}, {"start_s": 50.0, "end_s": 60.0}]
    st = rv.shuttle_stats(csv_path, 10.0, rallies)
    assert st["detected_pct"] == 90.0 and st["longest_gap_s"] == 1.0
    assert st["per_rally"][0] == {"detected_pct": 75.0, "longest_gap_s": 1.0}
    assert st["per_rally"][1]["detected_pct"] == 100.0 and st["per_rally"][2] is None
    mv = movement_json()
    mv["rallies"] = rallies
    s = rv.build_summary(mv, shuttle=st)
    assert s["rallies"][0]["shuttle_detected_pct"] == 75.0 and "per_rally" not in s["shuttle"]
    assert "球被检测到 75%" in rv.summary_text(s)


def test_extras_are_compacted():
    extra = {"pairs": [{"spacing_m": {"mean": 2.123456}, "open_court": {"heatmap": {"counts": [[1]]}},
                        "swap_times_s": list(range(100))}]}
    s = rv.build_summary(movement_json(), extras={"positioning": extra})
    text = rv.summary_text(s)
    assert "【其他分析：positioning】" in text and "2.12" in text
    assert "heatmap" not in text and "已省略" in text


def test_prompt_sections_rules_and_caveats():
    s = rv.build_summary(movement_json(), source="match")
    system, user = rv.build_prompt(s)
    for section in rv.SECTIONS:
        assert f"## {section}" in user
    assert "正好三条" in user and "简体中文" in system
    assert "不要写出任何一次具体的击球" in system and "比分" in system
    assert "编号和时间" in system
    assert "【数据质量】" in user
    for c in s["caveats"]:
        assert c in user
    assert "第 2 回合 30.0–40.0 秒，长 10.0 秒，之前停了 5.0 秒" in user
    assert "和上一次比较" not in user
    assert "counts" not in user and "x_edges" not in user


def test_previous_session_comparison():
    prev_mv = movement_json()
    prev_mv["players"][0]["distance_m"] = 60.0  # 60 m/min then, 90 m/min now
    prev = rv.build_summary(prev_mv, source="上周")
    cur = rv.build_summary(movement_json(), source="本周")
    _, user = rv.build_prompt(cur, json.loads(json.dumps(prev)))  # as loaded from a file
    assert "【上一次】上周" in user and "## 和上一次比较" in user
    assert "每分钟跑动 90.0 米（上次 60.0，+30.0）" in user


def test_follow_up_reuses_context():
    s = rv.build_summary(movement_json())
    system, msgs = rv.build_messages(s, report="## 比赛概况\n旧点评", question="谁跑得多？")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == rv.build_prompt(s)[1] and "旧点评" in msgs[1]["content"]
    assert "追问：谁跑得多？" in msgs[2]["content"]
    _, msgs = rv.build_messages(s, question="谁跑得多？")  # no report yet: data + question in one turn
    assert len(msgs) == 1 and "【球员】" in msgs[0]["content"] and "追问：谁跑得多？" in msgs[0]["content"]
    assert "之前的点评" in rv.prompt_file_text(system, rv.build_messages(s, report="x", question="q")[1])


# --- backend ---------------------------------------------------------------------------


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeClient:
    def __init__(self, text="## 比赛概况\n好", stop_reason="end_turn"):
        self.calls = []
        self.reply = SimpleNamespace(stop_reason=stop_reason, content=[
            SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)])
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kw):
        self.calls.append(kw)
        return FakeStream(self.reply)


def test_call_anthropic_sends_text_only():
    client = FakeClient()
    system, msgs = rv.build_messages(rv.build_summary(movement_json()))
    assert rv.call_anthropic(system, msgs, client=client) == "## 比赛概况\n好"
    (kw,) = client.calls
    assert kw["model"] == "claude-sonnet-5-5" and kw["system"] == system and kw["messages"] == msgs
    assert all(isinstance(m["content"], str) for m in kw["messages"])  # no image / video blocks
    assert rv.call_anthropic(system, msgs, model="claude-opus-5-5", client=FakeClient()) and True


def test_call_anthropic_refusal_and_empty():
    with pytest.raises(rv.ReviewError):
        rv.call_anthropic("s", [], client=FakeClient(stop_reason="refusal"))
    with pytest.raises(rv.ReviewError):
        rv.call_anthropic("s", [], client=FakeClient(text=""))


def write_inputs(tmp_path):
    p = tmp_path / "match.movement.json"
    p.write_text(json.dumps(movement_json(), ensure_ascii=False))
    for i in (1, 2):
        (tmp_path / f"match.movement.{i}.png").write_bytes(b"\x89PNG fake")
    return p


def test_cli_without_key_writes_prompt_file(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = write_inputs(tmp_path)
    assert main(["review", str(p)]) == 0
    prompt = (tmp_path / "match.review.prompt.txt").read_text()
    assert "## 三个练习建议" in prompt and "【数据质量】" in prompt and "你是一位羽毛球教练" in prompt
    summary = json.loads((tmp_path / "match.review.summary.json").read_text())
    assert summary["schema"] == 1 and "heatmap" not in json.dumps(summary)
    assert not (tmp_path / "match.review.md").exists()
    # what the web site needs to ask a model itself: the same text, no key, nothing but text
    request = json.loads((tmp_path / "match.review.request.json").read_text())
    assert request["model"] == rv.DEFAULT_MODEL and request["system"] + "\n\n" + request["user"] + "\n" == prompt
    assert "{question}" in request["ask"] and request["title"] == "match"
    out = capsys.readouterr().out
    assert "没有联网" in out and "--response" in out and "--api" in out


def fake_anthropic(monkeypatch, client):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda: client))


def test_cli_with_key_calls_backend_and_writes_reports(tmp_path, monkeypatch, capsys):
    body = "\n\n".join(f"## {s}\n内容 **重点**\n1. 第一\n2. 第二" for s in rv.SECTIONS)
    client = FakeClient(text=body)
    fake_anthropic(monkeypatch, client)
    p = write_inputs(tmp_path)
    assert main(["review", str(p), "--model", "claude-haiku-5-5", "--api"]) == 0
    assert client.calls[0]["model"] == "claude-haiku-5-5"
    md = (tmp_path / "match.review.md").read_text()
    assert "由 claude-haiku-5-5 通过 Anthropic API 生成" in md and "## 三条观察" in md
    page = (tmp_path / "match.review.html").read_text()
    assert page.count("data:image/png;base64,") == 2 and 'class="cards"' in page
    assert "<h2>三条观察</h2>" in page and "<strong>重点</strong>" in page and "<ol>" in page
    assert "缺少这些小节" not in capsys.readouterr().out

    # follow-up: same context, earlier report as the assistant turn
    assert main(["review", str(p), "--ask", "下次先练什么？", "--api"]) == 0
    msgs = client.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"] and "下次先练什么？" in msgs[2]["content"]
    assert msgs[0]["content"] == client.calls[0]["messages"][0]["content"]
    assert "下次先练什么？" in (tmp_path / "match.review.qa.md").read_text()


def test_cli_ask_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = write_inputs(tmp_path)
    assert main(["review", str(p), "--ask", "谁更累？"]) == 0
    assert "追问：谁更累？" in (tmp_path / "match.review.ask.prompt.txt").read_text()


def test_cli_response_import_and_previous(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = write_inputs(tmp_path)
    assert main(["review", str(p), "-o", str(tmp_path / "old")]) == 0
    reply = tmp_path / "reply.md"
    reply.write_text("## 比赛概况\n只有一节")
    prev = tmp_path / "old" / "match.review.summary.json"
    assert main(["review", str(p), "--response", str(reply), "--previous", str(prev)]) == 0
    assert "从 reply.md 导入" in (tmp_path / "match.review.md").read_text()
    assert "【和上一次比较】" in (tmp_path / "match.review.prompt.txt").read_text()
    assert "缺少这些小节：球员跑动画像" in capsys.readouterr().out


def test_cli_rejects_non_movement_input(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    assert main(["review", str(p)]) == 1


def test_markdown_to_html_escapes():
    page = rv.markdown_to_html("# 标题\n\n- a <b>\n- b\n\n> 引用\n\n正文")
    assert "<h1>标题</h1>" in page and "<li>a &lt;b&gt;</li>" in page and "<blockquote>引用</blockquote>" in page
    assert page.count("<ul>") == 1 and "<p>正文</p>" in page


def test_key_alone_does_not_call_the_api(tmp_path, monkeypatch, capsys):
    client = FakeClient(text="## 比赛概况\n好")
    fake_anthropic(monkeypatch, client)
    p = write_inputs(tmp_path)
    assert main(["review", str(p)]) == 0
    assert main(["review", str(p), "--ask", "谁更累？"]) == 0
    assert client.calls == [] and not (tmp_path / "match.review.md").exists()
    assert "已设置" in capsys.readouterr().out


def test_api_without_key_is_an_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = write_inputs(tmp_path)
    assert main(["review", str(p), "--api"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_review_help_works(capsys):
    with pytest.raises(SystemExit) as e:
        main(["review", "--help"])
    assert e.value.code == 0 and "--shuttle-accuracy" in capsys.readouterr().out


def test_imported_answer_cannot_inject_html(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = write_inputs(tmp_path)
    reply = tmp_path / "reply.md"
    reply.write_text("## 比赛概况\n<script>alert(1)</script> **<img src=x onerror=alert(1)>**\n"
                     "- `<iframe>`\n> <b>x</b>\n### <svg onload=alert(1)>")
    assert main(["review", str(p), "--response", str(reply)]) == 0
    page = (tmp_path / "match.review.html").read_text()
    body = page.split("<body>", 1)[1]
    for bad in ("<script", "<img src=x", "<iframe", "<b>", "<svg"):
        assert bad not in body
    assert "&lt;script&gt;" in body


def test_compact_keeps_file_names_not_folders():
    out = rv.compact({"video": "/Users/someone/clips/match.mp4", "court": "C:\\x\\m.court.json",
                      "rel": "../data/a.json", "ratio": "1/2", "label": "前/后"})
    assert out == {"video": "match.mp4", "court": "m.court.json", "rel": "a.json", "ratio": "1/2", "label": "前/后"}


def test_exact_calibration_has_no_reprojection_claim(camera):
    calib = calibration(camera)  # markers projected exactly, as an automatic calibration saves them
    assert calib.reprojection_error_px < 0.05
    s = rv.build_summary(movement_json(), calib=calib)
    assert "reprojection_error_px" not in s["camera"] and "重投影" not in rv.summary_text(s)
    noisy = CourtCalibration({n: (x + (3 if i % 2 else -3), y) for i, (n, (x, y)) in enumerate(calib.image_points.items())},
                             calib.frame_size)
    s = rv.build_summary(movement_json(), calib=noisy)
    assert s["camera"]["reprojection_error_px"] > 0 and "重投影" in rv.summary_text(s)
