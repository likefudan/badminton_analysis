import shutil
import subprocess

import pytest

from badminton.cut import Clip, cut_match, ffmpeg_command, longest, plan_clips

RALLIES = [{"start_s": 7.4, "end_s": 35.4}, {"start_s": 38.8, "end_s": 48.6}, {"start_s": 51.5, "end_s": 72.7},
           {"start_s": 93.2, "end_s": 129.4}]


def test_padding_is_clamped_to_the_video():
    clips = plan_clips([{"start_s": 1.0, "end_s": 5.0}, {"start_s": 20.0, "end_s": 29.5}], duration_s=30.0)
    assert [(c.start_s, c.end_s) for c in clips] == [(0.0, 6.0), (18.0, 30.0)]


def test_clips_that_touch_after_padding_are_joined():
    clips = plan_clips(RALLIES, duration_s=160.0, pre_s=2.0, post_s=1.0, merge_gap_s=1.0)
    # 35.4+1 -> 38.8-2 leaves 0.4 s, 48.6+1 -> 51.5-2 overlaps: one clip; 72.7 -> 93.2 is a real gap
    assert [(c.start_s, c.end_s) for c in clips] == [(5.4, 73.7), (91.2, 130.4)]
    assert [n for n, _, _ in clips[0].rallies] == [1, 2, 3]
    # labels change in order and never before the previous one
    times = [at for _, _, at in clips[0].rallies]
    assert times == sorted(times) and times[0] == 5.4
    # the short gaps are kept, so nothing is shown twice and nothing jumps
    assert sum(c.duration_s for c in clips) == pytest.approx(68.3 + 39.2)


def test_no_merge_when_gap_is_long_enough():
    clips = plan_clips(RALLIES, pre_s=0.5, post_s=0.5, merge_gap_s=1.0)
    assert len(clips) == 4


def test_longest_keeps_match_order_and_numbers():
    idx = longest(RALLIES, 2)
    assert idx == [0, 3]
    clips = plan_clips([RALLIES[i] for i in idx], numbers=[i + 1 for i in idx])
    assert [c.rallies[0][0] for c in clips] == [1, 4]
    assert longest(RALLIES, 10) == [0, 1, 2, 3]


def test_ffmpeg_command_without_audio_or_labels():
    cmd = ffmpeg_command("in.mp4", "out.mp4", [Clip(1.0, 3.0, [(1, 2.0, 1.0)]), Clip(5.0, 6.0, [(2, 1.0, 5.0)])],
                         {}, has_audio=False, margin=10)
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0[v]" in fc and "overlay" not in fc and "[a]" not in cmd
    assert cmd.count("-ss") == 2


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
@pytest.mark.parametrize("audio", [True, False])
def test_cut_real_video(tmp_path, audio):
    src = tmp_path / "match.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=20"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=20", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    rallies = [{"start_s": 3.0, "end_s": 6.0}, {"start_s": 12.0, "end_s": 17.0}]
    res = cut_match(src, rallies, tmp_path / "out", top=1)

    def dur(p, stream):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", "stream=duration",
                              "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
        return float(out) if out else None

    assert dur(res["rallies"]["path"], "v:0") == pytest.approx(6 + 8, abs=0.15)   # (1..7) + (10..18)
    assert dur(res["longest"]["path"], "v:0") == pytest.approx(8, abs=0.15)
    assert res["longest"]["numbers"] == [2]
    a = dur(res["rallies"]["path"], "a:0")
    assert (a == pytest.approx(14, abs=0.2)) if audio else (a is None)
