import json
import shutil
import subprocess

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def _streams(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


def _clip(path, rotation=None, audio_tracks=0):
    """1 s test clip; `rotation` stored as display metadata (like an iPhone), `audio_tracks` AAC tracks."""
    plain = path.with_name("plain_" + path.name)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1"]
    for _ in range(audio_tracks):
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
    cmd += ["-map", "0:v"] + sum((["-map", f"{k + 1}:a"] for k in range(audio_tracks)), [])
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(plain)]
    subprocess.run(cmd, check=True)
    rot = [] if rotation is None else ["-display_rotation", str(rotation)]
    subprocess.run(["ffmpeg", "-y", "-v", "error", *rot, "-i", str(plain), "-map", "0", "-c", "copy", str(path)], check=True)
    return path


def test_writer_takes_one_decodable_audio_track(tmp_path, monkeypatch):
    """IMG_1069: a second, undecodable audio track (iPhone spatial audio) made the skeleton video fail."""
    from badminton import video
    from badminton.video import VideoWriter

    src = _clip(tmp_path / "two_audio.mp4", audio_tracks=2)
    out = tmp_path / "out.mp4"
    with VideoWriter(out, 64, 48, 10, audio_from=src) as wr:
        for _ in range(10):
            wr.write(np.zeros((48, 64, 3), np.uint8))
    assert [s["codec_type"] for s in _streams(out)].count("audio") == 1
    monkeypatch.setattr(video, "UNDECODABLE_AUDIO", {"aac"})  # nothing decodable: no audio, still a video
    with VideoWriter(out, 64, 48, 10, audio_from=src) as wr:
        for _ in range(10):
            wr.write(np.zeros((48, 64, 3), np.uint8))
    assert [s["codec_type"] for s in _streams(out)] == ["video"]


@pytest.mark.parametrize("rotation, expected", [(None, ""), (180, "hflip,vflip"), (-180, "hflip,vflip"),
                                                (-90, "transpose=clock"), (90, "transpose=cclock")])
def test_upright_filter_follows_the_stored_rotation(tmp_path, rotation, expected):
    from badminton.video import upright_filter

    assert upright_filter(_clip(tmp_path / "c.mp4", rotation)) == expected
