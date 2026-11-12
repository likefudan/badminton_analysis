"""Cut the dead time out of a match video, using the rally segmentation.

Two outputs: the whole match with only the rallies left, and a highlight reel
of the longest rallies. Rally starts from `movement.py` are 1-2 s late (the
serve itself is slow, so the motion-based detector only fires once players
start running), hence the default lead-in.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PRE_S = 2.0        # lead-in before each rally: covers the late rally start
POST_S = 1.0       # tail after each rally: the shuttle landing and the reaction
MERGE_GAP_S = 1.0  # padded clips closer than this are joined instead of jump-cut


@dataclass
class Clip:
    start_s: float
    end_s: float
    # (rally number counted from 1, rally length in seconds, time the label appears)
    rallies: list = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def plan_clips(rallies: list[dict], duration_s: float | None = None, pre_s: float = PRE_S,
               post_s: float = POST_S, merge_gap_s: float = MERGE_GAP_S, numbers: list[int] | None = None) -> list[Clip]:
    """Pad each rally and join clips that touch. `numbers` are the rally numbers
    to show (default 1..N); pass them when `rallies` is a subset."""
    numbers = numbers or list(range(1, len(rallies) + 1))
    items = sorted(zip(rallies, numbers), key=lambda x: x[0]["start_s"])
    clips: list[Clip] = []
    for r, n in items:
        a = max(0.0, r["start_s"] - pre_s)
        b = r["end_s"] + post_s
        if duration_s is not None:
            b = min(b, duration_s)
        if b <= a:
            continue
        length = r["end_s"] - r["start_s"]
        if clips and a - clips[-1].end_s < merge_gap_s:
            c = clips[-1]
            # the label switches where the new rally's lead-in would have started
            c.rallies.append((n, length, max(a, c.rallies[-1][2])))
            c.end_s = max(c.end_s, b)
        else:
            clips.append(Clip(a, b, [(n, length, a)]))
    return clips


def longest(rallies: list[dict], top: int) -> list[int]:
    """Indices of the `top` longest rallies, in match order."""
    order = sorted(range(len(rallies)), key=lambda i: rallies[i]["start_s"] - rallies[i]["end_s"])
    return sorted(order[:top])


def label_text(number: int, length_s: float, total: int, zh: bool = True) -> str:
    return f"回合 {number}/{total} · {length_s:.0f} 秒" if zh else f"Rally {number}/{total} · {length_s:.0f} s"


def render_label(text: str, path: Path, video_height: int) -> Path:
    """Text on a translucent dark pill, as a PNG for ffmpeg's overlay filter.
    (Homebrew's ffmpeg has no drawtext, and OpenCV cannot draw Chinese.)"""
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    from .card import _CJK_FONTS

    size = max(16, round(video_height * 0.035))
    font = None
    known = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in known:
            try:
                font = ImageFont.truetype(known[name], size)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    x0, y0, x1, y1 = font.getbbox(text)
    pad = round(size * 0.45)
    img = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=pad, fill=(0, 0, 0, 150))
    d.text((pad - x0, pad - y0), text, font=font, fill=(255, 255, 255, 255))
    img.save(path)
    return path


def has_cjk_font() -> bool:
    from matplotlib import font_manager

    from .card import _CJK_FONTS

    return bool({f.name for f in font_manager.fontManager.ttflist} & set(_CJK_FONTS))


def ffmpeg_command(video: Path, out: Path, clips: list[Clip], labels: dict[int, Path], has_audio: bool,
                   margin: int, encoder: str = "libx264") -> list[str]:
    """One ffmpeg run: every clip is its own seeked input (frame-accurate since
    we re-encode), labels are overlaid, then everything is concatenated."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for c in clips:
        cmd += ["-ss", f"{c.start_s:.3f}", "-t", f"{c.duration_s:.3f}", "-i", str(video)]
    label_input: dict[int, int] = {}
    for n, png in labels.items():
        label_input[n] = len(clips) + len(label_input)
        cmd += ["-i", str(png)]
    parts = []
    for i, c in enumerate(clips):
        # timestamps restart at 0 for each input; normalise so concat is seamless
        cur = f"v{i}_0"
        parts.append(f"[{i}:v:0]setpts=PTS-STARTPTS,format=yuv420p[{cur}]")
        shown = [r for r in c.rallies if r[0] in label_input]
        for k, (n, _, at) in enumerate(shown):
            t0 = at - c.start_s
            t1 = shown[k + 1][2] - c.start_s if k + 1 < len(shown) else c.duration_s + 1
            nxt = f"v{i}_{k + 1}"
            parts.append(f"[{cur}][{label_input[n]}:v]overlay={margin}:{margin}:enable='between(t,{t0:.3f},{t1:.3f})'[{nxt}]")
            cur = nxt
        parts.append(f"[{cur}]null[v{i}]")
        if has_audio:
            parts.append(f"[{i}:a:0]asetpts=PTS-STARTPTS,aresample=async=1[a{i}]")
    streams = "".join(f"[v{i}]" + (f"[a{i}]" if has_audio else "") for i in range(len(clips)))
    parts.append(f"{streams}concat=n={len(clips)}:v=1:a={1 if has_audio else 0}[v]" + ("[a]" if has_audio else ""))
    cmd += ["-filter_complex", ";".join(parts), "-map", "[v]"]
    if has_audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", encoder]
    cmd += ["-b:v", "8M"] if "videotoolbox" in encoder else ["-preset", "veryfast", "-crf", "20"]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    return cmd


def cut_video(video: Path, out: Path, clips: list[Clip], total_rallies: int, label: bool = True,
              encoder: str = "libx264") -> None:
    from .video import probe

    if not shutil.which("ffmpeg"):
        raise RuntimeError("剪辑需要 ffmpeg（brew install ffmpeg）")
    if not clips:
        raise ValueError("没有可剪的回合")
    info = probe(video)
    with tempfile.TemporaryDirectory() as tmp:
        labels: dict[int, Path] = {}
        if label:
            zh = has_cjk_font()
            for c in clips:
                for n, length, _ in c.rallies:
                    labels[n] = render_label(label_text(n, length, total_rallies, zh), Path(tmp) / f"{n}.png", info.height)
        cmd = ffmpeg_command(video, out, clips, labels, info.has_audio, margin=round(info.height * 0.03), encoder=encoder)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg 剪辑失败：{r.stderr.strip()[-800:]}")


def cut_match(video: Path, rallies: list[dict], out_dir: Path, pre_s: float = PRE_S, post_s: float = POST_S,
              merge_gap_s: float = MERGE_GAP_S, top: int = 3, label: bool = True, encoder: str = "libx264") -> dict:
    """Write <stem>.rallies.mp4 and <stem>.longest.mp4; returns what was cut."""
    from .video import probe

    info = probe(video)
    duration = info.frames / info.fps if info.frames and info.fps else None
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    all_clips = plan_clips(rallies, duration, pre_s, post_s, merge_gap_s)
    out = out_dir / f"{video.stem}.rallies.mp4"
    cut_video(video, out, all_clips, len(rallies), label, encoder)
    result["rallies"] = {"path": out, "clips": all_clips, "duration_s": sum(c.duration_s for c in all_clips),
                         "source_s": duration}
    if top > 0:
        idx = longest(rallies, top)
        clips = plan_clips([rallies[i] for i in idx], duration, pre_s, post_s, merge_gap_s, numbers=[i + 1 for i in idx])
        out = out_dir / f"{video.stem}.longest.mp4"
        cut_video(video, out, clips, len(rallies), label, encoder)
        result["longest"] = {"path": out, "clips": clips, "duration_s": sum(c.duration_s for c in clips),
                             "numbers": [i + 1 for i in idx]}
    return result
