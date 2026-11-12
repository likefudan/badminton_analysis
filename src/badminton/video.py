"""Video reading/writing and iPhone HDR handling (through ffmpeg)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ and HLG (iPhone records HLG / Dolby Vision)


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    codec: str = ""
    color_transfer: str = ""
    has_audio: bool = False

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS


def probe(path: str | Path) -> VideoInfo:
    """Video properties via ffprobe, falling back to OpenCV when ffprobe is missing."""
    path = str(path)
    if shutil.which("ffprobe"):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", path],
            capture_output=True, text=True, check=True,
        ).stdout
        streams = json.loads(out)["streams"]
        v = next(s for s in streams if s["codec_type"] == "video")
        num, den = (int(x) for x in v.get("avg_frame_rate", "0/1").split("/"))
        fps = num / den if den else 0.0
        frames = int(v.get("nb_frames") or 0)
        w, h = int(v["width"]), int(v["height"])
        # iPhone stores portrait/landscape as rotation metadata; OpenCV applies it.
        rot = 0
        for sd in v.get("side_data_list", []):
            rot = int(sd.get("rotation", rot) or 0)
        rot = int(v.get("tags", {}).get("rotate", rot) or 0)
        if abs(rot) % 180 == 90:
            w, h = h, w
        return VideoInfo(
            w, h, fps, frames, v.get("codec_name", ""), v.get("color_transfer", ""),
            any(s["codec_type"] == "audio" for s in streams),
        )
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    info = VideoInfo(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def _has_filter(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if len(line.split()) > 1)


UNDECODABLE_AUDIO = {"apple_apac"}  # iPhone spatial audio; ffmpeg cannot decode it


def _audio_map(src) -> list[str]:
    """-map for the first audio stream ffmpeg can decode (newer iPhones put spatial
    audio, apple_apac, first and plain AAC second); none if there is no such stream."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.split()
    for k, codec in enumerate(out):
        if codec not in UNDECODABLE_AUDIO:
            return ["-map", f"0:a:{k}"]
    return []


def to_sdr(src: str | Path, dst: str | Path, height: int | None = None) -> None:
    """Convert an HDR (HLG/PQ) iPhone clip to SDR BT.709 H.264 so colours are not washed out.

    Uses zscale + tonemap when ffmpeg is built with zimg; Homebrew's ffmpeg is
    not, so on a Mac it falls back to VideoToolbox (scale_vt converts the
    transfer function and primaries with Apple's own tone mapping).
    Optionally downscale to `height` (e.g. 1080 for 4K footage).
    """
    if _has_filter("zscale"):
        scale = f"zscale=w=-2:h={height}," if height else ""
        vf = (scale + "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
              "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
        pre = []
    elif _has_filter("scale_vt"):
        size = f"w=-2:h={height}:" if height else ""
        vf = (f"scale_vt={size}color_matrix=bt709:color_primaries=bt709:color_transfer=bt709,"
              "hwdownload,format=p010le,format=yuv420p")
        pre = ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"]
    else:
        raise RuntimeError("这个 ffmpeg 既没有 zscale 也没有 scale_vt，无法把 HDR 转成 SDR")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *pre, "-i", str(src), "-map", "0:v:0", *_audio_map(src), "-vf", vf,
         "-c:v", "libx264", "-crf", "18", "-preset", "fast",
         "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
         "-c:a", "aac", "-b:a", "160k", str(dst)],
        check=True,
    )


def read_frames(path: str | Path, start_frame: int = 0, max_frames: int | None = None):
    """Yield (frame_index, BGR frame)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    i = start_frame
    try:
        while max_frames is None or i - start_frame < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


def read_frame(path: str | Path, index: int = 0) -> np.ndarray:
    for _, f in read_frames(path, index, 1):
        return f
    raise ValueError(f"读不到第 {index} 帧: {path}")


class VideoWriter:
    """H.264 MP4 writer that pipes raw frames into ffmpeg (plays everywhere, incl. WeChat).

    If `audio_from` is given, that file's audio (from `audio_start_s`) is muxed in.
    Falls back to OpenCV's mp4v writer when ffmpeg is not installed.
    """

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        audio_from: str | Path | None = None,
        audio_start_s: float = 0.0,
        encoder: str = "libx264",
        crf: int = 20,
    ) -> None:
        self.path = str(path)
        self.size = (width, height)
        self._proc = None
        self._cv = None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-v", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-"]
            if audio_from:
                cmd += ["-ss", f"{audio_start_s}", "-i", str(audio_from), "-map", "0:v", "-map", "1:a?",
                        "-c:a", "aac", "-shortest"]
            if encoder == "libx264":
                cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
            else:  # e.g. h264_videotoolbox on a Mac, h264_nvenc on NVIDIA
                cmd += ["-c:v", encoder, "-b:v", "12M"]
            cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self._cv = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1::-1] != self.size:
            raise ValueError(f"帧尺寸 {frame.shape[1::-1]} 和输出尺寸 {self.size} 不一致")
        if self._proc:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self._cv.write(frame)

    def close(self) -> None:
        if self._proc:
            self._proc.stdin.close()
            if self._proc.wait() != 0:
                raise RuntimeError(f"ffmpeg 写入 {self.path} 失败")
            self._proc = None
        if self._cv:
            self._cv.release()
            self._cv = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
