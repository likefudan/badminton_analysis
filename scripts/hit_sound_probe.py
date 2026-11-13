"""Does the hit sound help? Measures how often a sound onset coincides with a hit.

    python scripts/hit_sound_probe.py VIDEO --hits VIDEO.hits.csv [--rallies VIDEO.movement.json]

A racket meeting a shuttle makes a short broadband click. Per video frame this
script computes how strongly the sound in the 2-9 kHz band rises within a few
milliseconds (1.0 = the band's energy rose by 9 dB between two spectrum frames
8 ms apart), then compares the frames of the hits (any CSV with a `frame` or
`Frame` column: detected hits or reviewed labels) with random frames in play.

Result on the two phone recordings of 2026-11 (docs/hits.md): 20-23 % of the
hits have a clear onset within one frame, against 10-11 % of random moments;
about three quarters of the onsets during play belong to no hit (neighbouring
courts, footsteps, shouts). Hit sound was therefore left out of `badminton
hits`. Run this again when reviewed labels or cleaner recordings exist.

Only ffmpeg and numpy are used.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SR = 22050
BAND_HZ = (2000.0, 9000.0)
SUBFRAMES = 4  # spectrum frames per video frame
CLEAR_DB = 9.0  # the band's energy rising this much between two spectrum frames scores 1.0


def load_audio(video: str | Path, sr: int = SR, start_s: float = 0.0) -> np.ndarray:
    """Mono float32 samples of the first decodable audio stream; empty if there is none."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_s}", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "s16le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return np.zeros(0, np.float32)
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0


def band_flux(x: np.ndarray, sr: int, hop: int, n_fft: int = 512, band=BAND_HZ) -> np.ndarray:
    """Rise of the log energy in `band` from one spectrum frame to the next (>= 0)."""
    if len(x) < n_fft:
        return np.zeros(0)
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    e = np.log(spec[:, (freqs >= band[0]) & (freqs <= band[1])].sum(axis=1) + 1e-10)
    flux = np.zeros(n)
    flux[1:] = np.maximum(0.0, np.diff(e))
    return flux


def onset_strength(x: np.ndarray, sr: int, fps: float, n_frames: int) -> np.ndarray:
    """(n_frames,) onset strength per video frame; 1.0 = clear onset, 0 = nothing.

    The flux is a ratio of energies (a difference of logs), so it does not depend
    on how loud the recording is: 1.0 means the band's energy rose by CLEAR_DB
    from one spectrum frame (about 8 ms) to the next."""
    out = np.zeros(n_frames)
    hop = max(1, int(round(sr / fps / SUBFRAMES)))
    flux = band_flux(x, sr, hop)
    if len(flux) == 0:
        return out
    z = flux / (CLEAR_DB * np.log(10) / 10)
    t = (np.arange(len(flux)) * hop + 256) / sr  # centre of each spectrum frame, seconds
    f = np.clip((t * fps).astype(int), 0, n_frames - 1)
    np.maximum.at(out, f, z)
    return out


def video_onsets(video: str | Path, fps: float, n_frames: int) -> np.ndarray | None:
    """Onset strength for a video file, or None when it has no sound."""
    x = load_audio(video)
    if len(x) < SR // 10:
        return None
    return onset_strength(x, SR, fps, n_frames)


def main() -> int:
    import argparse
    import csv
    import json

    import cv2

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--hits", required=True, help="CSV with a frame / Frame column")
    ap.add_argument("--rallies", help="movement.json: random frames are drawn from its rallies only")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    on = video_onsets(a.video, fps, n)
    if on is None:
        raise SystemExit("这个视频没有声音")
    with open(a.hits, newline="") as f:
        rows = list(csv.DictReader(f))
    frames = [int(r.get("frame") or r["Frame"]) for r in rows if (r.get("Note") or "") != "ignore"]
    frames = [f for f in frames if f < n]
    play = np.ones(n, bool)
    if a.rallies:
        play[:] = False
        for r in json.loads(Path(a.rallies).read_text())["rallies"]:
            play[max(0, int((r["start_s"] - 2) * fps)):int((r["end_s"] + 1) * fps)] = True
    rand = np.random.default_rng(0).choice(np.flatnonzero(play), 5000)

    def near(i: int, w: int) -> float:
        return float(on[max(0, i - w):i + w + 1].max())

    print(f"{len(frames)} 次击球；回合内 {int(play.sum())} 帧")
    print("声音起点强度  窗口    击球处有起点  随机时刻有起点")
    for thr in (0.75, 1.0, 1.5):
        for w in (1, 3):
            hit = np.mean([near(i, w) >= thr for i in frames])
            chance = np.mean([near(int(i), w) >= thr for i in rand])
            print(f"  >= {thr:<4}      ±{w} 帧   {100 * hit:5.1f}%        {100 * chance:5.1f}%")
    peaks: list[int] = []
    for i in np.argsort(-on):
        if on[i] < 1.0:
            break
        if play[i] and all(abs(i - p) > 6 for p in peaks):
            peaks.append(int(i))
    matched = sum(any(abs(p - f) <= 3 for f in frames) for p in peaks)
    print(f"回合内强度 >= 1 的声音起点 {len(peaks)} 个，其中 {matched} 个在某次击球的 ±3 帧内")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
