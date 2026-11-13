"""Where a number came from. Four kinds, used in every data file:

  measured      实测  a model saw it in this very frame (pose keypoint, shuttle blob)
  interpolated  插值  filled in across a short loss (airborne frames, InpaintNet,
                      skipped frames between two processed ones)
  derived       推算  computed from geometry or motion (speed, distance, hit time,
                      hitter, landing point)
  manual        人工  entered or confirmed by a person

Files keep the English word; `LABEL_ZH` is for anything shown to people. The AI
review must not draw firm conclusions from interpolated or derived numbers.
"""

from __future__ import annotations

MEASURED = "measured"
INTERPOLATED = "interpolated"
DERIVED = "derived"
MANUAL = "manual"
ALL = (MEASURED, INTERPOLATED, DERIVED, MANUAL)

LABEL_ZH = {MEASURED: "实测", INTERPOLATED: "插值", DERIVED: "推算", MANUAL: "人工"}

MEANING_ZH = {
    MEASURED: "模型在这一帧直接检测到的",
    INTERPOLATED: "短暂丢失后补出来的",
    DERIVED: "用几何或运动规律算出来的",
    MANUAL: "人工标注或确认的",
}


def legend() -> dict:
    """Block to embed in a JSON file so it explains its own source tags."""
    return {k: f"{LABEL_ZH[k]}：{MEANING_ZH[k]}" for k in ALL}


def tagged(value, source: str) -> dict:
    if source not in ALL:
        raise ValueError(f"unknown source {source!r}")
    return {"value": value, "source": source}


def share(sources) -> dict:
    """Fraction of each kind in a sequence of source tags ("" = no value); for summaries."""
    items = [s for s in sources if s]
    n = len(items)
    return {k: round(sum(s == k for s in items) / n, 4) if n else 0.0 for k in ALL}
