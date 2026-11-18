"""Analysis region drawn on the web page for a fixed camera: parts of the frame to look at only
("keep") or to ignore ("drop"). Polygons are in normalised frame coordinates (0..1, x right, y down),
so they fit any resolution of the same recording.

A point counts when it is inside some keep polygon (or there is none) and inside no drop polygon.
The shuttle candidates and the players' feet are tested against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_POLYGONS = 12
MAX_POINTS = 64


def _clean_polygon(raw) -> np.ndarray | None:
    try:
        pts = np.asarray(raw, float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(pts) < 3 or len(pts) > MAX_POINTS or not np.isfinite(pts).all():
        return None
    return np.clip(pts, 0.0, 1.0)


def inside(poly: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Even-odd ray casting, vectorised over the points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.zeros(np.broadcast(x, y).shape, bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if y0 == y1:
            continue
        crosses = (y0 > y) != (y1 > y)
        xs = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        out ^= crosses & (x < xs)
    return out


@dataclass
class Region:
    keep: list  # polygons (k, 2) normalised
    drop: list

    @classmethod
    def from_dict(cls, raw) -> "Region | None":
        """{"keep": [[[x, y], ...], ...], "drop": [...]}; None when nothing usable is in it."""
        if not isinstance(raw, dict):
            return None
        keep = [p for p in (_clean_polygon(r) for r in (raw.get("keep") or [])[:MAX_POLYGONS]) if p is not None]
        drop = [p for p in (_clean_polygon(r) for r in (raw.get("drop") or [])[:MAX_POLYGONS]) if p is not None]
        return cls(keep, drop) if keep or drop else None

    @classmethod
    def load(cls, path: str | Path | None) -> "Region | None":
        if not path:
            return None
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {"keep": [np.round(p, 5).tolist() for p in self.keep],
                "drop": [np.round(p, 5).tolist() for p in self.drop]}

    def contains(self, u, v) -> np.ndarray:
        """u, v: normalised coordinates (arrays). -> bool array."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        ok = np.ones(np.broadcast(u, v).shape, bool)
        if self.keep:
            ok &= np.logical_or.reduce([inside(p, u, v) for p in self.keep])
        for p in self.drop:
            ok &= ~inside(p, u, v)
        return ok

    def contains_px(self, x, y, size: tuple[float, float]) -> np.ndarray:
        """x, y in pixels of a frame of size (width, height)."""
        return self.contains(np.asarray(x, float) / size[0], np.asarray(y, float) / size[1])

    def box_filter(self, size: tuple[float, float]):
        """Box filter for the pose estimator: keep a person box whose bottom centre (the feet) is in the region."""

        def keep(boxes: np.ndarray) -> np.ndarray:
            boxes = np.asarray(boxes, float).reshape(-1, 4)
            return self.contains_px((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3], size)

        return keep

    def filter_points(self, per_frame: list, size: tuple[float, float]) -> list:
        """Per-frame candidate arrays (x, y, ...) in pixels of `size`: drop the rows outside."""
        out = []
        for c in per_frame:
            c = np.asarray(c)
            if c.ndim == 2 and len(c):
                c = c[self.contains_px(c[:, 0], c[:, 1], size)]
            out.append(c)
        return out


def combine_filters(*filters):
    """Box filter that keeps a box only when every given filter (None = no opinion) keeps it."""
    filters = [f for f in filters if f is not None]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]

    def keep(boxes: np.ndarray) -> np.ndarray:
        out = np.ones(len(boxes), bool)
        for f in filters:
            out &= f(boxes)
        return out

    return keep
