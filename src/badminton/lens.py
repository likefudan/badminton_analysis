"""Radial lens distortion (ultra-wide lenses bend straight lines).

A homography only describes a pinhole camera. A phone's 0.5x lens is not one:
straight court lines come out bowed, more so towards the edges of the picture,
and a homography fitted to them is a compromise that is wrong by several
pixels at the near corners. The lens model removes the bend first; everything
else (court homography, camera pose) then works on "straightened" pixels.

Model, in the direction that is needed most often (picture -> straightened):

    u = c + (p - c) * (1 + k1 r^2 + k2 r^4),   r = |p - c| / R

with c the picture centre and R half the picture diagonal, so r = 1 in the
corners. Because r is relative to the picture size, the same numbers describe
the video at any resolution (4K or downscaled to 1080p), but not a crop.
k1 > 0 is barrel distortion (the usual ultra-wide look: lines bow outwards);
k1 tells how much further out a corner pixel really belongs, e.g. 0.05 = 5 %.

The other direction (straightened -> picture) has no closed form and is solved
by Newton's method. Outside the picture (r > R_MAX) the curve is continued as
a straight line so that the mapping stays one-to-one for court points far
outside the frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

R_MAX = 1.2  # the polynomial is used up to here (1 = picture corner)
_MIN_SLOPE = 0.2


@dataclass(frozen=True)
class Lens:
    k1: float = 0.0
    k2: float = 0.0
    _r_lim: float = field(init=False, repr=False, compare=False, default=R_MAX)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.k1) and np.isfinite(self.k2)):
            raise ValueError("镜头畸变参数必须是有限的数")
        r = np.linspace(0, R_MAX, 241)
        bad = np.flatnonzero(self._slope(r) < _MIN_SLOPE)
        lim = float(r[bad[0]]) if len(bad) else R_MAX
        if lim < 1.0:
            raise ValueError(f"镜头畸变参数 k1={self.k1:g}, k2={self.k2:g} 太大：画面内的点会折叠到一起")
        object.__setattr__(self, "_r_lim", lim)

    @property
    def is_identity(self) -> bool:
        return self.k1 == 0 and self.k2 == 0

    def _poly(self, r):
        r2 = r * r
        return r * (1 + self.k1 * r2 + self.k2 * r2 * r2)

    def _slope(self, r):
        r2 = r * r
        return 1 + 3 * self.k1 * r2 + 5 * self.k2 * r2 * r2

    def _g(self, r: np.ndarray) -> np.ndarray:
        """Straightened radius of picture radius r."""
        lim = self._r_lim
        return np.where(r <= lim, self._poly(np.minimum(r, lim)), self._poly(lim) + self._slope(lim) * (r - lim))

    def _g_inv(self, rho: np.ndarray) -> np.ndarray:
        lim = self._r_lim
        g_lim, s_lim = self._poly(lim), self._slope(lim)
        r = np.clip(rho, 0, lim)
        for _ in range(25):  # the curve is monotone on [0, lim], so clipped Newton steps converge
            r = np.clip(r - (self._poly(r) - rho) / self._slope(r), 0, lim)
        return np.where(rho <= g_lim, r, lim + (rho - g_lim) / s_lim)

    def _radial(self, pts, size, fn) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.is_identity or len(p) == 0:
            return p.copy()
        w, h = size
        c = np.array([w / 2, h / 2])
        R = float(np.hypot(w, h)) / 2
        d = p - c
        r = np.linalg.norm(d, axis=1) / R
        with np.errstate(invalid="ignore", divide="ignore"):
            gain = np.where(r > 1e-12, fn(r) / r, 1.0)
        return c + d * gain[:, None]

    def undistort(self, pts, size: tuple[int, int]) -> np.ndarray:
        """Picture pixels -> straightened pixels. `size` is (width, height) of the picture."""
        return self._radial(pts, size, self._g)

    def distort(self, pts, size: tuple[int, int]) -> np.ndarray:
        """Straightened pixels -> picture pixels."""
        return self._radial(pts, size, self._g_inv)

    def to_json(self) -> dict:
        return {"model": "radial", "k1": self.k1, "k2": self.k2}

    @classmethod
    def from_json(cls, data: dict | None) -> "Lens | None":
        if not data:
            return None
        if data.get("model", "radial") != "radial":
            raise ValueError(f"不认识的镜头模型: {data.get('model')}")
        return cls(float(data.get("k1", 0.0)), float(data.get("k2", 0.0)))


def parse_lens(spec: str) -> Lens | None:
    """'0.05' or '0.05,-0.01' -> Lens; 'none' -> None."""
    if spec.strip().lower() in ("none", "off", "0", ""):
        return None
    try:
        vals = [float(v) for v in spec.split(",")]
    except ValueError:
        raise ValueError(f"镜头参数应为 k1 或 k1,k2（两个数），得到 {spec!r}") from None
    if len(vals) > 2:
        raise ValueError(f"镜头参数应为 k1 或 k1,k2（两个数），得到 {spec!r}")
    lens = Lens(*vals)
    return None if lens.is_identity else lens


def rectify(frame: np.ndarray, lens: Lens | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """The picture with the lens distortion removed, the matrix A that maps its
    pixels to straightened pixels of the original (u = A q), and which of its
    pixels show a part of the original (None = all of them).

    Straightening a barrel-distorted picture pushes the corners outwards. The
    result keeps the scale of the picture centre and is made larger instead
    (up to 1.5 times; A is the shift of its origin), so the whole picture
    stays in view and no detail is lost.
    """
    if lens is None or lens.is_identity:
        return frame, np.eye(3), None
    h, w = frame.shape[:2]
    r = np.linspace(0.05, 1.0, 96)
    s = min(1.5, max(1.0, float((lens._g(r) / r).max())))
    W, H = int(np.ceil(w * s)), int(np.ceil(h * s))
    A = np.array([[1, 0, (w - W) / 2], [0, 1, (h - H) / 2], [0, 0, 1.0]])
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    q = np.stack([gx.ravel(), gy.ravel()], 1)
    src = lens.distort(q + A[:2, 2], (w, h)).astype(np.float32)
    out = cv2.remap(frame, src[:, 0].reshape(H, W), src[:, 1].reshape(H, W), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = (src[:, 0] >= 0) & (src[:, 0] <= w - 1) & (src[:, 1] >= 0) & (src[:, 1] <= h - 1)
    return out, A, inside.reshape(H, W)
