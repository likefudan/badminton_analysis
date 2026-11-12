"""Full camera (intrinsics + pose) from the court homography.

The court calibration is a floor homography: it maps image points to floor
points but says nothing about height. Assuming square pixels, no skew and the
principal point at the image centre (true enough for phone video), the
homography also fixes the focal length (the floor's two axes must be
orthogonal and equally scaled after removing K) and the camera pose, which
lets us project 3D points such as the net tape or a shuttle 5 m above the
court.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .court import CourtCalibration


@dataclass
class Camera:
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3) world -> camera
    t: np.ndarray  # (3,)
    size: tuple[int, int]  # (width, height)

    @property
    def center(self) -> np.ndarray:
        """Camera position in court coordinates (metres; z up)."""
        return -self.R.T @ self.t

    @property
    def focal_px(self) -> float:
        return float(self.K[0, 0])

    def project(self, xyz) -> np.ndarray:
        p = np.atleast_2d(np.asarray(xyz, float))
        cam = p @ self.R.T + self.t
        uv = cam @ self.K.T
        return uv[:, :2] / uv[:, 2:3]

    def depth(self, xyz) -> np.ndarray:
        p = np.atleast_2d(np.asarray(xyz, float))
        return (p @ self.R.T + self.t)[:, 2]


def camera_from_calibration(calib: CourtCalibration) -> Camera:
    w, h = calib.frame_size
    cx, cy = w / 2, h / 2
    Hc = np.linalg.inv(calib.H)  # court (x, y, 1) -> image
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    G = T @ Hc  # principal point at the origin
    h1, h2 = G[:, 0], G[:, 1]
    # With K = diag(f, f, 1) and w = 1 / f^2, the floor axes K^-1 h1, K^-1 h2 must be
    #   orthogonal:    (h1x h2x + h1y h2y) w + h1z h2z = 0
    #   equally long:  (h1x^2 + h1y^2 - h2x^2 - h2y^2) w + (h1z^2 - h2z^2) = 0
    # Either equation alone degenerates for some views (e.g. looking straight down
    # the court makes h1z ~ 0), so both are solved together by least squares; a
    # degenerate equation has near-zero coefficients and so carries little weight.
    G = G / np.linalg.norm(G[:, :2])
    h1, h2 = G[:, 0], G[:, 1]
    rows = np.array([[h1[0] * h2[0] + h1[1] * h2[1], h1[2] * h2[2]],
                     [h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2, h1[2] ** 2 - h2[2] ** 2]])
    a_, b_ = rows[:, 0], rows[:, 1]
    w_ = -float(a_ @ b_) / float(a_ @ a_)
    if not np.isfinite(w_) or w_ <= 0:
        raise ValueError("无法从球场标定推出焦距（画面可能几乎正对地面）")
    f = float(1 / np.sqrt(w_))
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ Hc
    lam = (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1])) / 2
    r1, r2, t = M[:, 0] / lam, M[:, 1] / lam, M[:, 2] / lam
    if t[2] < 0:  # the court must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    U, _, Vt = np.linalg.svd(np.stack([r1, r2, r3], 1))
    R = U @ Vt
    cam = Camera(K, R, t, (w, h))
    if cam.center[2] < 0:  # court axes (x across, y along) give z = x * y; make z point up
        R = R @ np.diag([1, 1, -1.0])
        cam = Camera(K, R, t, (w, h))
    return cam
