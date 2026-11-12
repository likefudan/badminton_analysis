"""A synthetic camera behind the near baseline, like the recommended iPhone setup."""

import numpy as np
import pytest

W, H = 1280, 720


class Camera:
    def __init__(self, pos=(3.05, -3.0, 3.0), target=(3.05, 7.5, 0.0), f=700.0, size=(W, H)):
        self.size = size
        self.K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
        c = np.asarray(pos, float)
        fwd = np.asarray(target, float) - c
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])  # world -> camera
        self.c = c

    def project(self, xyz) -> np.ndarray:
        p = np.atleast_2d(np.asarray(xyz, float))
        if p.shape[1] == 2:
            p = np.hstack([p, np.zeros((len(p), 1))])
        cam = (p - self.c) @ self.R.T
        uv = cam @ self.K.T
        return uv[:, :2] / uv[:, 2:3]


@pytest.fixture
def camera():
    return Camera()


def fake_person(camera: Camera, x: float, y: float, height: float = 1.75, score: float = 0.9):
    """17 COCO keypoints for an upright person standing at court point (x, y)."""
    # (dx across, z up) of each keypoint for a person facing the camera
    layout = {
        0: (0, 1.62), 1: (-0.03, 1.66), 2: (0.03, 1.66), 3: (-0.07, 1.63), 4: (0.07, 1.63),
        5: (-0.2, 1.42), 6: (0.2, 1.42), 7: (-0.28, 1.12), 8: (0.28, 1.12), 9: (-0.3, 0.85),
        10: (0.3, 0.85), 11: (-0.13, 0.95), 12: (0.13, 0.95), 13: (-0.14, 0.5), 14: (0.14, 0.5),
        15: (-0.15, 0.07), 16: (0.15, 0.07),
    }
    s = height / 1.75
    pts3 = np.array([(x + dx * s, y, z * s if j not in (15, 16) else 0.0) for j, (dx, z) in layout.items()])
    kp = camera.project(pts3)
    x0, y0 = kp.min(0) - 8
    x1, y1 = kp.max(0) + 8
    return np.array([x0, y0, x1, y1]), kp, np.full(17, score)


def marker_pixels(camera: Camera, names, noise_px: float = 0.0, seed: int = 0):
    from badminton.court import MARKERS

    rng = np.random.default_rng(seed)
    out = {}
    for n in names:
        uv = camera.project([MARKERS[n]])[0] + rng.normal(0, noise_px, 2)
        out[n] = (float(uv[0]), float(uv[1]))
    return out
