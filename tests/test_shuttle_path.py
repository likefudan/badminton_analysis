import numpy as np

from badminton.shuttle_path import choose_path, fill_short_gaps, static_map


def test_path_ignores_static_spot_and_alternating_distractor():
    """A shuttle flies across; a wall light sits still with a higher score; a second
    object pops up every other frame with a bigger blob. The path follows the shuttle."""
    F = 60
    cands = []
    for f in range(F):
        c = [(100 + 4 * f, 50 + 0.05 * (f - 30) ** 2, 0.7, 20, 20),  # shuttle (heatmap px)
             (400, 30, 0.9, 30, 30)]  # static light
        if f % 2:
            c.append((200, 200, 0.55, 60, 60))  # flickering distractor
        cands.append(np.array(c, np.float32))
    st = static_map(cands)
    assert st[30 // 8, 400 // 8]
    path = choose_path(cands, (1.0, 1.0), st)
    assert path[:, 0].mean() > 0.95
    np.testing.assert_allclose(path[5:55, 1], 100 + 4 * np.arange(5, 55), atol=1e-3)


def test_short_gaps_are_filled_long_ones_not():
    p = np.zeros((12, 3))
    p[0] = (1, 0, 0); p[3] = (1, 30, 0); p[10] = (1, 100, 0)
    q = fill_short_gaps(p, 3)
    assert q[1, 0] == 1 and abs(q[2, 1] - 20) < 1e-9
    assert q[5, 0] == 0


def test_overlay_command_writes_video(tmp_path):
    import cv2

    from badminton.cli import main

    vid = tmp_path / "clip.skeleton.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(20):
        w.write(np.zeros((240, 320, 3), np.uint8))
    w.release()
    csv = tmp_path / "clip.shuttle.csv"
    csv.write_text("Frame,Visibility,X,Y\n" + "".join(f"{i},1,{10 + 5 * i},100\n" for i in range(20)))
    assert main(["overlay", str(vid), "--shuttle", str(csv)]) == 0
    assert (tmp_path / "clip.combined.mp4").exists()
