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


def test_speed_unit_and_frame_step():
    from badminton.shuttle_path import rows_from_candidates, speed_unit

    assert speed_unit(1920, 29.97) == 1.0
    assert speed_unit(3840, 59.94) == 1.0  # twice the pixels, half the time
    assert speed_unit(3840, 59.94, frame_step=2) == 2.0

    # 150 video px between processed frames: too fast to follow by the 1080p/30 fps rule,
    # normal when the frames are every second one of a 4K 60 fps video
    cands = [np.array([[20 + 20 * f, 100, 0.9, 9, 9]], np.float32) for f in range(24)]
    assert sum(r[1] for r in rows_from_candidates(cands, (3840, 2160), fps=59.94)) == 0
    rows = rows_from_candidates(cands, (3840, 2160), fps=59.94, frame_step=2, n_frames=47)
    assert [r[0] for r in rows] == list(range(47)) and all(r[1] for r in rows)
    assert rows[1][2] == round((20 * 7.5 + 40 * 7.5) / 2)  # skipped frame: halfway


def test_far_roi_is_the_far_part_of_the_court():
    from badminton.camera import Camera
    from badminton.court import LENGTH, WIDTH, CourtCalibration
    from badminton.shuttle_path import far_roi

    # a 4K camera 2 m behind the baseline, 2.5 m up, looking down the court; focal 1400 px (0.5x phone lens)
    w, h, f = 3840, 2160, 1400.0
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    tilt = np.radians(12)
    R = np.array([[1, 0, 0], [0, -np.sin(tilt), -np.cos(tilt)], [0, np.cos(tilt), -np.sin(tilt)]])  # world -> camera
    cam = Camera(K, R, -R @ np.array([WIDTH / 2, -2.0, 2.5]), (w, h))
    corners = {"near_left_corner": (0, 0), "near_right_corner": (WIDTH, 0), "far_left_corner": (0, LENGTH),
               "far_right_corner": (WIDTH, LENGTH)}
    pts = {k: tuple(cam.project([x, y, 0.0])[0]) for k, (x, y) in corners.items()}
    calib = CourtCalibration(pts, (w, h))
    roi = far_roi(calib, (w, h))
    x0, y0, x1, y1 = roi
    # shuttle smaller than 2 model px beyond 1400 * 0.065 / (2 * 7.5) = 6.1 m: the far half and more, but
    # not the floor at the camera's feet, and narrower than the frame
    assert y0 == 0 and y1 < h * 0.8 and x0 > w * 0.1 and x1 < w * 0.9
    far_net = cam.project([WIDTH / 2, LENGTH, 1.5])[0]
    assert x0 < far_net[0] < x1 and y0 < far_net[1] < y1
    # with a long lens the shuttle is large everywhere: nothing to tile
    assert far_roi(calib, (w, h), min_px=0.2) is None


def test_still_stretches_of_the_path_are_dropped():
    """A rally, then the path sits on a shuttle lying on the next court's floor (jitters by a pixel
    or two, one frame missing), then creeps slowly, then another rally."""
    from badminton.shuttle_path import drop_still_runs

    p = np.zeros((80, 3))
    for f in range(0, 20):                       # flying: 30 px a frame
        p[f] = (1, 100 + 30 * f, 500 - 10 * f)
    for f in range(20, 40):                      # lying still
        if f != 27:
            p[f] = (1, 772 + f % 2, 1024 - f % 3)
    for f in range(40, 52):                      # creeping 2.5 px a frame: no block of 8 leaves a 19 px box
        p[f] = (1, 774 + 2.5 * (f - 40), 1024)
    for f in range(52, 58):                      # six slow frames at the top of a clear are kept
        p[f] = (1, 1500 + 2 * (f - 52), 300)
    for f in range(58, 80):
        p[f] = (1, 1510 + 25 * (f - 58), 300 + 12 * (f - 58))
    q = drop_still_runs(p, 19.0)
    assert q[:20, 0].all() and not q[20:52, 0].any() and q[52:, 0].all()
    np.testing.assert_array_equal(q[:20], p[:20])
    # consecutive blocks only: the creeping part leaves a short tail behind
    blocks = drop_still_runs(p, 19.0, overlap=False)
    assert not blocks[20:40, 0].any() and 0 < blocks[40:52, 0].sum() < 12
    assert drop_still_runs(p, 19.0, min_frames=40)[:, 0].sum() == p[:, 0].sum()   # nothing that long
