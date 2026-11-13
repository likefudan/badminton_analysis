import pytest
import numpy as np

from badminton.shuttle import _WindowEnsemble, ensemble_weight, inpaint_mask, largest_blob_center


def reference_ensemble(window_outputs: np.ndarray, n: int) -> np.ndarray:
    """Numpy port of TrackNetV3 predict.py's temporal ensemble (eval_mode 'weight'),
    keeping its buffer/index arithmetic: window_outputs (W, n, ...) -> (W + n - 1, ...)."""
    W = len(window_outputs)
    shape = window_outputs.shape[2:]
    weight = ensemble_weight(n)
    batch_i, frame_i = np.arange(n), np.arange(n - 1, -1, -1)
    buf = np.concatenate([np.zeros((n - 1, n, *shape)), window_outputs])
    out = []
    for b in range(W):
        if b < n - 1:
            y = buf[batch_i + b, frame_i].sum(0) / (b + 1)
        else:
            y = (buf[batch_i + b, frame_i] * weight.reshape(-1, *[1] * len(shape))).sum(0)
        out.append(y)
    buf = np.concatenate([buf, np.zeros((n - 1, n, *shape))])
    b = W - 1
    for f in range(1, n):
        out.append(buf[batch_i + b + f, frame_i].sum(0) / (n - f))
    return np.array(out)


def test_window_ensemble_matches_reference():
    rng = np.random.default_rng(0)
    for n, W in ((8, 30), (16, 5), (8, 1)):
        outs = rng.random((W, n, 3, 4))
        ens = _WindowEnsemble(n)
        mine = [ens.push(o) for o in outs] + ens.flush()
        np.testing.assert_allclose(np.array(mine), reference_ensemble(outs, n), atol=1e-12)


def test_ensemble_weight():
    np.testing.assert_allclose(ensemble_weight(8) * 20, [1, 2, 3, 4, 4, 3, 2, 1])


def test_largest_blob_wins():
    h = np.zeros((40, 60))
    h[5:7, 5:7] = 0.9  # small blob
    h[20:26, 30:40] = 0.8  # big blob
    h[0, 59] = 0.4  # below threshold
    assert largest_blob_center(h) == (35, 23)
    assert largest_blob_center(np.zeros((10, 10))) is None


def test_inpaint_mask_fills_gaps_but_not_exits_at_the_top():
    vis = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    y = np.array([200, 210, 0, 0, 220, 15, 0, 0, 0, 230, 240])  # second gap starts at the top edge
    m = inpaint_mask(y, vis, th_h=30)
    assert m.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]


def test_expand_rows_interpolates_skipped_frames():
    from badminton.shuttle import expand_rows

    xs, ys, vis = np.array([10, 20, 0, 40]), np.array([100, 110, 0, 130]), np.array([1, 1, 0, 1])
    rows = expand_rows(xs, ys, vis, 2, 100)
    assert [r[0] for r in rows] == list(range(7))
    assert rows[1] == (1, 1, 15, 105)  # between two seen frames
    assert rows[3] == (3, 0, 0, 0) and rows[5] == (5, 0, 0, 0)  # next to a frame without shuttle
    assert rows[6] == (6, 1, 40, 130)


def test_load_models_from_a_folder_without_models(tmp_path):
    pytest.importorskip("coremltools")
    from badminton.shuttle import load_models

    with pytest.raises(SystemExit, match="tracknetv3_tracknet"):
        load_models(model_dir=tmp_path)


def test_csv_source_column(tmp_path):
    from badminton.shuttle import row_sources, write_csv

    rows = [(0, 1, 10, 20), (1, 1, 12, 22), (2, 0, 0, 0)]
    src = row_sources(rows, [True, False, False])
    assert src == ["measured", "interpolated", ""]
    write_csv(rows, tmp_path / "a.csv", src)
    assert (tmp_path / "a.csv").read_text().splitlines() == [
        "Frame,Visibility,X,Y,Source", "0,1,10,20,measured", "1,1,12,22,interpolated", "2,0,0,0,"]
    write_csv(rows, tmp_path / "b.csv")  # without sources: the old four columns
    assert (tmp_path / "b.csv").read_text().splitlines()[0] == "Frame,Visibility,X,Y"


def test_path_reports_which_rows_are_measured():
    import numpy as np

    from badminton.shuttle_path import rows_from_candidates

    cands = [np.array([[20.0 + 12 * f, 100.0 + 3 * f, 0.9]]) if f not in (4, 5) else np.zeros((0, 3)) for f in range(40)]
    measured: list = []
    rows = rows_from_candidates(cands, (1920, 1080), measured=measured)
    assert [r[1] for r in rows] == [1] * 40  # the two-frame gap is filled
    assert measured == [f not in (4, 5) for f in range(40)]


def test_tile_views_cover_the_region_with_overlap():
    from badminton.shuttle import tile_views

    assert tile_views((1920, 1080), 1) == [(0, 0, 1920, 1080)]
    views = tile_views((3840, 2160), 3)
    assert views[0] == (0, 0, 3840, 2160)
    tiles = views[1:]
    assert all(x1 - x0 == 1280 and y1 - y0 == 720 for x0, y0, x1, y1 in tiles)
    cover = np.zeros((2160, 3840), int)
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 and x1 <= 3840 and 0 <= y0 and y1 <= 2160
        cover[y0:y1, x0:x1] += 1
    assert cover.min() >= 1
    xs = sorted({t[0] for t in tiles})
    assert all(b - a <= 1280 * 0.875 + 1 for a, b in zip(xs[:-1], xs[1:]))  # neighbours overlap by 1/8 or more
    # a region smaller than one tile gets a single tile around it, kept inside the frame
    assert tile_views((3840, 2160), 3, roi=(3500, 100, 3800, 300))[1:] == [(2560, 0, 3840, 720)]
    # only the region is tiled
    part = tile_views((3840, 2160), 3, roi=(1000, 500, 3000, 1500))[1:]
    assert len(part) < len(tiles) and min(t[0] for t in part) == 1000 and max(t[2] for t in part) == 3000


def test_merge_view_candidates_maps_to_the_full_frame():
    from badminton.shuttle import merge_view_candidates

    size = (1920, 1080)
    views = [(0, 0, 1920, 1080), (0, 0, 960, 540), (896, 0, 1856, 540)]
    full = np.array([[100.0, 50.0, 0.6, 9, 9]], np.float32)  # at video (375, 187.5)
    tile = np.array([[200.0, 100.0, 0.9, 16, 16],  # the same object, seen larger: video (375, 187.5)
                     [508.0, 100.0, 0.9, 4, 4],  # at this tile's inner (right) border: cut off, dropped
                     [3.0, 283.0, 0.9, 4, 4],
                     [300.0, 200.0, 0.5, 4, 4]], np.float32)  # too weak for a tile  # the frame's left border is fine, but the tile's inner bottom border
    right = np.array([[40.0, 100.0, 0.8, 16, 16]], np.float32)  # video (971, 187.5), well inside the right tile
    out = merge_view_candidates([full, tile, right], views, size)
    np.testing.assert_allclose(out[:, :3], [[100, 50, 0.9], [971 / 3.75, 50, 0.8]], atol=1e-4)
    np.testing.assert_allclose(out[0, 3:], [4, 4])  # areas in full-frame heatmap pixels: 16 / 2^2


def test_step_ensemble():
    from badminton.shuttle import _StepEnsemble

    n, F = 8, 30
    truth = np.arange(F, dtype=float)
    for step in (1, 2, 4, 8, 3):
        ens = _StepEnsemble(n, step)
        got = []
        starts = list(range(0, F - n + 1, step))
        if starts[-1] != F - n:
            starts.append(F - n)
        for s in starts:
            got += ens.push(s, truth[s:s + n] + 100)  # every window agrees: the mean must return it
        got += ens.take(F)
        np.testing.assert_allclose(got, truth + 100)
    # two windows that disagree: weights 4:4 at the first window's frame 4 = the second's frame 0 -> 1/(4+1) ...
    ens = _StepEnsemble(8, 4)
    a = ens.push(0, np.zeros(8))
    b = ens.push(4, np.ones(8))
    w = ensemble_weight(8)
    assert a == [0, 0, 0, 0]
    np.testing.assert_allclose(b, [w[k] / (w[k] + w[k + 4]) for k in range(4)])


def test_resize_crops_and_prescales_large_frames():
    from badminton.shuttle import HEIGHT, WIDTH, _resize

    rng = np.random.default_rng(0)
    big = np.kron(rng.integers(0, 255, (54, 96, 3), dtype=np.uint8), np.ones((40, 40, 1), np.uint8))  # 3840x2160, blocky
    out = _resize(big)
    assert out.shape == (HEIGHT, WIDTH, 3)
    from PIL import Image

    ref = np.asarray(Image.fromarray(np.ascontiguousarray(big[..., ::-1])).resize((WIDTH, HEIGHT), Image.BICUBIC))
    assert np.abs(out.astype(int) - ref).mean() < 3  # prescaling changes little
    # a crop of exactly the model's size is passed through (BGR -> RGB only)
    crop = _resize(big, rect=(100, 200, 100 + WIDTH, 200 + HEIGHT))
    np.testing.assert_array_equal(crop, big[200:200 + HEIGHT, 100:100 + WIDTH, ::-1])
