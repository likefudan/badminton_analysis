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
