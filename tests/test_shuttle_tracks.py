import numpy as np

from tests.test_shuttle_path import _flight


def _scene(fps: float = 30.0):
    """A low camera beside the court (like the phone clips) and its court calibration."""
    from tests.conftest import Camera
    from badminton.court import MARKERS, CourtCalibration

    w, h = 1920, 1080
    cam = Camera(pos=(8.5, -1.5, 1.9), target=(2.0, 8.0, 1.5), f=900.0, size=(w, h))
    names = ["near_left_corner", "near_right_corner", "far_left_corner", "far_right_corner",
             "near_short_service_left", "far_short_service_right"]
    calib = CourtCalibration({n: tuple(p) for n, p in zip(names, cam.project(np.array([MARKERS[n] for n in names])))},
                             (w, h))
    return cam, calib, (w, h)


def _peaks(n, flights, size, noise=0.7, seed=0):
    """Per frame heatmap peaks (x, y, peak) in heatmap pixels from [(first frame, pixels, peak)]."""
    w, h = size
    rng = np.random.default_rng(seed)
    out = [[] for _ in range(n)]
    for f0, px, pk in flights:
        for i, (x, y) in enumerate(px):
            if 0 <= x < w and 0 <= y < h and f0 + i < n:
                x, y = np.array([x, y]) + rng.normal(0, noise, 2)
                out[f0 + i].append((x * 512 / w, y * 288 / h, pk))
    return [np.array(sorted(o, key=lambda r: -r[2]), float).reshape(-1, 3) for o in out]


def test_link_follows_two_crossing_flights():
    from badminton.shuttle_tracks import link

    t = np.arange(40.0)
    a = np.c_[100 + 20 * t, 500 - 30 * t + 0.8 * t**2]
    b = np.c_[900 - 20 * t, 400 - 25 * t + 0.8 * t**2]
    cands = [np.array([[*a[i], 0.8], [*b[i], 0.7]]) for i in range(40)]
    tracks = [tr for tr in link(cands, 1920, 30.0) if len(tr) >= 5]
    assert len(tracks) == 2
    for tr in tracks:
        assert len(tr) == 40
        dx = np.diff(tr[:, 2])
        assert (dx > 0).all() or (dx < 0).all()             # stays on the same flight through the crossing


def test_flight_fit_tells_this_court_from_the_next():
    from badminton.shuttle_tracks import FlightFit

    fps = 30.0
    cam, calib, size = _scene(fps)
    ours = cam.project(_flight([3.0, 2.0, 2.4], [0.5, 11.0, 9.0], 1.0, fps))
    theirs = cam.project(_flight([-8.0, 2.0, 2.4], [0.5, 11.0, 9.0], 1.0, fps))
    tracks = [np.c_[np.arange(len(p)), np.zeros(len(p)), p, np.ones(len(p))] for p in (ours, theirs)]
    (r1, in1, d1), (r2, in2, d2) = FlightFit(calib, fps).fit(tracks)
    assert r1 < 1.0 and r2 < 1.0          # both fit within a pixel
    assert in1 > 0.9 and in2 < 0.1
    assert d2 > d1


def test_path_keeps_this_courts_flight_over_a_brighter_one_next_door():
    """Both courts play at once and the next court's shuttle lights up the heatmap more: choosing by
    flights keeps this court's shuttle. When only the next court plays, nothing is reported."""
    from badminton.shuttle_tracks import rows_from_peaks

    fps = 30.0
    cam, calib, size = _scene(fps)
    ours = cam.project(_flight([3.0, 2.0, 2.4], [0.5, 11.0, 9.0], 1.6, fps))
    theirs = cam.project(_flight([-8.0, 2.0, 2.4], [0.5, 11.0, 9.0], 1.6, fps))
    later = cam.project(_flight([-8.0, 12.0, 2.4], [0.5, -11.0, 9.0], 1.6, fps))
    n = 150
    peaks = _peaks(n, [(5, ours, 0.55), (5, theirs, 0.9), (90, later, 0.9)], size)
    rows = rows_from_peaks(peaks, size, calib, fps=fps)
    seen = np.array([r[1] for r in rows])
    xy = np.array([r[2:] for r in rows], float)
    m = len(ours)
    on_ours = seen[5:5 + m].astype(bool) & (np.hypot(*(xy[5:5 + m] - ours).T) < 6)
    assert on_ours.mean() > 0.9
    assert seen[90:].sum() < 5                              # only the next court is playing



def test_next_courts_rally_between_two_of_ours_is_dropped():
    """This court waits for a serve while a court further back rallies, and the path joins that rally
    to ours into one stretch: the part between our flights is dropped when it holds enough flights
    surely above another court; a short excursion and our own flights stay."""
    from badminton.shuttle_tracks import drop_away_chains

    def track(f0, n, x0):
        f = np.arange(f0, f0 + n, dtype=float)
        return np.c_[f, np.zeros(n), x0 + 10 * (f - f0), 500 + 0 * f, np.full(n, 0.8)]

    tracks = [track(0, 40, 100), track(40, 20, 600), track(60, 20, 900), track(80, 10, 1300), track(90, 40, 1500)]
    weights = np.array([1.0, 0.15, 0.15, 0.15, 1.0])
    sure = np.array([False, True, True, False, False])
    rows = [(int(f), 1, int(x), int(y)) for t in tracks for f, _, x, y, _ in t]
    out = drop_away_chains(rows, tracks, weights, sure=sure)
    seen = np.array([r[1] for r in out])
    assert seen[:40].all() and seen[90:].all()                 # ours
    assert not seen[40:90].any()                               # 40 points surely elsewhere in between
    out = drop_away_chains(rows[:60] + [(f, 1, 0, 0) for f in range(60, 90)] + rows[90:], tracks, weights, sure=sure)
    assert np.array([r[1] for r in out])[40:60].all()          # only 20 points elsewhere: kept


def test_flight_never_near_our_players_hands_is_dropped():
    """With this court's players known, a flight that never comes within FAR_FROM_HANDS body heights
    of any of their wrists is another court's, even where its 3D fit cannot tell."""
    from badminton.shuttle_tracks import hand_distance, rows_from_peaks

    fps = 30.0
    cam, calib, size = _scene(fps)
    ours = cam.project(_flight([3.0, 2.0, 2.4], [0.5, 11.0, 9.0], 1.4, fps))
    n = len(ours) + 10
    peaks = _peaks(n, [(0, ours, 0.7)], size)

    def hands_at(xy):
        wr = np.full((n, 2, 2), np.nan)
        wr[:, 0] = xy
        return wr, np.full((n, 2), 400.0)

    near, far = hands_at(ours[0] + [30, 40]), hands_at([100.0, 1000.0])
    d = {}
    seen = np.array([r[1] for r in rows_from_peaks(peaks, size, calib, fps=fps, hands=near, debug=d)])
    assert seen[3:len(ours) - 3].mean() > 0.9
    assert hand_distance(d["tracks"][:1], near)[0] < 0.2 and hand_distance(d["tracks"][:1], far)[0] > 1.5
    seen = np.array([r[1] for r in rows_from_peaks(peaks, size, calib, fps=fps, hands=far)])
    assert seen.sum() == 0
    none = (np.full((n, 2, 2), np.nan), np.full((n, 2), np.nan))           # nobody seen: no judgement
    seen = np.array([r[1] for r in rows_from_peaks(peaks, size, calib, fps=fps, hands=none)])
    assert seen[3:len(ours) - 3].mean() > 0.9
