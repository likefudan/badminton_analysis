"""Load an exported dataset with TrackNetV3's own Dataset class and check what comes out.

Run it with TrackNetV3's Python (it needs torch, pandas, parse), not ours:

    ~/codebase/third_party/TrackNetV3/.venv/bin/python scripts/check_tracknet_dataset.py \
        data/tracknet_finetune --tracknet-dir ~/codebase/third_party/TrackNetV3

For both splits it builds Shuttlecock_Trajectory_Dataset the way train.py does
(seq_len 8, bg_mode "concat"; sliding step 1 for train, 8 for val) and checks:
the number of windows against manifest.json, the tensor shapes, and that the
heatmap dataset.py draws sits on the labelled point. dataset.py caches its
index next to the data (img_config_*.npz, data_l*_*.npz); the caches are
removed again unless --keep-cache is given, so a later export is not read
through a stale index.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SEQ = 8


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--tracknet-dir", default=os.environ.get("TRACKNETV3_DIR"))
    p.add_argument("--keep-cache", action="store_true")
    a = p.parse_args()
    if not a.tracknet_dir:
        raise SystemExit("--tracknet-dir or TRACKNETV3_DIR is needed")
    sys.path.insert(0, str(Path(a.tracknet_dir).expanduser()))
    from dataset import Shuttlecock_Trajectory_Dataset  # noqa: E402  (TrackNetV3's, not ours)

    root = str(Path(a.data_dir))
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    segments = [s for m in manifest["matches"] for s in m["segments"]]
    ok = True
    try:
        for split, step in (("train", 1), ("val", SEQ)):
            segs = [s for s in segments if s["split"] == split]
            if not segs:
                print(f"{split}: empty (train.py needs both splits)")
                ok = False
                continue
            ds = Shuttlecock_Trajectory_Dataset(root_dir=root, split=split, seq_len=SEQ, sliding_step=step,
                                                data_mode="heatmap", bg_mode="concat")
            expect = sum(len(range(0, s["frames"] - SEQ + 1, step)) for s in segs)
            good = len(ds) == expect
            worst = 0.0
            visible = 0
            for i in np.linspace(0, len(ds) - 1, min(len(ds), 40)).astype(int):
                _, frames, heatmaps, coor, vis = ds[int(i)]
                good &= frames.shape == ((SEQ + 1) * 3, 288, 512) and heatmaps.shape == (SEQ, 288, 512)
                good &= 0.0 <= frames.min() and frames.max() <= 1.0 and frames.max() > 0.2
                for k in range(SEQ):
                    ys, xs = np.nonzero(heatmaps[k])
                    if vis[k]:
                        visible += 1
                        if not len(xs):
                            good = False
                            continue
                        # coor is normalised to 0..1; dataset.py draws the disc at int(pixel)
                        worst = max(worst, abs(xs.mean() - coor[k][0] * 512), abs(ys.mean() - coor[k][1] * 288))
                    else:
                        good &= len(xs) == 0 and coor[k][0] == 0 and coor[k][1] == 0
            good &= worst <= 1.5
            ok &= bool(good)
            print(f"{split}: {len(segs)} rallies, {sum(s['frames'] for s in segs)} frames -> {len(ds)} windows "
                  f"(expected {expect}); frames {tuple(frames.shape)}, heatmaps {tuple(heatmaps.shape)}; "
                  f"{visible} visible labels checked, heatmap centre within {worst:.2f} px of the label: "
                  f"{'OK' if good else 'FAILED'}")
    finally:
        if not a.keep_cache:
            for pattern in ("img_config_*.npz", "data_l*_s*_*.npz"):
                for f in Path(root).glob(pattern):
                    f.unlink()
    print("dataset.py reads the export" if ok else "PROBLEMS, see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
