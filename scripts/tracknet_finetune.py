"""Fine-tune TrackNetV3's TrackNet from the released checkpoint on a Mac, without editing TrackNetV3.

Run it with TrackNetV3's Python:

    ~/codebase/third_party/TrackNetV3/.venv/bin/python scripts/tracknet_finetune.py \
        --data data/tracknet_finetune --save-dir data/tracknet_exp \
        --tracknet-dir ~/codebase/third_party/TrackNetV3 --epochs 10

It runs TrackNetV3's own train.py unchanged, after preparing four things that
train.py cannot do by itself (see docs/tracknet_finetune.md):

  1. train.py only knows "resume": it continues <save_dir>/TrackNet_cur.pt. So
     the released ckpts/TrackNet_best.pt is copied there with the epoch counter
     reset, the fields train.py expects but the released file lacks added
     (max_val_acc, scheduler, mask_ratio), and the learning rate, batch size and save_dir of this run written into
     it (train.py takes those from the checkpoint, not from the command line).
     An existing TrackNet_cur.pt is left alone: running again continues it.
  2. train.py reads the dataset from ./data: the script runs it in
     <save_dir>/run, where "data" is a link to --data.
  3. train.py and test.py call .cuda() everywhere: here .cuda() moves to
     --device instead (default: Apple GPU "mps" if there is one).
  4. train.py imports TensorBoard; if it is not installed, logging is skipped.

--epochs 0 goes through all of this (dataset, model, checkpoint, optimizer)
and stops before the first training step: a dry run.
"""

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory written by `badminton annotate-export`")
    p.add_argument("--save-dir", required=True, help="where checkpoints go (TrackNet_best.pt, TrackNet_cur.pt)")
    p.add_argument("--tracknet-dir", default=os.environ.get("TRACKNETV3_DIR"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4, help="released model was trained with 1e-3; lower for fine-tuning")
    p.add_argument("--batch-size", type=int, default=8, help="also the number of data loader processes (train.py)")
    p.add_argument("--device", help="mps, cuda or cpu; default: cuda, else mps, else cpu")
    a = p.parse_args()
    if not a.tracknet_dir:
        raise SystemExit("--tracknet-dir or TRACKNETV3_DIR is needed")
    tracknet = Path(a.tracknet_dir).expanduser().resolve()
    data = Path(a.data).resolve()
    save = Path(a.save_dir).resolve()
    for split in ("train", "val"):
        if not (data / split).is_dir() or not any((data / split).glob("match*")):
            raise SystemExit(f"{data / split} is missing or empty: train.py needs a train and a val split "
                             "(label at least two rallies, then badminton annotate-export)")

    import torch

    device = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 1. the released checkpoint as the state to "resume" from
    save.mkdir(parents=True, exist_ok=True)
    cur = save / "TrackNet_cur.pt"
    if cur.exists():
        print(f"{cur} exists: continuing that run (delete it to start again from the released checkpoint)")
    else:
        ckpt = torch.load(tracknet / "ckpts" / "TrackNet_best.pt", map_location="cpu")
        ckpt["epoch"] = -1  # train.py starts at epoch + 1
        ckpt["max_val_acc"] = 0.0  # so the best epoch ON OUR validation rallies is kept as TrackNet_best.pt
        ckpt.setdefault("scheduler", None)
        ckpt["param_dict"].update(save_dir=str(save), batch_size=a.batch_size, learning_rate=a.lr)
        ckpt["param_dict"].setdefault("mask_ratio", 0.3)  # InpaintNet only, but train.py reads the key
        for group in ckpt["optimizer"]["param_groups"]:
            group["lr"] = a.lr
        torch.save(ckpt, cur)
        print(f"start from {tracknet / 'ckpts' / 'TrackNet_best.pt'} (lr {a.lr}, batch {a.batch_size})")

    # 2. ./data
    run = save / "run"
    run.mkdir(exist_ok=True)
    link = run / "data"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(data, target_is_directory=True)
    os.chdir(run)
    # dataset.py caches its index in the data directory, with the frame paths as seen from
    # the directory it ran in; an index left by another run or an older export would be wrong
    for pattern in ("img_config_*.npz", "data_l*_s*_*.npz"):
        for f in data.glob(pattern):
            f.unlink()

    # 3. .cuda() -> the chosen device
    if device != "cuda":
        torch.Tensor.cuda = lambda self, *args, **kw: self.to(device)
        torch.nn.Module.cuda = lambda self, *args, **kw: self.to(device)

    # 4. TensorBoard is optional
    try:
        import torch.utils.tensorboard  # noqa: F401
    except ImportError:
        import types

        class _NoWriter:
            def __init__(self, *args, **kw):
                print("TensorBoard is not installed: no training curves (pip install tensorboard to get them)")

            def __getattr__(self, name):
                return lambda *args, **kw: None

        stub = types.ModuleType("torch.utils.tensorboard")
        stub.SummaryWriter = _NoWriter
        sys.modules["torch.utils.tensorboard"] = stub

    sys.path.insert(0, str(tracknet))
    sys.argv = ["train.py", "--model_name", "TrackNet", "--epochs", str(a.epochs), "--save_dir", str(save),
                "--resume_training", "--verbose"]
    print(f"device {device}; running {tracknet / 'train.py'} {' '.join(sys.argv[1:])}")
    runpy.run_path(str(tracknet / "train.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
