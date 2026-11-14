# 用自己的录像微调 TrackNetV3

手机拍的视频上，TrackNetV3 发布的模型只有一半到七成的帧是对的（墙上的灯、地上的球、隔壁场地、手和球拍都会被当成球）。办法是标一些自己的录像，接着发布的权重继续训练。整个过程分四步：

1. `badminton annotate`：在浏览器里逐帧确认或点击球的位置。
2. `badminton annotate-export`：导出成 TrackNetV3 训练用的目录。
3. 用 TrackNetV3 自己的 `train.py` 训练（本文第 3 节）。
4. 把训练出的权重转成 Core ML，替换缓存里的模型（第 4 节）。

TrackNetV3 的代码和权重不在本仓库里，下面假设它在 `~/codebase/third_party/TrackNetV3`，并且有自己的虚拟环境 `.venv`。

## 1. 标注文件

`badminton annotate match.mp4` 在视频旁边写三个文件：

| 文件 | 内容 |
| --- | --- |
| `match.labels.csv` | 标注。列：`Frame,Visibility,X,Y,Status` |
| `match.labels.json` | 谁、什么时候标的，工具版本，回合列表（帧号），各状态的帧数，预填记录 |
| `match.annotate.npz` | 模型预测的缓存（原版逐帧选点、按轨迹选的点、每帧所有候选点），删掉会重新跑 |

`match.labels.csv` 的前四列就是 TrackNetV3 的标注格式：`Frame` 是视频帧号（从 0 开始），`Visibility` 是 1（看得见）或 0，`X,Y` 是原视频的像素坐标，看不见时是 `0,0`。只有标过的帧才有一行。第五列 `Status` 是本工具加的：

| Status | 含义 | 默认是否导出 |
| --- | --- | --- |
| `human` | 人逐帧确认或点击过 | 是 |
| `bulk` | 人用 Shift+A 一次接受的（模型有把握的帧） | 是 |
| `machine` | 模型的预测，没人看过（`--prefill` 写的） | 否 |
| `skip` | 被挡住或拿不准（按 O） | 从不导出，并且把回合从这里切开 |

没有 `Status` 列的 CSV（比如 `badminton shuttle` 的输出改个名）也能读，全部当作 `machine`。

「没有球」（N）和「拿不准」（O）的区别：球飞出画面、落地后不算在飞、确实不在画面里，按 N，训练时这一帧告诉模型「这里什么都不该有」；球应该在但被人挡住、糊成一片分不清，按 O，这一帧不参加训练。

## 2. 导出的目录格式

```bash
badminton annotate-export data/IMG_8923_stab.mp4 -o data/tracknet_finetune
```

可以一次给多个视频；以后再导出同一个视频会替换它原来的目录，编号不变。

```
data/tracknet_finetune/
  manifest.json                          导出清单（本工具写的，TrackNetV3 不读）
  train/
    match1/                              一个视频是一个 match，编号从 1 开始
      median.npz                         整段视频的背景图
      csv/
        r001_f001164_ball.csv            一段连续标注的标签
      frame/
        r001_f001164/
          0.png 1.png ... 292.png        这一段的每一帧
          median.npz                     这一段自己的背景图
  val/
    match1/                              同样的结构
      median.npz
      csv/r002_f001544_ball.csv
      frame/r002_f001544/...
```

和 TrackNetV3 的 `dataset.py` 读的格式逐项对应（它的 `preprocess.py` 不需要再跑，`video/` 目录也不需要）：

- **match 目录**：名字必须是 `match<整数>`。`dataset.py` 会对 `train/`、`val/` 下的每个名字取 `match` 后面的整数，所以这两个目录里不能有别的东西（包括 macOS 的 `.DS_Store`：用访达打开过的话先删掉）。
- **一段（rally 目录）**：名字是 `r<回合序号，三位>_f<起始帧号，六位>`，回合序号从 0 开始，对应 `match.labels.json` 里 `rallies` 的下标；起始帧号是原视频的帧号。TrackNetV3 用连续 8 帧做一个训练样本，所以一个回合里遇到没标的帧或 `skip` 的帧就从那里切成两段，不足 8 帧的段丢掉。同一个回合切出来的各段总在同一个划分里。
- **帧**：`frame/<段>/<k>.png`，`k` 从 0 开始连续编号，第 `k` 张是原视频第 `起始帧号 + k` 帧。默认缩小到 512×288（TrackNet 的输入尺寸），缩小方法和 `badminton shuttle` 推理时完全一样（Pillow 双三次）；`dataset.py` 读到的图已经是输入尺寸，不会再缩放，训练看到的像素和推理看到的一致。512×288 的 PNG 每帧约 0.2 MB（930 帧 191 MB），原分辨率的要大十倍以上。加 `--full-size` 则按原分辨率保存。
- **标签**：`csv/<段>_ball.csv`，列正好是 `Frame,Visibility,X,Y`。`Frame` 是 `k`（0 到 n−1，和图片文件名一致）；`X,Y` 是**导出图片上的像素坐标**（默认是 512×288 上的坐标，保留两位小数；`--full-size` 时是原视频像素，整数）；看不见的帧是 `0,0,0`。`dataset.py` 用图片尺寸除以 512×288 得到比例再换算，所以两种尺寸都能直接读。
- **背景图**：`median.npz`，里面一个数组 `median`，RGB、uint8、形状 (高, 宽, 3)，尺寸和导出的帧一样。`dataset.py` 优先用 `match<N>/median.npz`，没有才用 `frame/<段>/median.npz`。match 级的那张是整段视频均匀取 300 帧的中位数（就是 `badminton shuttle` 推理时用的背景图；`--full-size` 时取 100 帧）；段级的是这一段最多 100 帧的中位数。镜头中途挪过位置的视频，可以删掉 match 级的那张，让每段用自己的背景。
- **train / val 划分**：按回合分，不按帧分。每个视频里有标注的回合按顺序排好，约 20%（`--val-frac`，至少 1 个）均匀地抽出来放进 `val/`，其余放 `train/`；只有一个回合时全部放 `train/`，这时没有验证集，`train.py` 跑不了，请至少标两个回合。
- **manifest.json**：每个 match 的视频路径、原始尺寸、导出尺寸、坐标比例，以及每一段的 `id`、`split`、回合序号、原视频起止帧号（`source_start`、`source_end`）、帧数、有球帧数、各 `Status` 的帧数。需要从导出的帧找回原视频的帧时查这个文件。

`dataset.py` 第一次读数据时会在数据目录里写索引缓存（`img_config_*.npz`、`data_l8_s*_heatmap_*.npz`），里面记着每个样本的图片路径，而且是相对于当时的运行目录的。重新导出、或者换一个目录运行之前必须删掉它们，否则会读到旧的索引。`badminton annotate-export`、下面两个脚本都会自动删。

检查导出的目录能不能被 TrackNetV3 读（用它自己的 `Shuttlecock_Trajectory_Dataset` 类加载，核对样本数、张量形状、热力图的位置是否在标注点上）：

```bash
~/codebase/third_party/TrackNetV3/.venv/bin/python scripts/check_tracknet_dataset.py \
    data/tracknet_finetune --tracknet-dir ~/codebase/third_party/TrackNetV3
```

IMG_8923 预填的两个回合（930 帧，全部是没人看过的机器预标注，加了 `--include-unreviewed` 才导得出来）的检查结果：

```
train: 1 rallies, 293 frames -> 286 windows (expected 286); frames (27, 288, 512), heatmaps (8, 288, 512); 303 visible labels checked, heatmap centre within 0.93 px of the label: OK
val: 1 rallies, 637 frames -> 79 windows (expected 79); frames (27, 288, 512), heatmaps (8, 288, 512); 274 visible labels checked, heatmap centre within 0.93 px of the label: OK
dataset.py reads the export
```

## 3. 训练

TrackNetV3 的 `train.py` 不能直接拿来在这台 Mac 上微调，有四个原因：

- 它只有「从头训练」和「接着上次训练」两种方式，没有「从发布的权重开始」。接着训练时读的是 `<save_dir>/TrackNet_cur.pt`，而且学习率、批大小、保存目录都取自这个文件而不是命令行；发布的 `TrackNet_best.pt` 里还缺它要读的三项（`max_val_acc`、`scheduler`、`mask_ratio`）。
- 数据目录写死是运行目录下的 `data/`。
- 代码里到处是 `.cuda()`，Mac 上没有 CUDA。
- 它要用 TensorBoard，而 TrackNetV3 的虚拟环境里没装。

`scripts/tracknet_finetune.py` 把这四件事在外面准备好，然后原样运行 TrackNetV3 的 `train.py`（不改它的任何文件）：把发布的权重复制成 `TrackNet_cur.pt` 并补上缺的项、改好学习率；在 `<save-dir>/run/` 下建一个指向数据目录的 `data` 链接并在那里运行；把 `.cuda()` 换成移到 Apple GPU（MPS）；没装 TensorBoard 就不记曲线。

```bash
T=~/codebase/third_party/TrackNetV3

# 先空跑一遍（--epochs 0）：数据、模型、权重、优化器都加载一次，不训练，几秒钟
$T/.venv/bin/python scripts/tracknet_finetune.py --tracknet-dir $T \
    --data data/tracknet_finetune --save-dir data/tracknet_exp --epochs 0
rm data/tracknet_exp/TrackNet_cur.pt        # 空跑留下的起点，删掉重新来

# 正式训练
$T/.venv/bin/python scripts/tracknet_finetune.py --tracknet-dir $T \
    --data data/tracknet_finetune --save-dir data/tracknet_exp \
    --epochs 10 --lr 1e-4 --batch-size 8
```

- 结果在 `data/tracknet_exp/`：`TrackNet_best.pt` 是在我们自己的验证回合上准确率最高的那一轮，`TrackNet_cur.pt` 是最后一轮。中断后原样再运行一次就接着训练；要重新从发布的权重开始，先删掉 `TrackNet_cur.pt`。
- 学习率：发布的模型是用 1e-3 训练的，微调先用 1e-4。其余设置（8 帧、背景图拼在前面、mixup 0.5、Adam）沿用发布权重里记的。
- 想看训练曲线：`$T/.venv/bin/pip install tensorboard`，训练时另开终端 `tensorboard --logdir data/tracknet_exp/logs`。
- 只微调 TrackNet（找球的网络）。InpaintNet 只看坐标序列、不看画面，沿用发布的权重。

实测时间（M5，24 GB，MPS，批大小 8）：用上面 IMG_8923 的导出（训练 286 个样本、验证 79 个样本，标签是机器预标注）从发布的权重训练了 2 轮，共 6 分 26 秒，没有算子退回 CPU。

| 阶段 | 实测 |
| --- | --- |
| 训练 | 每步 1.87 秒，即每个 8 帧样本 0.23 秒（每秒 4.3 个样本） |
| 验证 | 79 个样本 14 秒 |
| 每轮固定开销 | 约 100 秒：训练和验证各等约 50 秒，是 8 个读数据的进程启动和退出的时间，和数据量无关 |
| 内存 | 批大小 8 约 12 GB，批大小 4 约 6–8 GB（速度一样） |

训练集的样本数约等于训练帧数（每段少 7 个）。按上面的数字推算：

| 标注量（其中约 80% 进训练集） | 每轮 | 10 轮 |
| --- | --- | --- |
| 1000 帧 | 约 5 分钟 | 约 50 分钟 |
| 全部 7 个回合（3831 帧） | 约 13–14 分钟 | 约 2 小时 15 分钟 |

说明：这次试跑的标签是模型自己的预测，只验证了流程和耗时，训练出的模型好不好没有意义；`train.py` 打印的验证准确率也是和机器预标注比的。有了人工标注的回合再看准确率。中断后续跑（再运行一次同样的命令）也实测过。

## 4. 转成 Core ML

`badminton shuttle` 用的是缓存在 `~/.cache/badminton/` 的 Core ML 模型，由 `badminton.shuttle.convert_models(文件夹)` 从一个「里面有 `model.py` 和 `ckpts/TrackNet_best.pt`、`ckpts/InpaintNet_best.pt`」的文件夹转换出来。给它拼一个这样的文件夹，TrackNet 用微调的，InpaintNet 用原来的：

```bash
T=~/codebase/third_party/TrackNetV3
mkdir -p data/tracknet_exp/convert/ckpts
ln -sf $T/model.py data/tracknet_exp/convert/model.py
cp data/tracknet_exp/TrackNet_best.pt data/tracknet_exp/convert/ckpts/
cp $T/ckpts/InpaintNet_best.pt data/tracknet_exp/convert/ckpts/

# 先转到一个单独的目录
python -c "
from pathlib import Path
from badminton.shuttle import convert_models
print(convert_models('data/tracknet_exp/convert', Path('data/tracknet_exp/coreml')))"
```

转换约 6 秒。然后用 `--model-dir` 让 `badminton shuttle` 用这个文件夹里的模型，缓存里发布的模型不动：

```bash
badminton shuttle data/IMG_8923_stab.mp4 --model-dir data/tracknet_exp/coreml -o outputs/IMG_8923.finetuned.shuttle.csv
```

速度和发布的模型一样（M5 上约 75 帧/秒）。确认微调的模型更好之后，想让所有命令（包括 `badminton annotate` 的预标）都默认用它，就把它换进缓存（文件名必须不变）：

```bash
C=~/.cache/badminton
mv $C/tracknetv3_tracknet_fp16_b2.mlpackage $C/tracknetv3_tracknet_fp16_b2.released.mlpackage
cp -R data/tracknet_exp/coreml/tracknetv3_tracknet_fp16_b2.mlpackage $C/
```

换回去：删掉 `$C/tracknetv3_tracknet_fp16_b2.mlpackage`，把 `.released` 那个改回原名。`badminton annotate` 缓存的预测（`<视频名>.annotate.npz`）是旧模型的，换模型后删掉它再标，预测才是新模型的。

比较微调前后：用两个模型各跑一遍 `badminton shuttle`（一次不加、一次加 `--model-dir`），和验证回合的人工标注（`match.labels.csv` 里 `human` 的帧）比对有无球是否一致、位置相差多少像素。

## 5. 发布微调的模型，默认使用

微调好、确认更好的 TrackNet 以 zip（`.mlpackage` 整个文件夹）发到 GitHub 仓库的 Release，模型文件不进 git 历史。代码里 `shuttle.TRACKNET_RELEASES` 记着每个版本的文件名、大小和 sha256，`TRACKNET_DEFAULT` 是默认用哪个。第一次用到时 `badminton shuttle` 从 `https://github.com/likefudan/badminton_analysis/releases/download/<版本>/<文件>` 下载，校验 sha256（不对就报错，什么也不留下），解压到 `~/.cache/badminton/<版本>/`，以后直接用缓存。InpaintNet 一直用发布的原版。

| 版本 | 训练数据 | 验证回合（IMG_0971 1528–1754 帧）TrackNet 原始输出 |
| --- | --- | --- |
| `released` | TrackNetV3 原版 | 漏检 31.3%，误报 11/16 帧，位置误差中位数 8.4 px，>30 px 43.4% |
| `tracknet-ft-v1` | IMG_0971（手机 60 fps 4K 双打）1247 帧人工标注，10 轮里第 8 轮 | 漏检 15.2%，误报 0/16 帧，中位数 4.1 px，>30 px 2.2% |

（4K 原图像素；只有一段录像的数字，换球馆、换机位不一定一样。）

- 换回原版：`BADMINTON_TRACKNET=released badminton shuttle …`（需要 `--tracknet-dir` 或 `TRACKNETV3_DIR` 转换一次）。
- 用别的文件夹：`--model-dir` 仍然优先。
- 处理节点的 `<视频名>.shuttle.version` 和标注工具的预测缓存都记着用的是哪个模型，换模型后重新处理会重新追踪。

发新版本：按第 4 节转出 `.mlpackage`，在 `coreml/` 里 `zip -qrX <版本>.zip tracknetv3_tracknet_fp16_b2.mlpackage`，`shasum -a 256` 算校验值，`gh release create <版本> <zip> <zip>.sha256`，再把版本、文件名、大小、sha256 加进 `TRACKNET_RELEASES`。
