# 羽毛球视频分析

用开源模型在本地分析业余羽毛球（以双打为主）比赛视频：从 iPhone 录像里得到跑动、站位、回合、击球等数据，用来复盘比赛、改进技术，再让通用大模型把数字写成球友看得懂的点评。

完整的技术方案见 [docs/design.md](docs/design.md)（模型选型、功能、硬件、数据、机位、分步计划和风险）。

## 设计原则

- **全部本地、全部开源。** 视频不离开自己的电脑；优先选 Apache-2.0 / MIT 许可证的组件，不用 AGPL 的 YOLO 和 BoxMOT，以后放进网站或 App 没有授权问题。
- **一套模型，多个平台。** 模型统一导出成 ONNX，用 ONNX Runtime 推理：Mac 上走 Core ML（GPU 和神经网络引擎），NVIDIA 上走 CUDA / TensorRT，以后可以直接转成 iPhone 用的 Core ML 模型。
- **大模型只读数字。** 所有统计由程序算好，存成 JSON；通用大模型只负责写报告和回答问题，不看视频。
- **每一步都独立可用。** 每个处理阶段的结果都存成文件，后面的步骤可以单独重跑；任何一步停下来，前面的功能照常能用。

## 处理流程

1. 在画面上手动点一次球场标记点（四角加发球线交点，6 个以上），得到画面到球场俯视图的换算关系。
2. 逐帧检测场上的人，并估计每人 17 个关节点（RTMPose）；球场范围以外的人直接滤掉。
3. 跨帧跟踪，给每人固定编号；每一场开头标一次「谁是谁」。
4. 把脚的位置换算成球场上的真实坐标（米），得到跑动轨迹；按运动状态粗分「回合中」和「死球时间」。
5. 追踪羽毛球（TrackNetV3），找出击球时刻，精确切分回合。
6. 用骨架序列判断击球类型。
7. 汇总成统计 JSON，交给大模型写点评。

## 路线图

| 步骤 | 球友拿到什么 | 状态 |
| --- | --- | --- |
| 1. 骨架视频 | 叠加了骨架和球场线的比赛视频，可以慢放看动作 | 完成，已在 Mac（M5）上用转播视频实测 |
| 2. 跑动卡片 | 每人的跑动距离、最高速度、热力图 | 未开始 |
| 3. 双打站位报告 + 一键处理 | 前后站 / 左右站比例、轮转、搭档间距、空当；拖进视频就出结果 | 未开始 |
| 4. AI 赛后点评 | 文字点评和练习建议，可以和以前比较、可以追问 | 未开始 |
| 5. 自动剪掉死球时间 | 只保留回合的精简视频和「最长回合」集锦 | 未开始 |
| 6. 每一拍的统计 | 拍数、谁打的、击球点和落点、回位时间 | 未开始 |
| 7. 击球类型 | 杀、吊、高远、挑、网前、平抽的次数，剪出「我的全部杀球」 | 未开始 |
| 8. 第二机位 | 远端半场更准，击球瞬间的 3D 动作 | 未开始 |
| 9. 接入网站 | 活动页和个人页显示跑动数据和排行榜 | 未开始 |
| 10. iPhone App | 录完在手机上直接出结果 | 未开始 |

## 录像建议

详见 [docs/design.md](docs/design.md) 第 7 节，要点是：

- 手机放在底线后方正中，架高 2.5–4 米往下拍，横屏，整片场地都在画面内。
- 1080p 60fps，关闭 HDR 视频，锁定对焦和曝光。
- 快门尽量 1/500 秒或更快（系统相机不能设，可用 Blackmagic Camera 等 App），否则球会拖成模糊的线。
- 录到别人之前先征得本人同意，原始视频留在本地。

## 安装

首先支持 MacBook（Apple silicon），之后支持 NVIDIA 显卡的 Linux / Windows 机器。需要 Python 3.10 以上和 ffmpeg。

Mac：

```bash
brew install ffmpeg python@3.12
git clone https://github.com/likefudan/badminton_analysis.git
cd badminton_analysis
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Mac 上想再快一倍，加装可选的原生 Core ML 后端（约 700 MB，主要是只在转换模型时用到的 PyTorch）：

```bash
pip install -e ".[dev,coreml]"
```

装了它，`--device auto` 会自动选 `coremlnative`。第一次运行时把模型转换成 Core ML 半精度格式（每个模式约 15 秒），存在 `~/.cache/badminton`，以后直接用。

NVIDIA：同上，再把 CPU 版 ONNX Runtime 换成 GPU 版：`pip uninstall -y onnxruntime && pip install onnxruntime-gpu`。

第一次运行会从 OpenMMLab 下载模型（约 100 MB，存在 `~/.cache/rtmlib`）。

## 第 1 步：骨架视频

```bash
# 1. 看视频参数，以及这台机器能用哪些推理后端
badminton info match.mov

# iPhone 录的 HDR 视频先转成 SDR，否则颜色发灰
badminton sdr match.mov match.mp4

# 2. 点球场标记点（开一个窗口，按提示依次点；看不到的点按 n 跳过）
badminton calibrate match.mp4
#   生成 match.court.json 和预览图 match.court.png，请看一眼球场线是否贴合

# 3. 出骨架视频（双打）；单打加 --singles；先试一分钟可以加 --duration 60
badminton skeleton match.mp4 --court match.court.json
#   生成 match.skeleton.mp4（带原声）和 match.skeleton.npz（每帧的关节点和球场坐标）

# 测速：比较 Core ML 和纯 CPU
badminton bench match.mp4 --court match.court.json --devices coreml,cpu
```

几个常用参数：

- `--mode performance`：用更大的模型，远端球员小、骨架不准时试试；`lightweight` 更快。
- `--stride 1`：每帧都跑模型（默认每 2 帧跑一次，60fps 视频即每秒 30 次）。
- `--device cpu|coreml|coremlnative|cuda|tensorrt`：手动指定推理后端，默认自动选（Mac 上装了 `[coreml]` 选 `coremlnative`，否则 `coreml`）。
- `--show-roi`：画出检测区域。程序默认只在球场附近检测，这样远端的人在模型输入里更大，场外的人也少。

### 关于推理后端和速度

ONNX Runtime 不是只能用 CPU。它把模型交给「执行后端」：Mac 上是 Core ML，NVIDIA 上是 CUDA 或 TensorRT。后端不支持的算子会退回 CPU，所以程序会打印实际用到的后端和每帧耗时。`badminton bench` 对比几个后端；`badminton profile` 把时间拆成解码、检测、姿态、画图、编码，再单独测每个模型在各种后端设置下的速度。

Mac 上有两个后端：

- `coreml`：ONNX Runtime 的 Core ML 后端。它交给 Core ML 的是 32 位模型，只能在 GPU 上跑，用不上神经网络引擎。检测模型末尾的去重（NMS）已经从模型里切出来改用 numpy 做：那部分每帧输出长度不同，Core ML 每次都要重新准备（每帧多花约 10 ms），画面里没人时还会出错。
- `coremlnative`：转换成原生 Core ML 半精度模型，在 GPU 上跑；同一帧的所有球员一次送进姿态模型。

实测速度（MacBook Apple M5，1080p 转播单打，`badminton profile`，每帧都跑模型，单位：帧/秒）：

| 模式 | cpu | coreml（改进前） | coreml | coremlnative |
| --- | --- | --- | --- | --- |
| lightweight | 30.9 | 33.3 | 38.5 | 69.8 |
| balanced（默认） | 8.9 | 22.1 | 29.1 | 42.7 |
| performance | 2.2 | — | 7.2 | 20.6 |

「改进前」是 ONNX Runtime 跑完整检测模型（带 NMS）。默认每 2 帧跑一次模型，所以 balanced + coremlnative 处理 72 秒的 30fps 视频（含读写视频）只要 27 秒，`coreml` 是 40 秒。

`coremlnative` 和 CPU 结果的差别（每种模式 20 帧真实画面）：检测到的人数完全相同，球员框差 2.5 像素以内（1080p 画面，相当于模型输入的不到 1 像素），可信关节点在 192×256 的人物裁剪图里差 1–2 像素，换算成球场位置中位数差 4 毫米。

几个试过但没采用的方向：

- 神经网络引擎：这两个模型在 M5 上用神经网络引擎反而比 GPU 慢（yolox-m 半精度 23 ms 对 GPU 10 ms）。
- 8 位量化（权重和激活都是 int8，`scripts/quantize_coreml.py`）：GPU 上检测只快 12%（8.5 对 9.6 ms），姿态不变；神经网络引擎编译不了量化后的姿态模型，检测在神经网络引擎上 55 ms。精度不能用：检测框最多偏 36 像素，姿态关节点几乎全错（激活量化对 RTMPose 的 SimCC 输出太粗）。
- fp8：coremltools 9.0 的模型格式里有 fp8 类型编号，但转换器和运行时都不支持 fp8 张量，只有 PyTorch 端的模拟量化能用。

### 没有自己的录像时

可以先用网上的单打比赛视频试（转播机位也行），只在本地测试用，不要再分发。例如用 [yt-dlp](https://github.com/yt-dlp/yt-dlp)：

```bash
brew install yt-dlp
yt-dlp -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" -o singles.mp4 "<视频链接>"
badminton calibrate singles.mp4 --time 30
badminton skeleton singles.mp4 --court singles.court.json --singles --start 30 --duration 60
```

## 代码结构

| 文件 | 内容 |
| --- | --- |
| `src/badminton/court.py` | 球场尺寸、标记点、画面与球场坐标的单应变换、检测区域 |
| `src/badminton/calibrate.py` | 点标记点的窗口（带放大镜） |
| `src/badminton/pose.py` | 人体检测 + 姿态（rtmlib 的 ONNX 模型），推理后端选择和计时 |
| `src/badminton/detector.py` | 把检测模型在 NMS 之前切开，去重改用 numpy |
| `src/badminton/coreml_native.py` | 可选的原生 Core ML 半精度后端（模型转换、缓存、成批推理） |
| `src/badminton/profiling.py` | `badminton profile` 的分段计时和单模型测速 |
| `scripts/quantize_coreml.py` | 8 位量化实验（不在主流程里） |
| `src/badminton/players.py` | 从检测到的人里挑出本场球员（按半场、滤掉场外的人） |
| `src/badminton/render.py` | 画球场线和骨架 |
| `src/badminton/video.py` | 读写视频、HDR 转 SDR |
| `src/badminton/pipeline.py` | 第 1 步的完整流程 |
| `src/badminton/cli.py` | 命令行 `badminton` |

运行测试：`pytest`。测试用合成的摄像机和假的姿态模型，不需要下载模型。

## 许可证说明

检测和姿态模型来自 [rtmlib](https://github.com/Tau-J/rtmlib)（Apache-2.0）：人体检测用的是它自带的 YOLOX（Megvii，Apache-2.0，在 COCO + HumanArt 上训练），姿态用 RTMPose。方案里写的是 RTMDet，两者都是 Apache-2.0；rtmlib 的现成模型配的是 YOLOX，先用它，远端漏检多再换。
