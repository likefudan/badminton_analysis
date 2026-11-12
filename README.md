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
| 2. 跑动卡片 | 每人的跑动距离、最高速度、热力图 | 完成，已用转播单打实测 |
| 3. 双打站位报告 + 一键处理 | 前后站 / 左右站比例、轮转、搭档间距、空当；拖进视频就出结果 | 代码完成，用合成双打数据测过；等真实双打录像 |
| 4. AI 赛后点评 | 文字点评和练习建议，可以和以前比较、可以追问 | 代码完成；用第一段球友录像（单打）生成了摘要和提示词，还没有用 API 实际跑过 |
| 5. 自动剪掉死球时间 | 只保留回合的精简视频和「最长回合」集锦 | 未开始 |
| 6. 每一拍的统计 | 拍数、谁打的、击球点和落点、回位时间 | 未开始 |
| 7. 击球类型 | 杀、吊、高远、挑、网前、平抽的次数，剪出「我的全部杀球」 | 未开始 |
| 8. 第二机位 | 远端半场更准，击球瞬间的 3D 动作 | 未开始 |
| 9. 接入网站 | 活动页和个人页显示跑动数据和排行榜 | 未开始 |
| 10. iPhone App | 录完在手机上直接出结果 | 未开始 |

## 录像建议

详见 [docs/recording_guide.md](docs/recording_guide.md)（一页清单，每一条都有算出来的数字；`scripts/recording_planner.py` 可以算你自己的机位），要点是：

- 手机用三脚架或灯架固定在底线后方正中，离底线 3 米、架高 3 米（至少 2 米和 2 米）往下拍，横屏。
- 用 0.5x 超广角，整片场地和它上面的空间都在画面内：近端底线在画面下沿往上一点，远端底线大约在画面正中的高度。
- 4K 60fps，关闭 HDR 视频，锁定对焦和曝光。0.5x 下 1080p 的远端球员只有约 70 像素高，姿态不可靠。
- 快门 1/500 秒（系统相机不能设，可用 Blackmagic Camera 等 App），否则球会拖成模糊的线。
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

### 自动标定（不用手点）

```bash
badminton calibrate match.mp4 --auto              # 默认用第 1 秒起的几帧；换时刻加 --time 30
badminton calibrate match.mp4 --auto --hint 900,700   # 画面里有好几片场地时，给要的那片场地内任意一点的像素坐标
badminton calibrate match.mp4 --auto --every 20   # 标定后每 20 秒再对一次线，报告机位漂移了多少像素
```

输出和手动标定一样（`match.court.json` 和预览图），后面的命令不用改。做法是纯图像处理，不用模型：

1. 取几帧做中值去掉走动的人（画面在动就只用第一帧），找出「白色、细长」的像素；
2. 在上面找直线，按消失点分成沿场地和横跨场地两组；
3. 把每组线和标准球场的线（双打场地 13.40 × 6.10 m）逐一试着对应，得到几千个候选的单应矩阵，按「投影出来的球场线有多少压在白线上」打分；
4. 最好的几个再对着白线像素精调，取最好的一个。

结果会打印置信度（投影的球场线落在白线上的比例）、对上了几条线、离白线平均几个像素。置信度低于 70% 或对上的线不足 6 条时不保存，退出码非 0，并提示改用手动标定；被拒绝的结果画在 `.rejected.png` 里供参考。

实测（和手点的标定比 30 个线交点的像素距离，`scripts/validate_autocal.py`）：

| 视频 | 自动标定 | 和手点的差 | 说明 |
| --- | --- | --- | --- |
| 转播单打（高机位，一片场地） | 置信度 98%，12/12 条线 | 平均 2.5 px，最大 5.6 px | 和手点误差（2.0 px）同一量级 |
| 手机录像（1.4 m 低机位斜拍，旁边有别的场地），第 67 秒 | 置信度 95%，10/10 条线 | 平均 7.7 px，最大 15.6 px | 没达到 5 px：见下 |

手机录像差得多，主要不是自动标定偏了：自动结果离白线平均 0.5 px，手点的那份在同一帧上离白线 1.8 px、只有 79% 压在线上；而且半秒后的第 67.6 秒两者只差 4.4 px，说明手点那一帧本身也有几像素的偏差。另外这段「防抖后」的视频在两次碰机位之间的静止段里还有 5–9 px 的缓慢漂移（`--every` 在低机位远端测出的 10–50 px 大多是对线本身的误差，不全是机位在动），所以一次标定管全程最多也就准到这个量级。

局限：

- 只认白线（或明显比地面亮的浅色线）；黄线、木地板上的多种颜色的线、反光很强的地面不行。
- 球场要大部分在画面里、近半场的线要清楚；只拍到一个角、或者旁边的场地只露出一部分时会拒绝。
- 画面里有多片完整场地时默认选线最长最清楚的那片（一般就是正对着的），不对就用 `--hint`。
- 场地是对称的，分不出哪头是哪头：按约定把离相机近的底线当近端、相机视角的左边当左。
- 不处理镜头畸变（超广角边缘的线是弯的，拟合的是折中的直线）。

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

## 第 2 步：跑动卡片

在第 1 步的结果上跑，不用再跑模型，几秒钟出结果。

```bash
# 1. 给球员起名字（每局一次）：先生成一张标了编号的图，看图再起名
badminton label match.mp4 --court match.court.json
#   生成 match.players.png；在终端里按提示输入名字，或者直接给：
badminton label match.mp4 --court match.court.json --names "1=小王,2=小李,3=老张,4=老刘"

# 2. 出跑动卡片和统计
badminton movement match.mp4 --court match.court.json
#   每人一张 match.movement.<编号>.png，外加 match.movement.json（给后面的 AI 点评用）
```

卡片上是：球员半场的热力图和跑动轨迹，跑动距离、最高速度、平均速度、冲刺次数（超过 3 米/秒、持续 0.3 秒以上）、回合内时间，以及前中后场、左右两侧的时间比例（左右按球员自己面向球网的方向）。

怎么算的：

- **跟踪编号**：每人一直在自己那半场；每一帧按「上一帧位置推算的位置」和「球衣颜色」把人对上号，双打搭档交叉换位也不会换错。编号顺序是近端在前、从左到右。
- **跳起来的时候**：人在空中时，脚踝离地，按地面换算会以为人一下子跑远了。程序比较「肩到脚踝的画面高度」和「脚下地面的比例尺」，跳起时这个比值会突然变大，这些帧不用脚的位置，改用起跳前和落地后的位置插值。再加一道限速：两帧之间算出来超过 8 米/秒的也当作异常。
- **短暂丢失**（0.8 秒以内）用插值补上，更长的不补，也不算距离。
- **回合切分**：只看人的动作，有人在快速移动（超过 1.5 米/秒）就算回合中，两人都慢走超过 2.5 秒就算死球。所有统计只算回合内时间。这个切分比较粗：发球和前一两拍动作少，回合开头常被漏掉 1–2 秒；第 5 步加入羽毛球轨迹后会更准。
- **速度**：位置先做 0.5 秒窗口的 Savitzky-Golay 平滑再求导，单帧抖动不会变成假的高速。
- 剪辑过的视频，用 `--cuts "8.4,16.6"` 给出剪辑点，跑动不跨剪辑点计算（换镜头会自动识别，同一机位的拼接识别不出来）。

转播单打实测（2025 年总决赛男单决赛，72 秒拼接片段，7 个回合，回合内 54 秒）：近端球员跑了 99 米，最高 6.6 米/秒，平均 1.9 米/秒；远端球员跑了 107 米，最高 5.3 米/秒，平均 2.0 米/秒。最长的回合（16.5 秒）两人各跑了 35 米。高水平单打的速度峰值一般在 5–7 米/秒，数字是合理的。

已知的不足：转播机位很高，跳起时脚的换算位置偏得不多，判断是否在空中的办法只能抓到一部分跳跃帧（这段视频里近端一次后撤跳杀抓到 4/6 帧）。手机放在 2.5–4 米高时，跳起时的偏差大得多，这个判断也更灵。

## 羽毛球轨迹（准备第 5、6 步）

用 [TrackNetV3](https://github.com/qaz812345/TrackNetV3)（MIT 许可证）找出每一帧里球的位置，转换成原生 Core ML 模型在 Mac 上跑。

TrackNetV3 的代码和权重不放进本项目，只在第一次转换模型时用到：

```bash
pip install -e ".[coreml]"                       # 转换模型要用 PyTorch 和 coremltools
git clone https://github.com/qaz812345/TrackNetV3.git ~/codebase/third_party/TrackNetV3
# 按它的 README 下载权重，解压到 ckpts/（TrackNet_best.pt 和 InpaintNet_best.pt）
export TRACKNETV3_DIR=~/codebase/third_party/TrackNetV3

badminton shuttle match.mp4 --overlay
#   生成 match.shuttle.csv（每帧：Frame, Visibility, X, Y）和画了轨迹的 match.shuttle.mp4
```

转换好的模型缓存在 `~/.cache/badminton/`，以后不再需要 TrackNetV3 文件夹。

处理方式和 TrackNetV3 原版的 `predict.py` 一样：8 帧加一张背景中位数图一起送进 TrackNet，重叠的窗口按 1,2,3,4,4,3,2,1 加权平均，再用 InpaintNet 补短暂的丢失。只在下面几处做了提速：

- 每帧只解码、缩小一次（原版每帧要在 8 个窗口里各缩小一次）。
- 解码和缩小放在另一个进程里，和模型推理同时进行。
- 两个窗口一起送进模型。
- 模型转换成 Core ML 半精度格式。和骨架模型不同，TrackNet 在神经网络引擎上比 GPU 快。

实测（M5，72 秒 1080p30 转播单打，2166 帧）：

| | 原版（PyTorch，Apple GPU） | `badminton shuttle` |
| --- | --- | --- |
| 总耗时 | 779 秒 | 37 秒（背景图 5 秒 + 逐帧 32 秒） |
| 逐帧速度 | 约 2.8 帧/秒 | 68 帧/秒 |
| 每帧耗时 | — | 模型 13 毫秒；解码加缩小 12 毫秒，和模型同时进行 |

结果和原版对比：每一帧「有没有球」的判断完全一致；位置 99.4% 的帧相差 5 像素以内（中位数 1 像素）；之前人工核对过的 40 帧，全部在 3 像素以内。

60 fps 的视频建议加 `--frame-step 2`：每隔一帧送一帧给模型（TrackNetV3 是用 30 fps 训练的），跳过的帧用前后插值补上。程序会按实际解码出的画面尺寸换算坐标，所以手机视频带旋转标记也没关系；输入是 HDR 视频时会提示先转 SDR。

用 `--resize area` 改用 OpenCV 缩小画面，会稍快一点，但只有 95.9% 的帧在 5 像素以内，所以默认用和原版一样的 Pillow。

这段转播视频接近 TrackNetV3 的训练数据（30 fps 转播）。自己用手机拍的视频（机位更低、60 fps）很可能要用自己的录像微调。

## 手机录像的处理（第一段球友录像的经验）

第一段球友视频是手机手持拍的：HDR、机位低（离地 1.43 米）、在球场边上斜着拍，背景里有隔壁场地、观众和一排灯，手机偶尔被碰到。针对这些情况加了下面几样：

```bash
badminton sdr IMG_8923.mov IMG_8923.mp4            # HDR 转 SDR（ffmpeg 没有 zscale 时改用 Mac 的 VideoToolbox）
badminton stabilize IMG_8923.mp4 IMG_8923_stab.mp4 --ref-time 100 --start 33 --floor-from 0.42
badminton calibrate IMG_8923_stab.mp4 --time 67    # 在防抖后的视频上标定一次
badminton shuttle IMG_8923_stab.mp4 --court IMG_8923_stab.court.json   # 用改进的选点方法
badminton movement IMG_8923_stab.mp4 --court IMG_8923_stab.court.json --data ... --start 6
badminton overlay IMG_8923_stab.skeleton.mp4 --shuttle IMG_8923_stab.shuttle.csv   # 骨架视频上再画球的轨迹
```

- **iPhone 空间音频**：新款 iPhone 的视频第一条音轨可能是空间音频（apple_apac），ffmpeg 解不了；`sdr` 会自动改用能解的那条（通常是 AAC）。
- **防抖**：先测出相邻帧之间镜头动了多少，把视频分成「镜头静止」和「被碰动」两类时间段。每段静止时间用同一个对齐方式（几帧结果的中位数），被碰动的几帧跟着镜头走，再平滑过渡到下一段。处理后相邻帧之间只差 0.2 像素，和原片一样稳。`--floor-from` 表示只用画面下部（地面）的特征点来对齐。
- **跑动更准**：每个位置带着自己的精度（4 像素误差换算到球场上），用卡尔曼平滑。机位低时，远端 1 像素对应 12–17 厘米，原来的平滑会把抖动算成跑动（修正前远端多算了将近一倍）。
- **排除场外的人**：瞬间移动不可能那么远、球衣颜色也对不上的检测，不算同一个人；在线外站着不动超过 0.5 秒的，不算球员；补空缺需要的速度超过 5 米/秒的，不补。
- **羽毛球选点**（加 `--court` 时启用）：整段视频里经常出现候选点的位置（墙上的灯、地上的球、固定假点）直接排除；再从每帧所有候选点里挑一条最连贯的轨迹，不再逐帧选「面积最大的那块」。用户视频后半段：来回折返从 156 处降到 13 处，抽查正确率从约 48% 提到约 73%。
- **三维相机**：从球场标定推出焦距和手机位置（`camera.py`），把网画回画面检查。机位高的时候还能把「本场地上方的空间」以外的候选点去掉；机位太低时这块空间覆盖整个画面，起不到作用。

## 标注羽毛球位置，准备微调 TrackNetV3

手机拍的视频上，TrackNetV3 发布的模型只有五到七成的帧是对的，要用自己的录像微调。`badminton annotate` 是为此做的标注工具：在浏览器里一帧一帧地看，模型的预测已经画在画面上，对的按空格确认，不对的点一下球的位置。三步：

```bash
# 1. 打开标注页面（http://localhost:8765，只在本机能访问）。第一次会先跑一遍 TrackNetV3 并缓存预测
badminton annotate match.mp4 --rallies match.movement.json
#   标注每一步都自动保存到 match.labels.csv（TrackNetV3 的格式：Frame, Visibility, X, Y，另加一列 Status）
#   和 match.labels.json（谁、什么时候标的、工具版本）；标完按 Ctrl+C

# 2. 导出成 TrackNetV3 训练用的目录（帧、csv、背景图，按回合分 train/val）
badminton annotate-export match.mp4 -o data/tracknet_finetune

# 3. 微调、再转成 Core ML：见 docs/tracknet_finetune.md
```

标注页面的按键：

| 操作 | 含义 |
| --- | --- |
| 点击 | 球在这里 |
| 空格 / V | 接受模型的预测 |
| N | 这一帧没有球 |
| O | 被挡住或拿不准（这一帧不用于训练） |
| 1–5 | 选画面上编了号的候选点 |
| ← → / J K | 上一帧、下一帧（加 Shift 跳 10 帧）；下方的时间条和滑块可以直接跳 |
| U | 跳到本回合下一个模型没把握、还没看过的帧 |
| [ ] | 上一个、下一个回合 |
| Shift+A | 本回合里模型有把握、还没看过的帧一次全部接受 |
| ⌫ | 删掉这一帧的标注；Z 开关放大镜 |

每标一帧自动跳到下一帧，鼠标旁边有放大镜。画面上黄圈是这一帧的预测，前后几帧的预测画得淡（蓝色是之前、红色是之后），白色虚线圈是其他候选点，粉色虚线圈是 TrackNetV3 原版逐帧选的点（和黄圈不一致时才画）。

TrackNetV3 用连续 8 帧做一个训练样本，所以要整个回合连着标，不能东一帧西一帧。工具做的是排顺序：按「模型有多没把握」给每一帧打分（原版选点和按轨迹选的点不一致、热力图分数低、同一帧有多个候选点、和前后帧相比跳得很远），回合按平均分从高到低排，最没把握的回合先标；回合内按 U 可以先看最可疑的帧。右侧显示已标帧数和估计还要多少分钟。

- `--rallies` 是 `badminton movement` 生成的 `.movement.json`（和视频同名时自动找到）；没有就按球出现的时段切。
- `--prefill N`：把最没把握的 N 个回合先填上模型的预测，状态是「机器预标注」，表示没人看过；导出时默认不包含，确认过才算。
- 已经有 `badminton shuttle` 的结果、不想重跑：`--shuttle match.shuttle.csv --raw match.shuttle_raw.csv`（这样没有候选点和热力图分数，打分只看两份结果是否一致和轨迹跳变）。

速度：确认一帧约 0.4 秒，点一帧约 2.5 秒。第一段球友视频里约 44% 的帧模型没把握，按这个比例 1000 帧大约 20–25 分钟（估计值，还没有人实际标完一遍来核对）。

导出目录的格式（目录结构、CSV 的列、帧的命名、背景图、train/val 怎么分）、训练命令、预计时间、转 Core ML 的办法都写在 [docs/tracknet_finetune.md](docs/tracknet_finetune.md)。

## 第 3 步：双打站位报告和一键处理

一条命令跑完前面所有步骤：

```bash
badminton analyze match.mp4
#   1. 球场标定：有 match.court.json 就沿用，没有就打开窗口点标记点
#   2. 骨架：结果已存在就沿用（--force 重跑）
#   3. 球员编号：生成 players.png，第一次会在终端里问名字
#   4. 跑动卡片，双打再加站位报告
#   全部输出在 match_analysis/，汇总数据在 match_analysis/match.report.json（给第 4 步的 AI 点评用）

# 看了编号图之后带上名字再跑一次（骨架不会重跑，几秒钟）
badminton analyze match.mp4 --names "1=小王,2=小李,3=老张,4=老刘"
```

单打加 `--singles`；只想先试一分钟加 `--duration 60`。站位报告也可以单独跑：`badminton positioning match.mp4 --court match.court.json`。

每对搭档一张站位卡片（`match.positioning.near.png` / `far.png`），外加 `match.positioning.json`：

- **站位形式**：两人连线和球网的夹角。大于 60° 算前后站位（进攻阵形），小于 30° 算左右站位（防守阵形），中间算过渡。给出各占回合内时间的比例，以及前后站位时各自在前面的时间。
- **轮转**：前后站位时谁在前、左右站位时谁在左。换人并且保持 0.5 秒以上算一次轮转，给出次数、每分钟次数和发生的时间点。左右换位途中短暂经过前后站位，不算前后轮转。
- **搭档间距**：平均值、中位数、10% 和 90% 分位；太近（小于 1.5 米，容易两人抢一个球）和太远（大于 4.5 米，中间露出空当）的时间比例。
- **空当**：每一帧找出本方半场里离两人都最远的点，就是最大的空当。给出它平均离最近的人多远，以及最常出现在哪个区域（前、中、后场 × 左、右，按球员面向球网的方向）；卡片上的蓝色是空当出现的位置。
- **各自覆盖**：半场里离谁更近就算谁负责，给出两人各占的比例。

现在还不知道每一拍是谁在进攻，所以「进攻时是否前后站位、防守时是否左右站位」要等第 5、6 步加入羽毛球轨迹之后才能回答。

目前只有合成的双打数据测过（测试里有一段脚本：先前后站位、中途前后换位，再左右站位、中途左右换位）。真实的双打录像到了以后，再检查阈值是否合适。

## 第 4 步：AI 赛后点评

把前面算出的统计交给通用大模型，写成球友看得懂的点评。大模型只读数字，不看视频和图片。

```bash
# 默认不联网：只写出 match.review.summary.json（摘要）和 match.review.prompt.txt（完整提示词）
badminton review match.movement.json
#   双打可以再加上站位报告：badminton review match.movement.json match.positioning.json

# 把提示词贴进任意聊天模型，回答存成文本文件，再导入：生成 match.review.md 和 match.review.html（并排嵌入跑动卡片）
badminton review match.movement.json --response 回答.md

# 或者通过 Anthropic API 直接出点评：必须加 --api，光设置了 key 不会调用
pip install -e ".[review]"
export ANTHROPIC_API_KEY=...
badminton review match.movement.json --api

# 追问（沿用同一份数据和已有的点评），回答追加到 match.review.qa.md；不加 --api 就只写 match.review.ask.prompt.txt
badminton review match.movement.json --ask "我下次最该先练哪一项？" --api

# 和上一次比较：给上一次的摘要
badminton review match.movement.json --previous last_week.review.summary.json
```

**发送了什么。** 只有加了 `--api`（并且设置了 `ANTHROPIC_API_KEY`）才会联网，发送的内容就是 `match.review.prompt.txt`（追问时是 `match.review.ask.prompt.txt`）里的文字，发送前会打印出文件名和字数，可以先打开看。里面有：

- 程序算出的数字：时长、回合起止时间、跑动距离、速度、冲刺、站位比例、误差估计、羽毛球检测比例；
- 球员的名字（`--names` / `label` 里起的名字，不想发真名就起个代号）和输入文件名（例如 `match`）；
- `--previous` 给的上一次摘要、`--ask` 的问题、追问时之前的点评；
- 再加上固定的说明和写作要求。

不发送视频、画面、图片、骨架数据、文件夹路径（附加的 JSON 里的路径只保留文件名）。跑动卡片图片只嵌进本地的 HTML 报告。

导入的回答（`--response` 或 API 返回的文字）在 HTML 里全部按纯文本转义，只认标题、列表、引用、粗体和代码这几种 Markdown 格式，回答里的 HTML 或脚本不会被执行。

几个常用参数：

- `--court match.court.json`：算出相机高度和每人平均站位处的位置误差，写进「数据质量」。
- `--shuttle match.shuttle.csv --video match.mp4`：每个回合里羽毛球被检测到的比例（视频只用来读帧率，也可以直接给 `--fps`）。
- `--shuttle-accuracy 73`：人工核对过的羽毛球位置正确率，程序算不出来，只能手动给。
- `--model`：和 `--api` 一起用，默认 `claude-sonnet-5-5`。

摘要里有什么（都是程序算的，结果固定，同样的输入每次一样）：

- **比赛**：视频时长、回合内时间和比例、回合数、回合平均和中位长度、最长回合。
- **球员**：跑动距离、每分钟跑动、平均和最高速度、冲刺次数和每分钟冲刺、前中后场和左右比例、腾空帧数、回合内被跟踪到的时间比例、位置误差。热力图的原始格子不放进去。
- **回合**：每个回合一行，编号、起止时间、长度、之前停了多久、每人的距离和速度、球被检测到的比例。
- **对比**：每项指标谁高、高多少。
- **数据质量**：按数字自动生成，例如某人被跟踪到的时间不到 90%、远端前后误差多少米、相机低于 2.5 米、回合不足 10 个、羽毛球轨迹分不出回合和死球。

提示词要求模型按固定的五节写：比赛概况、球员跑动画像、三条观察（带数字，引用回合编号和时间）、三个练习建议（带理由）、数据不确定的地方；并且不许写数据里没有的击球、比分和胜负。

用第一段球友录像试过（单打，159 秒，7 个回合）。程序算出：相机高度约 1.43 米；近端位置误差前后 0.09 米，远端 0.57 米；远端球员回合内只有 85% 的时间被跟踪到。这台机器没有 API key，所以只生成了摘要和提示词，另外按提示词手写了点评再用 `--response` 导入（第二段录像 jay_cdl 也是这样做的），还没有用 API 实际跑过。

已知的不足：摘要目前只有跑动和站位，没有每一拍的数据，所以点评说不了技术动作和战术细节；第 6、7 步完成后摘要会加上拍数和击球类型。

## 代码结构

| 文件 | 内容 |
| --- | --- |
| `src/badminton/court.py` | 球场尺寸、标记点、画面与球场坐标的单应变换、检测区域 |
| `src/badminton/calibrate.py` | 点标记点的窗口（带放大镜） |
| `src/badminton/autocal.py` | 自动标定：找白线、和标准球场对应、精调、机位漂移 |
| `src/badminton/pose.py` | 人体检测 + 姿态（rtmlib 的 ONNX 模型），推理后端选择和计时 |
| `src/badminton/detector.py` | 把检测模型在 NMS 之前切开，去重改用 numpy |
| `src/badminton/coreml_native.py` | 可选的原生 Core ML 半精度后端（模型转换、缓存、成批推理） |
| `src/badminton/profiling.py` | `badminton profile` 的分段计时和单模型测速 |
| `scripts/quantize_coreml.py` | 8 位量化实验（不在主流程里） |
| `src/badminton/players.py` | 从检测到的人里挑出本场球员（按半场、滤掉场外的人） |
| `src/badminton/render.py` | 画球场线和骨架 |
| `src/badminton/video.py` | 读写视频、HDR 转 SDR |
| `src/badminton/pipeline.py` | 第 1 步的完整流程 |
| `src/badminton/tracking.py` | 第 2 步：球员跟踪编号、跳起检测、补短暂丢失 |
| `src/badminton/movement.py` | 第 2 步：回合切分、跑动距离、速度、冲刺、站位、热力图 |
| `src/badminton/card.py` | 第 2 步：跑动卡片 PNG |
| `src/badminton/shuttle.py` | 羽毛球轨迹：TrackNetV3 转 Core ML、加权平均、补丢失 |
| `src/badminton/shuttle_path.py` | 羽毛球选点：静止干扰点、场地上方空间、按轨迹连续性挑点 |
| `src/badminton/stabilize.py` | 防抖：镜头静止时固定对齐，被碰动时跟着镜头走 |
| `src/badminton/camera.py` | 从球场标定推出焦距和相机位置 |
| `src/badminton/positioning.py` | 第 3 步：双打站位、轮转、搭档间距、空当、覆盖和站位卡片 |
| `src/badminton/annotate.py`、`annotate.html` | 羽毛球标注：标注文件、模型没把握程度、本机网页（只用标准库） |
| `src/badminton/annotate_export.py` | 把标注导出成 TrackNetV3 的训练目录 |
| `scripts/check_tracknet_dataset.py` | 用 TrackNetV3 自己的 Dataset 类检查导出的目录（在它的虚拟环境里运行） |
| `scripts/tracknet_finetune.py` | 从发布的权重开始微调：准备好之后原样运行 TrackNetV3 的 `train.py` |
| `src/badminton/review.py` | 第 4 步：统计摘要、数据质量说明、提示词、Anthropic API 调用、Markdown / HTML 报告 |
| `src/badminton/cli.py` | 命令行 `badminton` |

运行测试：`pytest`。测试用合成的摄像机和假的姿态模型，不需要下载模型。

## 许可证说明

羽毛球轨迹用的是 [TrackNetV3](https://github.com/qaz812345/TrackNetV3)（Chen & Wang，代码和权重都是 MIT 许可证）。它的代码和权重不在本仓库里；`shuttle.py` 按它的 `predict.py` 重写了推理流程，转换出来的 Core ML 模型只缓存在本机。

检测和姿态模型来自 [rtmlib](https://github.com/Tau-J/rtmlib)（Apache-2.0）：人体检测用的是它自带的 YOLOX（Megvii，Apache-2.0，在 COCO + HumanArt 上训练），姿态用 RTMPose。方案里写的是 RTMDet，两者都是 Apache-2.0；rtmlib 的现成模型配的是 YOLOX，先用它，远端漏检多再换。
