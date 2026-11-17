# 击球动作指标（技术分析第 1 阶段）

`badminton technique` 在 `badminton strokes` 的结果上，按击球类型给每一拍量几样动作数字。
调研和完整计划见 [调研文档](https://claude.ai/artifact/DPaeCc5cycxLWNfgVQkzJx)。

```bash
badminton strokes  VIDEO.mp4 --court VIDEO.court.json --no-video   # 先有击球类型
badminton technique VIDEO.mp4 --court VIDEO.court.json              # 写 VIDEO.technique.json / .csv
```

## 量什么

窗口是击球前 0.8 秒到击球后 0.4 秒。长度都以躯干长（两肩中点到两髋中点）为单位，所以远近、高矮的人能比。
角度是画面里看到的角度，不是真实关节角，只适合同一机位之间比较。

| 类型 | 指标 | 意思 |
| --- | --- | --- |
| 杀球、高远/平高、吊球 | `contact_height` | 击球时手腕高出两肩多少 |
| | `elbow_deg` | 击球时肩-肘-腕夹角 |
| | `backswing_elbow` | 引拍时持拍肘最高抬到肩上多少 |
| | `chain_ms`、`chain_order_ok` | 髋、肩、肘、腕各自最快的时刻（相对击球，毫秒），是否从髋到腕依次 |
| | `wrist_speed` | 手腕最快速度（躯干长/秒） |
| | `turn` | 肩线左右宽度变化的幅度，转体的代理量 |
| | `jump` | 击球前脚踝升高多少 |
| | `follow_through` | 击球后手腕往身体另一侧扫过多远 |
| | `disguise`（只有吊球） | 吊球的引拍和自己杀球的引拍差多少，除以杀球之间本身的差别；1 左右说明看不出来 |
| 放网、勾对角、挑球、推扑、抽挡 | `contact_height`、`wrist_speed` | 同上 |
| | `stance` | 击球时两脚间距 |
| | `knee_deg` | 弯曲那条腿的膝角 |
| | `trunk_lean` | 躯干偏离竖直的角度 |

默认只分析近端球员（`--sides near`），远端人太小；默认跳过类型没把握的击球（`--include-review` 也算上）。
关节在窗口里有 30% 以上时间看不清的拍不出数字，`reason` 写原因。持拍手按每人所有击球里哪只手腕挥得最快投票决定。

## 参考分布和提示

`badminton technique-ref ref.json A.technique.json B.technique.json ...` 把多段视频（比如职业比赛）的数字合成
每种球、每个指标的 10/25/50/75/90 百分位。之后 `badminton technique VIDEO --reference ref.json` 会给每个数字标百分位，
落在 10% 以下或 90% 以上的写进 `notes`。

没有参考时只有几条草稿规则（发力顺序反了、击球点低于 1 个躯干长、几乎没转体），阈值还没用标注校准过。

`summary` 里每种球、每个人有均值和标准差；`steadiness` 是「同一个人拍与拍之间的波动 ÷ 所有人的波动」，
越小说明这个指标越能区分人、越值得信。第 1 阶段的门槛就是看这几个指标在真实录像上稳不稳。
