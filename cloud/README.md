# cloud/：ba.llmat.dev 的后端和手机网页

手机上传录像、Mac 处理、手机看报告这条线的云端部分，跑在 Cloudflare 上（Worker + D1 + R2）。
设计见方案文档《手机上传录像、手机看报告：可复用的后端方案》。

流程：手机网页上传录像（分片、可断点续传）→ 家里的 Mac 上常驻的处理节点（`badminton node`）取走原片，云端立刻删掉原片 → Mac 跑分析，把跑动卡片、统计、精简视频传回云端 → 手机网页上点开这一局看结果。云端只留最近约 10 GB 的结果，更早的只在 Mac 上。

## 目录

- `src/index.ts`：Worker，接口都在 `/api/v1` 下，用 `Authorization: Bearer <口令>` 登录。
- `public/`：手机网页（纯 HTML + JS，不需要构建）。
- `migrations/`：D1 表结构。
- `test/`：集成测试，启动本地 `wrangler dev` 走一遍上传流程。

## 接口（阶段 0）

| 接口 | 作用 |
| --- | --- |
| `GET /me` | 检查口令 |
| `POST /videos` | 新建一个上传，返回编号、分片大小（8 MiB，超大文件会按倍数放大）和分片数 |
| `PUT /videos/{id}/parts/{n}` | 传第 n 片，可重复传 |
| `GET /videos/{id}` | 查状态和已传的片（续传用） |
| `POST /videos/{id}/complete` | 合并，核对总大小 |
| `GET /videos/{id}/file` | 下载原片，支持 Range（以后 Mac 处理节点用） |
| `GET /videos`、`DELETE /videos/{id}` | 列表、删除（结果一起删） |
| `PATCH /videos/{id}` | 改标题，请求体 `{"title": "…"}`（新建时也可以带 `title`） |
| `POST /session` | 发一个只读 cookie，网页里的图片和视频靠它读取 |
| `POST /node/claim` | 处理节点领取一个排队中的录像（节点失联 10 分钟后可被重新领取） |
| `POST /videos/{id}/job` | 节点汇报进度（`stage` 文字加 `progress` 步骤列表：每步状态、用时、百分比、预计剩余），或以 `done` / `failed` 结束 |
| `POST /videos/{id}/job/retry` | 重新处理，可以改单打/双打 |
| `POST /videos/{id}/original/release` | 节点下载完原片后调用，云端删掉原片 |
| `PUT /videos/{id}/results/{name}` | 传一个结果文件；大文件用 `.../multipart` 分片传 |
| `GET /videos/{id}/results/{name}` | 读结果，支持 Range |
| `POST /videos/{id}/annot`、`DELETE /videos/{id}/annot` | 请 Mac 准备手机标注用的图片（`{"fresh": true}` 用新模型重新预测）；删掉图片（标注保留） |
| `POST /videos/{id}/annot/job` | 节点汇报准备进度，或以 `ready` / `failed` 结束 |
| `PUT/GET /videos/{id}/annot/files/{name}` | 标注用的图片包和 `state.json`（大文件用 `.../multipart`），读的时候支持 Range |
| `GET/PUT /videos/{id}/labels`、`POST /videos/{id}/labels/synced` | 读、存标注（`{items: [{frame, visibility, x, y, status}]}`，status 是 human / bulk / skip）；节点同步完后记一笔 |

## Mac 上的处理节点

节点用自己的钥匙连网站，和登录口令分开：`~/.config/badminton/node.json` 里的 `token` 要和仓库 secret `BA_NODE_TOKEN` 一致（部署时写成 Worker 的 `NODE_TOKEN`）。也可以直接填登录口令。

```bash
umask 077; mkdir -p ~/.config/badminton
python3 -c 'import json,secrets; print(json.dumps({"token": secrets.token_hex(32)}))' > ~/.config/badminton/node.json
python3 -c 'import json,sys,pathlib; sys.stdout.write(json.loads((pathlib.Path.home()/".config/badminton/node.json").read_text())["token"])' | gh secret set BA_NODE_TOKEN
# 然后重新运行一次部署流程
badminton node            # 常驻：每 15 秒查一次有没有新录像
badminton node --once     # 处理完排队中的就退出
```

原片和结果放在 `~/badminton_uploads/<录像编号>/`。每一局依次做：下载原片（断了接着下）→ 通知云端删原片 → HDR 转 SDR → 自动标定球场（`calibrate --auto`，在录像的几个时刻各试一次，取线条重合度最高的）→ `badminton analyze` → 羽毛球追踪（`badminton shuttle --court` 加 `overlay`，画在骨架视频上；失败不影响其它结果）→ 写出 AI 报告要用的统计摘要和提示词（`badminton review`，不联网）→ 结果视频转成手机能放的 1080p H.264 → 上传（另外截一帧 `cover.jpg` 作为列表里的封面）→ 让网站出 AI 报告。
网页上能看到每一步的状态、用时和预计剩余时间。预计时间来自这台 Mac 之前处理的记录（`~/badminton_uploads/timings.json`，按录像时长折算），第一次处理时只有下载和骨架识别这两步有预计。网页还显示 Mac 是否在线（节点最后一次联系的时间）和排队位置。

自动标定失败时这一局标为失败，网页上能看到原因；在 Mac 上手动标定（把 `court.json` 放在原片旁边）之后，在网页上点“重新处理”即可。

## 手机上标注球的位置

录像页的「标注」里点「准备标注」：节点领到 `annotate` 任务（只有没有录像要分析时才领），运行 `badminton annotate-pack`，挑模型最没把握的 3 个回合（优先 20 秒以内的，最多 3600 帧），每个回合的帧切成 1280 宽的 JPEG 拼成一个文件，`state.json` 记每帧在文件里的位置和模型的预测，传到 R2 的 `annot/` 下（不算在 10 GB 结果里）。网页（`public/label.js`，`#/label/<编号>`）一次用 Range 取 8 帧。
每标一帧存进 D1 的 `labels` 表；停手一分钟后节点领到 `labels` 任务，写回录像旁边的 `<视频名>.labels.csv`（网页回合里的帧以网站为准，其它帧保留 Mac 上原来的标注），之后 `badminton annotate-export` 照常用。
再次准备时沿用同样的回合，所以微调后点「用新模型重新预测」，标注页上的准确率就是同一批帧上微调前后的对比。

## 本地运行

```bash
cd cloud
npm install
echo 'ACCESS_TOKEN=随便一个口令' > .dev.vars
npm run migrate:local
npm run dev        # http://localhost:8787
npm test           # 集成测试
npm run typecheck
```

## 自动部署（GitHub Actions）

`.github/workflows/deploy-cloud.yml` 在 `cloud/` 的改动合并到 main 时自动发布，也可以在 Actions 页面手动运行。
它会按需创建 R2 桶和 D1 数据库、设置 7 天清理未完成分片上传的规则、跑迁移、发布、设置登录口令，最后检查线上首页和 `/api/v1/me`。

需要在仓库 Settings → Secrets and variables → Actions 里加三个 secret：

| secret | 内容 |
| --- | --- |
| `CLOUDFLARE_API_TOKEN` | Cloudflare 令牌，权限：Workers Scripts 编辑、Workers R2 Storage 编辑、D1 编辑、llmat.dev 的 Workers Routes 编辑 |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare 账号 ID |
| `BA_ACCESS_TOKEN` | 网站登录口令，自己定 |

## 手动部署

需要 `llmat.dev` 在同一个 Cloudflare 账号下。

```bash
npx wrangler login
npx wrangler r2 bucket create ba-files
npx wrangler d1 create ba-db          # 把输出的 database_id 填进 wrangler.toml
npm run migrate:remote
npx wrangler secret put ACCESS_TOKEN  # 设置登录口令
npm run deploy                        # 发布到 https://ba.llmat.dev
```

没传完就放弃的分片会占 R2 空间，建议给桶加一条生命周期规则，自动清理 7 天前未完成的分片上传：

```bash
npx wrangler r2 bucket lifecycle add ba-files abort-stale-uploads --abort-multipart-days 7
```

## AI 报告

Mac 上不放 API key。处理节点只上传 `review.request.json`（`badminton review` 写的统计摘要和提示词，只有数字和文字，没有视频和图片）。网站（Worker）用自己的 key 调模型：

- `POST /api/v1/videos/{id}/review`，请求体 `{"model": "…"}`（可以不填）：生成报告，存成结果 `review.md`。节点上传完会自动调一次（用默认模型），结果页可以换模型重新生成。
- `POST /api/v1/videos/{id}/review/ask`，请求体 `{"question": "…", "model": "…"}`：追问，带上摘要、报告和之前的问答，回答追加到 `review.qa.md`。
- `GET /api/v1/ai`：可选的模型和单价、默认模型、按模型汇总的累计用量。
- `PUT /api/v1/ai/settings`，请求体 `{"defaultModel": "…", "prices": {"<模型>": {"in": 2, "out": 10}}}`：改默认模型和单价（美元 / 100 万 token）。

配置：

| 名字 | 放在哪 | 含义 |
|---|---|---|
| `AI_API_KEY` | 仓库的 Actions secret（`AI_API_KEY` 或 `AZURE_API_KEY`），部署时写进 Worker | 模型的 API key。没有就不出报告（处理步骤里「AI 报告」显示跳过），网站其它功能照常 |
| `AI_ENDPOINT_ANTHROPIC` | `wrangler.toml` 的 `[vars]` | Claude 模型的地址（Messages 接口，头 `x-api-key`），会补上 `/v1/messages`。现在是 Azure AI Foundry 资源的 `…services.ai.azure.com/anthropic` |
| `AI_ENDPOINT_OPENAI` | `[vars]` | GPT、Grok 等模型的地址（Chat Completions 接口，头 `api-key`），会补上 `/chat/completions`。现在是 `…openai.azure.com/openai/v1` |
| `AI_MODELS` | `[vars]`，可选 | JSON 数组 `[{"id": "部署名", "label": "显示名", "format": "anthropic" 或 "openai", "in": 单价, "out": 单价}]`，替换 `src/index.ts` 里内置的模型列表（Azure 上部署的 8 个） |

**用量和价格。** 每次调用模型，接口返回的输入、输出 token 数记在 D1 的 `ai_usage` 表里，同时按当时的单价算出估算价格。结果页显示每份报告、每次追问的用量，设置页显示累计。单价默认是公开的标价，可以在设置页改成账单上的数字（存在 `settings` 表）；没有单价的模型只记 token 数。价格只是估算，以 Azure 账单为准。

报告在网页上只按标题、列表、引用、粗体、行内代码渲染，里面的 HTML 不会被执行。
