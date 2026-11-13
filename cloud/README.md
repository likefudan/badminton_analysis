# cloud/：ba.llmat.dev 的后端和手机网页

手机上传录像、Mac 处理、手机看报告这条线的云端部分，跑在 Cloudflare 上（Worker + D1 + R2）。
设计见方案文档《手机上传录像、手机看报告：可复用的后端方案》。

现在是**阶段 0**：只有分片上传和一个上传测试页，用来在 iPhone 上实测三件事：

1. 从相册或“文件”里选 4K60 录像时，网页拿到的是不是原片（页面会显示分辨率、估计帧率、每分钟大小）。
2. 实际上传速度。
3. 切到后台、锁屏、断网之后能不能接着传（页面会记录进入后台的次数和时长）。

## 目录

- `src/index.ts`：Worker，接口都在 `/api/v1` 下，用 `Authorization: Bearer <口令>` 登录。
- `public/`：手机网页（纯 HTML + JS，不需要构建）。
- `migrations/`：D1 表结构。
- `test/`：集成测试，启动本地 `wrangler dev` 走一遍上传流程。

## 接口（阶段 0）

| 接口 | 作用 |
| --- | --- |
| `GET /me` | 检查口令 |
| `POST /videos` | 新建一个上传，返回编号、分片大小（32 MiB）和分片数 |
| `PUT /videos/{id}/parts/{n}` | 传第 n 片，可重复传 |
| `GET /videos/{id}` | 查状态和已传的片（续传用） |
| `POST /videos/{id}/complete` | 合并，核对总大小 |
| `GET /videos/{id}/file` | 下载原片，支持 Range（以后 Mac 处理节点用） |
| `GET /videos`、`DELETE /videos/{id}` | 列表、删除 |

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

## 第一次部署

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
