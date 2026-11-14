// 集成测试：启动本地 wrangler dev（R2 和 D1 都是本地模拟），走一遍分片上传。
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "node:http";

const PORT = 8799;
const BASE = `http://127.0.0.1:${PORT}/api/v1`;
const TOKEN = "test-token";
const auth = { authorization: `Bearer ${TOKEN}` };
const state = mkdtempSync(join(tmpdir(), "ba-test-"));
let server;

// 假的模型服务：两种接口都认，记下收到的请求，按流式格式回一段文字和用量
const AI_PORT = 8798;
const aiCalls = [];
const ai = createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    const openai = req.url === "/openai/chat/completions";
    if ((openai ? req.headers["api-key"] : req.headers["x-api-key"]) !== "test-ai-key" || (!openai && req.url !== "/anthropic/v1/messages")) {
      res.writeHead(401, { "content-type": "application/json" });
      return res.end(JSON.stringify({ type: "error", error: { type: "authentication_error", message: "invalid key or path " + req.url } }));
    }
    const data = JSON.parse(body);
    aiCalls.push({ ...data, path: req.url });
    const last = data.messages.at(-1).content;
    const answer = last.startsWith("追问") ? ["先练", "步法。"] : ["## 比赛概况\n\n打了 ", "**25 秒**。<script>x</script>"];
    res.writeHead(200, { "content-type": "text/event-stream" });
    const send = (ev) => res.write(`${ev.type ? `event: ${ev.type}\n` : ""}data: ${JSON.stringify(ev)}\n\n`);
    if (openai) {
      for (const text of answer) send({ choices: [{ delta: { content: text } }] });
      send({ choices: [{ delta: {}, finish_reason: "stop" }] });
      send({ choices: [], usage: { prompt_tokens: 900, completion_tokens: 40, total_tokens: 1500 } }); // 600 个思考 token 只在 total 里
      res.write("data: [DONE]\n\n");
    } else {
      send({ type: "message_start", message: { content: [], usage: { input_tokens: 1000, output_tokens: 1 } } });
      for (const text of answer) send({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text } });
      send({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 500 } });
      send({ type: "message_stop" });
    }
    res.end();
  });
});

const call = (path, init = {}) => fetch(BASE + path, { ...init, headers: { ...auth, ...(init.headers || {}) } });

before(async () => {
  await new Promise((r) => ai.listen(AI_PORT, "127.0.0.1", r));
  execFileSync("npx", ["wrangler", "d1", "migrations", "apply", "ba-db", "--local", "--persist-to", state], {
    stdio: "ignore", env: { ...process.env, CI: "1" },
  });
  server = spawn("npx", ["wrangler", "dev", "--local", "--port", String(PORT), "--persist-to", state,
    "--var", `ACCESS_TOKEN:${TOKEN}`, "--var", "NODE_TOKEN:node-key", "--var", "LABEL_SYNC_DELAY_S:0",
    "--var", "AI_API_KEY:test-ai-key", "--var", `AI_ENDPOINT_ANTHROPIC:http://127.0.0.1:${AI_PORT}/anthropic`,
    "--var", `AI_ENDPOINT_OPENAI:http://127.0.0.1:${AI_PORT}/openai`, "--show-interactive-dev-session=false"], { stdio: "ignore", env: { ...process.env, CI: "1" } });
  for (let i = 0; i < 120; i++) {
    try { if ((await fetch(BASE + "/me")).status === 401) return; } catch {}
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("wrangler dev 没有起来");
});

after(() => {
  server?.kill();
  ai.close();
  rmSync(state, { recursive: true, force: true });
});

test("没有口令或口令不对都是 401", async () => {
  assert.equal((await fetch(BASE + "/videos")).status, 401);
  assert.equal((await fetch(BASE + "/videos", { headers: { authorization: "Bearer nope" } })).status, 401);
  // 处理节点有自己的钥匙
  assert.equal((await fetch(BASE + "/me", { headers: { authorization: "Bearer node-key" } })).status, 200);
});

test("首页和脚本能打开", async () => {
  const page = await fetch(`http://127.0.0.1:${PORT}/`);
  assert.equal(page.status, 200);
  assert.match(await page.text(), /选择录像/);
  assert.equal((await fetch(`http://127.0.0.1:${PORT}/app.js`)).status, 200);
});

test("分片上传、断点续传、合并、下载、删除", async () => {
  const data = randomBytes(20 * 1024 * 1024 + 12345); // 3 片：8 + 8 + 4 MiB 多一点
  const res = await call("/videos", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ filename: "game1.MOV", size: data.length, contentType: "video/quicktime", meta: { width: 3840, height: 2160, fps: 60 } }),
  });
  assert.equal(res.status, 201);
  const video = await res.json();
  assert.equal(video.partCount, 3);
  assert.equal(video.status, "uploading");
  const part = (n) => data.subarray((n - 1) * video.partSize, Math.min(n * video.partSize, data.length));
  const put = (n, body = part(n)) => call(`/videos/${video.id}/parts/${n}`, { method: "PUT", body });

  // 大小不对的片被拒绝
  assert.equal((await put(1, part(1).subarray(0, 1000))).status, 400);
  assert.equal((await put(4)).status, 400);

  // 先传第 1、3 片，模拟中断
  assert.equal((await put(1)).status, 200);
  assert.equal((await put(3)).status, 200);
  let status = await (await call(`/videos/${video.id}`)).json();
  assert.deepEqual(status.uploadedParts, [1, 3]);

  // 没传完不能合并，也不能下载
  const early = await call(`/videos/${video.id}/complete`, { method: "POST" });
  assert.equal(early.status, 409);
  assert.deepEqual((await early.json()).missing, [2]);
  assert.equal((await call(`/videos/${video.id}/file`)).status, 409);

  // 续传：重传第 1 片（幂等）再补第 2 片
  assert.equal((await put(1)).status, 200);
  assert.equal((await put(2)).status, 200);
  const done = await call(`/videos/${video.id}/complete`, { method: "POST" });
  assert.equal(done.status, 200);
  status = await done.json();
  assert.equal(status.status, "uploaded");
  assert.equal(status.uploadedBytes, data.length);
  assert.equal(status.meta.fps, 60);

  // 传完后不能再传片；重复 complete 没有副作用
  assert.equal((await put(2)).status, 409);
  assert.equal((await call(`/videos/${video.id}/complete`, { method: "POST" })).status, 200);

  // 下载回来内容一致
  const file = await call(`/videos/${video.id}/file`);
  assert.equal(file.status, 200);
  assert.equal(file.headers.get("content-type"), "video/quicktime");
  const back = Buffer.from(await file.arrayBuffer());
  assert.equal(createHash("sha256").update(back).digest("hex"), createHash("sha256").update(data).digest("hex"));

  // Range 请求
  const ranged = await call(`/videos/${video.id}/file`, { headers: { range: "bytes=100-199" } });
  assert.equal(ranged.status, 206);
  assert.equal(ranged.headers.get("content-range"), `bytes 100-199/${data.length}`);
  assert.deepEqual(Buffer.from(await ranged.arrayBuffer()), data.subarray(100, 200));

  // 列表和删除
  const list = await (await call("/videos")).json();
  assert.equal(list.videos.length, 1);
  assert.equal((await call(`/videos/${video.id}`, { method: "DELETE" })).status, 200);
  assert.equal((await call(`/videos/${video.id}`)).status, 404);
});

test("删除没传完的录像会放弃分片上传", async () => {
  const res = await call("/videos", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ filename: "x.mp4", size: 1000, contentType: "video/mp4" }),
  });
  const video = await res.json();
  assert.equal(video.partCount, 1);
  assert.equal((await call(`/videos/${video.id}`, { method: "DELETE" })).status, 200);
  assert.equal((await (await call("/videos")).json()).videos.length, 0);
});

test("参数不对的新建请求被拒绝", async () => {
  for (const body of [{}, { filename: "a.mov" }, { filename: "a.mov", size: -1 }, { filename: "a.mov", size: 1.5 }]) {
    const res = await call("/videos", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    assert.equal(res.status, 400);
  }
});

test("处理任务：排队、领取、汇报进度、传结果、读结果、删云端原片、重新处理", async () => {
  const json = { "content-type": "application/json" };
  const post = (path, body) => call(path, { method: "POST", headers: json, body: JSON.stringify(body ?? {}) });
  const upload = async (name, mode) => {
    const data = randomBytes(1000);
    const v = await (await post("/videos", { filename: name, size: data.length, contentType: "video/mp4", options: { mode, junk: 1 } })).json();
    assert.equal((await call(`/videos/${v.id}/parts/1`, { method: "PUT", body: data })).status, 200);
    return { v: await (await post(`/videos/${v.id}/complete`)).json(), data };
  };

  // 没有可处理的录像时领不到
  assert.equal((await (await post("/node/claim", { node: "mac" })).json()).video, null);

  const { v, data } = await upload("a.mp4", "doubles");
  assert.equal(v.job.status, "queued");
  assert.deepEqual(v.options, { mode: "doubles" });

  // 领取一次，第二次领不到
  const claimed = (await (await post("/node/claim", { node: "mac" })).json()).video;
  assert.equal(claimed.id, v.id);
  assert.equal(claimed.job.status, "processing");
  assert.equal((await (await post("/node/claim", { node: "mac2" })).json()).video, null);

  // 取原片后让云端删掉
  assert.deepEqual(Buffer.from(await (await call(`/videos/${v.id}/file`)).arrayBuffer()), data);
  assert.equal((await post(`/videos/${v.id}/original/release`)).status, 200);
  assert.equal((await call(`/videos/${v.id}/file`)).status, 410);

  // 进度
  const progress = { steps: [{ key: "skeleton", label: "骨架识别", status: "running", percent: 40 }], etaSeconds: 90 };
  let st = await (await post(`/videos/${v.id}/job`, { status: "processing", stage: "骨架 40%", progress })).json();
  assert.equal(st.job.stage, "骨架 40%");
  assert.deepEqual(st.job.progress, progress);
  assert.ok(st.job.startedAt);
  // 只续租不带进度时，进度保持不变；网页能看到 Mac 最近在线
  st = await (await post(`/videos/${v.id}/job`, { status: "processing" })).json();
  assert.deepEqual(st.job.progress, progress);
  const listing = await (await call("/videos")).json();
  assert.equal(listing.node.name, "mac");
  assert.ok(Math.abs(new Date(listing.now) - new Date(listing.node.seenAt)) < 60000);
  assert.equal(st.originalDeleted, true);
  assert.equal((await post(`/videos/${v.id}/job`, { status: "weird" })).status, 400);

  // 小结果直接传，大结果分片传
  const card = randomBytes(5000);
  assert.equal((await call(`/videos/${v.id}/results/movement.1.png`, { method: "PUT", body: card })).status, 200);
  assert.equal((await call(`/videos/${v.id}/results/..%2Fevil`, { method: "PUT", body: card })).status, 404);
  const clip = randomBytes(6 * 1024 * 1024 + 77);
  const { uploadId } = await (await post(`/videos/${v.id}/results/rallies.mp4/multipart`)).json();
  const q = `uploadId=${encodeURIComponent(uploadId)}`;
  const parts = [];
  for (const [i, body] of [clip.subarray(0, 5 * 1024 * 1024), clip.subarray(5 * 1024 * 1024)].entries()) {
    const r = await call(`/videos/${v.id}/results/rallies.mp4/multipart?${q}&part=${i + 1}`, { method: "PUT", body });
    assert.equal(r.status, 200);
    parts.push(await r.json());
  }
  assert.equal((await post(`/videos/${v.id}/results/rallies.mp4/multipart?${q}`, { parts })).status, 200);

  st = await (await post(`/videos/${v.id}/job`, { status: "done", stage: "完成" })).json();
  assert.equal(st.job.status, "done");
  assert.deepEqual(st.results.map((r) => [r.name, r.size, r.contentType]), [
    ["movement.1.png", 5000, "image/png"],
    ["rallies.mp4", clip.length, "video/mp4"],
  ]);
  // 任务结束后节点不能再改状态
  assert.equal((await post(`/videos/${v.id}/job`, { status: "processing" })).status, 409);

  // 网页用 cookie 读结果（<img>、<video> 带不了 Authorization），支持 Range；cookie 不能用来改数据
  const session = await post("/session");
  const cookie = session.headers.get("set-cookie").split(";")[0];
  assert.match(session.headers.get("set-cookie"), /HttpOnly/);
  const img = await fetch(`${BASE}/videos/${v.id}/results/movement.1.png`, { headers: { cookie } });
  assert.equal(img.status, 200);
  assert.equal(img.headers.get("content-type"), "image/png");
  assert.deepEqual(Buffer.from(await img.arrayBuffer()), card);
  const ranged = await fetch(`${BASE}/videos/${v.id}/results/rallies.mp4`, { headers: { cookie, range: "bytes=10-19" } });
  assert.equal(ranged.status, 206);
  assert.deepEqual(Buffer.from(await ranged.arrayBuffer()), clip.subarray(10, 20));
  // ?download=1 saves it as a file named after the recording
  const dl = await fetch(`${BASE}/videos/${v.id}/results/rallies.mp4?download=1`, { headers: { cookie } });
  assert.equal(dl.status, 200);
  assert.match(dl.headers.get("content-disposition"), /^attachment; filename=".*_rallies\.mp4"; filename\*=UTF-8''.*_rallies\.mp4$/);
  await dl.arrayBuffer();
  assert.equal((await fetch(`${BASE}/videos/${v.id}/results/rallies.mp4`, { headers: { cookie } })).headers.get("content-disposition"), null);
  assert.equal((await fetch(`${BASE}/videos/${v.id}/results/movement.1.png`, { headers: { cookie: "ba_session=nope" } })).status, 401);
  assert.equal((await fetch(`${BASE}/videos/${v.id}`, { method: "DELETE", headers: { cookie } })).status, 401);

  // 起标题
  st = await (await call(`/videos/${v.id}`, { method: "PATCH", headers: json, body: JSON.stringify({ title: "  周六双打第 2 局 " }) })).json();
  assert.equal(st.title, "周六双打第 2 局");
  assert.equal((await call(`/videos/${v.id}`, { method: "PATCH", headers: json, body: "{}" })).status, 400);

  // 重新处理：回到排队，可以改单打/双打
  st = await (await post(`/videos/${v.id}/job/retry`, { options: { mode: "singles" } })).json();
  assert.equal(st.job.status, "queued");
  assert.deepEqual(st.options, { mode: "singles" });
  assert.deepEqual(st.results, []); // 旧结果清掉
  assert.equal(st.job.progress, null);
  assert.equal((await (await post("/node/claim", { node: "mac" })).json()).video.id, v.id);
  st = await (await post(`/videos/${v.id}/job`, { status: "failed", stage: "自动标定球场", error: "找不到球场线" })).json();
  assert.equal(st.job.error, "找不到球场线");

  // 删除录像时结果一起删
  assert.equal((await call(`/videos/${v.id}`, { method: "DELETE" })).status, 200);
  assert.equal((await fetch(`${BASE}/videos/${v.id}/results/movement.1.png`, { headers: { cookie } })).status, 404);
});

test("AI 报告：网站拿节点传上来的统计摘要去问模型，存成结果；可以追问、选模型，记用量和价格", async () => {
  const json = { "content-type": "application/json" };
  const post = (path, body) => call(path, { method: "POST", headers: json, body: JSON.stringify(body ?? {}) });
  const data = randomBytes(1000);
  let v = await (await post("/videos", { filename: "r.mp4", size: data.length, contentType: "video/mp4" })).json();
  await call(`/videos/${v.id}/parts/1`, { method: "PUT", body: data });
  await post(`/videos/${v.id}/complete`);
  assert.equal((await (await call("/me")).json()).review, true);

  // 能选的模型、默认模型；单价有内置的公开价，查不到的是空的
  let info = await (await call("/ai")).json();
  assert.equal(info.available, true);
  assert.equal(info.defaultModel, "claude-sonnet-5-5");
  assert.ok(info.models.some((m) => m.id === "gpt-6.1-sol" && m.priceIn === 2 && m.priceOut === 10));
  assert.ok(info.models.some((m) => m.id === "gpt-6.1-terra" && m.priceIn === null));
  assert.deepEqual(info.usage, []);

  // 还没有统计摘要
  assert.equal((await post(`/videos/${v.id}/review`)).status, 409);
  const request = { schema: 1, model: "whatever", max_tokens: 16000, system: "你是教练", user: "摘要……请写点评", ask: "追问：{question}" };
  await call(`/videos/${v.id}/results/review.request.json`, { method: "PUT", body: JSON.stringify(request) });

  // 默认模型是 Claude：走 Anthropic 接口
  let res = await post(`/videos/${v.id}/review`);
  assert.equal(res.status, 200);
  v = await res.json();
  assert.ok(v.results.some((r) => r.name === "review.md" && r.contentType.startsWith("text/markdown")));
  assert.deepEqual([aiCalls[0].path, aiCalls[0].model, aiCalls[0].system, aiCalls[0].stream], ["/anthropic/v1/messages", "claude-sonnet-5-5", "你是教练", true]);
  assert.deepEqual(aiCalls[0].messages, [{ role: "user", content: "摘要……请写点评" }]);
  const report = await (await call(`/videos/${v.id}/results/review.md`)).text();
  assert.match(report, /^## 比赛概况\n\n打了 \*\*25 秒\*\*。<script>x<\/script>\n\n> 由 Claude Sonnet 5.5 /);
  // 用量记下来了，按内置单价估算：(1000×2 + 500×10) / 100 万
  assert.deepEqual(v.ai.map((u) => [u.kind, u.model, u.inputTokens, u.outputTokens, u.costUsd]), [["report", "claude-sonnet-5-5", 1000, 500, 0.007]]);

  // 填单价、改默认模型
  assert.equal((await call("/ai/settings", { method: "PUT", headers: json, body: JSON.stringify({ defaultModel: "nope" }) })).status, 400);
  info = await (await call("/ai/settings", {
    method: "PUT", headers: json,
    body: JSON.stringify({ defaultModel: "gpt-6.1-sol", prices: { "gpt-6.1-sol": { in: 3, out: 12 }, "claude-sonnet-5-5": { in: 3, out: 15 }, junk: { in: 1, out: 1 } } }),
  })).json();
  assert.equal(info.defaultModel, "gpt-6.1-sol");
  assert.deepEqual(info.models.find((m) => m.id === "gpt-6.1-sol"), { id: "gpt-6.1-sol", label: "GPT-6.1 Sol", priceIn: 3, priceOut: 12 });

  // 追问：带上原来的摘要、报告和之前的问答；现在默认走 OpenAI 接口
  assert.equal((await post(`/videos/${v.id}/review/ask`, { question: " " })).status, 400);
  assert.equal((await post(`/videos/${v.id}/review/ask`, { question: "x", model: "nope" })).status, 400);
  res = await post(`/videos/${v.id}/review/ask`, { question: "先练什么？" });
  const asked = await res.json();
  assert.equal(asked.answer, "先练步法。");
  assert.deepEqual([aiCalls.at(-1).path, aiCalls.at(-1).model, aiCalls.at(-1).messages[0]], ["/openai/chat/completions", "gpt-6.1-sol", { role: "system", content: "你是教练" }]);
  assert.deepEqual(aiCalls.at(-1).stream_options, { include_usage: true });
  // 思考 token 算在输出里：1500 - 900 = 600；价格按刚填的单价 (900×3 + 600×12) / 100 万
  const u = asked.video.ai.at(-1);
  assert.deepEqual([u.kind, u.model, u.inputTokens, u.outputTokens], ["ask", "gpt-6.1-sol", 900, 600]);
  assert.ok(Math.abs(u.costUsd - 0.0099) < 1e-9);
  await post(`/videos/${v.id}/review/ask`, { question: "为什么？", model: "claude-sonnet-5-5" });
  assert.deepEqual(aiCalls.at(-1).messages.map((m) => m.role), ["user", "assistant", "user", "assistant", "user"]);
  assert.deepEqual(aiCalls.at(-1).messages.slice(2).map((m) => m.content), ["追问：先练什么？", "先练步法。", "追问：为什么？"]);
  const qa = await (await call(`/videos/${v.id}/results/review.qa.md`)).text();
  assert.equal(qa, "## 问：先练什么？\n\n先练步法。\n\n## 问：为什么？\n\n先练步法。\n\n");

  // 重新生成报告后，旧的追问清掉；累计用量按模型分开，录像删了也还在
  v = await (await post(`/videos/${v.id}/review`, { model: "claude-sonnet-5-5" })).json();
  assert.ok(!v.results.some((r) => r.name === "review.qa.md"));
  await call(`/videos/${v.id}`, { method: "DELETE" });
  info = await (await call("/ai")).json();
  const claude = info.usage.find((x) => x.model === "claude-sonnet-5-5");
  assert.deepEqual([claude.calls, claude.inputTokens, claude.outputTokens, claude.unpricedCalls], [3, 3000, 1500, 0]);
  assert.ok(Math.abs(claude.costUsd - (0.007 + 2 * (1000 * 3 + 500 * 15) / 1e6)) < 1e-9);
});

test("标注：网页请求、节点领取切图任务、传图、网页按 Range 读帧、存标注、节点同步", async () => {
  const json = { "content-type": "application/json" };
  const post = (path, body) => call(path, { method: "POST", headers: json, body: JSON.stringify(body ?? {}) });
  const tasks = ["annotate", "labels"]; // 不领分析任务，别的测试留下的排队录像不影响这里
  const data = randomBytes(1000);
  const v = await (await post("/videos", { filename: "label.mp4", size: data.length, contentType: "video/mp4" })).json();
  await call(`/videos/${v.id}/parts/1`, { method: "PUT", body: data });
  await post(`/videos/${v.id}/complete`);
  let st = await (await call(`/videos/${v.id}`)).json();
  assert.equal(st.annot.status, "none");
  assert.deepEqual(st.annot.labels, {});

  // 旧节点（不带 tasks）领不到标注任务
  assert.equal((await post(`/videos/${v.id}/annot`, { fresh: true })).status, 200);
  const old = await (await post("/node/claim", { node: "old-mac" })).json();
  assert.ok(!old.video || old.task === "process");
  if (old.video) await post(`/videos/${old.video.id}/job`, { status: "done" });

  const got = await (await post("/node/claim", { node: "mac", tasks })).json();
  assert.equal(got.task, "annotate");
  assert.equal(got.video.id, v.id);
  assert.equal(got.video.annot.status, "preparing");
  assert.deepEqual(got.video.annot.options, { fresh: true });
  assert.equal((await (await post("/node/claim", { node: "mac", tasks })).json()).video, null);
  assert.equal((await post(`/videos/${v.id}/annot`)).status, 409); // 准备中不能再请求

  // 传图：小文件直接传，大文件分片
  const pack = randomBytes(5 * 1024 * 1024 + 300);
  const base = `/videos/${v.id}/annot/files/rally-0.bin`;
  const { uploadId } = await (await post(base + "/multipart")).json();
  const q = `uploadId=${encodeURIComponent(uploadId)}`;
  const parts = [];
  for (const [i, body] of [pack.subarray(0, 5 * 1024 * 1024), pack.subarray(5 * 1024 * 1024)].entries()) {
    parts.push(await (await call(`${base}/multipart?${q}&part=${i + 1}`, { method: "PUT", body })).json());
  }
  assert.equal((await post(`${base}/multipart?${q}`, { parts })).status, 200);
  const state = { rallies: [{ start: 10, end: 12, pack: "rally-0.bin", offsets: [[0, 100], [100, 100], [200, 100]] }] };
  assert.equal((await call(`/videos/${v.id}/annot/files/state.json`, { method: "PUT", body: JSON.stringify(state) })).status, 200);
  st = await (await post(`/videos/${v.id}/annot/job`, { status: "preparing", stage: "上传图片" })).json();
  assert.equal(st.annot.stage, "上传图片");
  st = await (await post(`/videos/${v.id}/annot/job`, { status: "ready", stage: "1 个回合，3 帧" })).json();
  assert.equal(st.annot.status, "ready");
  assert.equal((await post(`/videos/${v.id}/annot/job`, { status: "preparing" })).status, 409);

  // 网页读 state.json 和一段帧
  assert.deepEqual(await (await call(`/videos/${v.id}/annot/files/state.json`)).json(), state);
  const part = await call(base, { headers: { range: "bytes=100-299" } });
  assert.equal(part.status, 206);
  assert.deepEqual(Buffer.from(await part.arrayBuffer()), pack.subarray(100, 300));
  assert.equal((await call(`/videos/${v.id}/annot/files/missing.bin`)).status, 404);

  // 存标注：一条不对就整批不写
  const put = (items) => call(`/videos/${v.id}/labels`, { method: "PUT", headers: json, body: JSON.stringify({ items }) });
  assert.equal((await put([{ frame: 10, visibility: 1, x: 5, y: 6, status: "machine" }])).status, 400);
  assert.equal((await put([{ frame: 10, visibility: 1, x: 5, y: 6 }, { frame: -1, visibility: 0 }])).status, 400);
  assert.equal((await put([{ frame: 10, visibility: 1, x: -5, y: 6 }])).status, 400);
  assert.equal((await put([])).status, 400);
  let res = await put([
    { frame: 10, visibility: 1, x: 100.25, y: 50.5, status: "human" },
    { frame: 11, visibility: 1, x: 7, y: 8, status: "skip" },   // 看不清：不管给了什么都是没球、0,0
    { frame: 12, visibility: 0, x: 9, y: 9, status: "bulk" },
  ]);
  assert.equal(res.status, 200);
  res = await put([{ frame: 12, delete: true }, { frame: 10, visibility: 1, x: 101, y: 50, status: "human" }]);
  const { changedAt } = await res.json();
  const labels = await (await call(`/videos/${v.id}/labels`)).json();
  assert.deepEqual(labels.labels, { 10: [1, 101, 50, "human"], 11: [0, 0, 0, "skip"] });
  assert.equal(labels.changedAt, changedAt);
  st = await (await call(`/videos/${v.id}`)).json();
  assert.deepEqual(st.annot.labels, { human: 1, skip: 1 });

  // 节点同步标注，同步完就不再领到
  await new Promise((r) => setTimeout(r, 20));
  const sync = await (await post("/node/claim", { node: "mac", tasks })).json();
  assert.equal(sync.task, "labels");
  assert.equal(sync.video.id, v.id);
  assert.equal((await post(`/videos/${v.id}/labels/synced`, { changedAt })).status, 200);
  assert.equal((await (await post("/node/claim", { node: "mac", tasks })).json()).video, null);
  st = await (await call(`/videos/${v.id}`)).json();
  assert.equal(st.annot.labelsSyncedAt, changedAt);

  // 删图片，标注留着；删录像，全部清掉
  assert.equal((await call(`/videos/${v.id}/annot`, { method: "DELETE" })).status, 200);
  assert.equal((await call(`/videos/${v.id}/annot/files/state.json`)).status, 404);
  assert.deepEqual((await (await call(`/videos/${v.id}/labels`)).json()).labels, labels.labels);
  assert.equal((await call(`/videos/${v.id}`, { method: "DELETE" })).status, 200);
  assert.equal((await call(`/videos/${v.id}/labels`)).status, 404);
});

test("击球类型：请求、节点准备、读小视频、存标注、下载 CSV、同步", async () => {
  const json = { "content-type": "application/json" };
  const post = (path, body) => call(path, { method: "POST", headers: json, body: JSON.stringify(body ?? {}) });
  const tasks = ["strokes", "stroke-labels"];
  const data = randomBytes(100);
  const v = await (await post("/videos", { filename: "game 1.mp4", size: data.length, contentType: "video/mp4" })).json();
  await call(`/videos/${v.id}/parts/1`, { method: "PUT", body: data });
  await post(`/videos/${v.id}/complete`);
  assert.equal((await post(`/videos/${v.id}/strokes`)).status, 200);
  const got = await (await post("/node/claim", { node: "mac", tasks })).json();
  assert.equal(got.task, "strokes");
  assert.equal(got.video.strokes.status, "preparing");
  const clip = randomBytes(3000);
  assert.equal((await call(`/videos/${v.id}/strokes/files/strokes.mp4`, { method: "PUT", body: clip })).status, 200);
  assert.equal((await call(`/videos/${v.id}/strokes/files/strokes.state.json`, { method: "PUT", body: "{\"strokes\":[]}" })).status, 200);
  let st = await (await post(`/videos/${v.id}/strokes/job`, { status: "ready", stage: "3 拍" })).json();
  assert.equal(st.strokes.status, "ready");
  const part = await call(`/videos/${v.id}/strokes/files/strokes.mp4`, { headers: { range: "bytes=10-19" } });
  assert.equal(part.status, 206);
  assert.equal(part.headers.get("content-type"), "video/mp4");
  assert.deepEqual(Buffer.from(await part.arrayBuffer()), clip.subarray(10, 20));

  const put = (items) => call(`/videos/${v.id}/stroke-labels`, { method: "PUT", headers: json, body: JSON.stringify({ items }) });
  assert.equal((await put([{ frame: 10, type: "大力扣杀" }])).status, 400);
  assert.equal((await put([{ frame: 10, type: "杀球", tags: ["乱写"] }])).status, 400);
  assert.equal((await put([
    { frame: 60, player_id: 3, type: "吊球", tags: ["对角", "劈吊"], note: "很快, \"假动作\"" },
    { frame: 10, player_id: null, type: "发短球", tags: [] },
    { frame: 99, type: "未知" },
  ])).status, 200);
  const { changedAt } = await (await put([{ frame: 99, delete: true }])).json();
  const labels = await (await call(`/videos/${v.id}/stroke-labels`)).json();
  assert.deepEqual(labels.labels, {
    10: { player_id: null, type: "发短球", tags: [], note: "" },
    60: { player_id: 3, type: "吊球", tags: ["劈吊", "对角"], note: "很快, \"假动作\"" },
  });
  const csv = await call(`/videos/${v.id}/stroke-labels?format=csv`);
  assert.match(csv.headers.get("content-disposition"), /game_1\.strokes\.labels\.csv/);
  assert.equal((await csv.text()).replace(/^﻿/, ""), 'frame,player_id,type,tags,note\r\n10,,发短球,,\r\n60,3,吊球,劈吊;对角,"很快, ""假动作"""\r\n');
  st = await (await call(`/videos/${v.id}`)).json();
  assert.equal(st.strokes.labels, 2);

  await new Promise((r) => setTimeout(r, 20));
  const sync = await (await post("/node/claim", { node: "mac", tasks })).json();
  assert.equal(sync.task, "stroke-labels");
  await post(`/videos/${v.id}/stroke-labels/synced`, { changedAt });
  assert.equal((await (await post("/node/claim", { node: "mac", tasks })).json()).video, null);

  assert.equal((await call(`/videos/${v.id}/strokes`, { method: "DELETE" })).status, 200);
  assert.equal((await call(`/videos/${v.id}/strokes/files/strokes.mp4`)).status, 404);
  assert.equal((await call(`/videos/${v.id}`, { method: "DELETE" })).status, 200);
});
