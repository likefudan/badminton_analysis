// 集成测试：启动本地 wrangler dev（R2 和 D1 都是本地模拟），走一遍分片上传。
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const PORT = 8799;
const BASE = `http://127.0.0.1:${PORT}/api/v1`;
const TOKEN = "test-token";
const auth = { authorization: `Bearer ${TOKEN}` };
const state = mkdtempSync(join(tmpdir(), "ba-test-"));
let server;

const call = (path, init = {}) => fetch(BASE + path, { ...init, headers: { ...auth, ...(init.headers || {}) } });

before(async () => {
  execFileSync("npx", ["wrangler", "d1", "migrations", "apply", "ba-db", "--local", "--persist-to", state], {
    stdio: "ignore", env: { ...process.env, CI: "1" },
  });
  server = spawn("npx", ["wrangler", "dev", "--local", "--port", String(PORT), "--persist-to", state,
    "--var", `ACCESS_TOKEN:${TOKEN}`, "--var", "NODE_TOKEN:node-key", "--show-interactive-dev-session=false"], { stdio: "ignore", env: { ...process.env, CI: "1" } });
  for (let i = 0; i < 120; i++) {
    try { if ((await fetch(BASE + "/me")).status === 401) return; } catch {}
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("wrangler dev 没有起来");
});

after(() => {
  server?.kill();
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
  assert.equal((await fetch(`${BASE}/videos/${v.id}/results/movement.1.png`, { headers: { cookie: "ba_session=nope" } })).status, 401);
  assert.equal((await fetch(`${BASE}/videos/${v.id}`, { method: "DELETE", headers: { cookie } })).status, 401);

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
