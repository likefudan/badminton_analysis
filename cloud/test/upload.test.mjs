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
    "--var", `ACCESS_TOKEN:${TOKEN}`, "--show-interactive-dev-session=false"], { stdio: "ignore", env: { ...process.env, CI: "1" } });
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
});

test("首页和脚本能打开", async () => {
  const page = await fetch(`http://127.0.0.1:${PORT}/`);
  assert.equal(page.status, 200);
  assert.match(await page.text(), /上传录像/);
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
