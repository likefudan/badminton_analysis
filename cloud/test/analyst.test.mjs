// 集成测试：「能用 Mac 分析」的成员：上传自己的录像、排在超级管理员后面、限额、别人看不到。
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "node:http";

const PORT = 8795;
const ORIGIN = `http://127.0.0.1:${PORT}`;
const BASE = `${ORIGIN}/api/v1`;
// wrangler dev 把请求的域名换成 wrangler.toml 里的网站域名，所以回调地址是它
const CALLBACK = "http://ba.llmat.dev/api/v1/auth/google/callback";
const TOKEN = "test-token";
const state = mkdtempSync(join(tmpdir(), "ba-analyst-"));
let server;

// 假的 Google：授权码就是邮箱，换回来的 id_token 里写着这个邮箱
const G_PORT = 8794;
const b64 = (o) => Buffer.from(JSON.stringify(o)).toString("base64url");
const google = createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    const form = new URLSearchParams(body);
    if (form.get("client_secret") !== "test-secret" || form.get("grant_type") !== "authorization_code" || form.get("redirect_uri") !== CALLBACK) {
      res.writeHead(400, { "content-type": "application/json" });
      return res.end(JSON.stringify({ error: "invalid_grant" }));
    }
    const [email, flag] = form.get("code").split("|");
    const claims = {
      iss: "https://accounts.google.com", aud: flag === "aud" ? "someone-else" : "test-client", email, email_verified: flag !== "unverified",
      name: email.split("@")[0], exp: Math.floor(Date.now() / 1000) + 600,
    };
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ id_token: `${b64({ alg: "none" })}.${b64(claims)}.sig` }));
  });
});

const json = { "content-type": "application/json" };
const bearer = { authorization: `Bearer ${TOKEN}` };
// 口令（节点、老的登录方式）
const call = (path, init = {}) => fetch(BASE + path, { ...init, headers: { ...bearer, ...(init.headers || {}) } });
const post = (path, body) => call(path, { method: "POST", headers: json, body: JSON.stringify(body ?? {}) });
// 用 Google 登录的浏览器：只带 cookie 和 x-ba
const as = (cookie) => ({
  get: (path) => fetch(BASE + path, { headers: { cookie } }),
  post: (path, body) => fetch(BASE + path, { method: "POST", headers: { cookie, "x-ba": "1", ...json }, body: JSON.stringify(body ?? {}) }),
  put: (path, body) => fetch(BASE + path, { method: "PUT", headers: { cookie, "x-ba": "1", ...json }, body: JSON.stringify(body ?? {}) }),
  send: (method, path, body) => fetch(BASE + path, { method, headers: { cookie, "x-ba": "1", ...json }, body: JSON.stringify(body ?? {}) }),
});
const cookieFrom = (res, name) => {
  const c = res.headers.getSetCookie().find((x) => x.startsWith(name + "="));
  return c ? c.split(";")[0] : "";
};

async function login(code) {
  const start = await fetch(BASE + "/auth/google/start", { redirect: "manual" });
  assert.equal(start.status, 302);
  const to = new URL(start.headers.get("location"));
  assert.equal(to.origin + to.pathname, `http://127.0.0.1:${G_PORT}/auth`);
  assert.equal(to.searchParams.get("client_id"), "test-client");
  assert.equal(to.searchParams.get("redirect_uri"), CALLBACK);
  assert.equal(to.searchParams.get("scope"), "openid email profile");
  const back = await fetch(`${BASE}/auth/google/callback?code=${encodeURIComponent(code)}&state=${to.searchParams.get("state")}`, {
    redirect: "manual", headers: { cookie: cookieFrom(start, "ba_oauth") },
  });
  assert.equal(back.status, 302);
  return { location: back.headers.get("location"), cookie: cookieFrom(back, "ba_sid") };
}

const sql = (command) =>
  execFileSync("npx", ["wrangler", "d1", "execute", "ba-db", "--local", "--persist-to", state, "--command", command], { stdio: "ignore", env: { ...process.env, CI: "1" } });

before(async () => {
  await new Promise((r) => google.listen(G_PORT, "127.0.0.1", r));
  execFileSync("npx", ["wrangler", "d1", "migrations", "apply", "ba-db", "--local", "--persist-to", state], { stdio: "ignore", env: { ...process.env, CI: "1" } });
  server = spawn("npx", ["wrangler", "dev", "--local", "--port", String(PORT), "--inspector-port", "9335", "--persist-to", state,
    "--var", `ACCESS_TOKEN:${TOKEN}`, "--var", "ADMIN_GOOGLE_EMAIL:boss@example.com", "--var", "AI_API_KEY:fake", "--var", "GOOGLE_CLIENT_ID:test-client", "--var", "GOOGLE_CLIENT_SECRET:test-secret",
    "--var", `GOOGLE_AUTH_URL:http://127.0.0.1:${G_PORT}/auth`, "--var", `GOOGLE_TOKEN_URL:http://127.0.0.1:${G_PORT}/token`,
    "--show-interactive-dev-session=false"], { stdio: "ignore", env: { ...process.env, CI: "1" } });
  for (let i = 0; i < 120; i++) {
    try { if ((await fetch(BASE + "/me")).status === 401) return; } catch {}
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("wrangler dev 没有起来");
});

after(() => {
  server?.kill();
  google.close();
  rmSync(state, { recursive: true, force: true });
});


// 传一段录像（用 who 的身份；不给就用口令 = 超级管理员）。done = false 时只建不传
async function upload(who, name, done = true) {
  const data = randomBytes(300);
  const make = (b) => (who ? who.post("/videos", b) : post("/videos", b));
  let res;
  for (let i = 0; i < 10; i++) {
    if (i) await new Promise((r) => setTimeout(r, 500)); // 本地模拟的 R2 刚启动时偶尔还没就绪
    res = await make({ filename: name, size: data.length, contentType: "video/mp4" });
    if (res.status !== 500) break;
  }
  if (res.status !== 201) return res;
  const v = await res.json();
  if (!done) return v;
  const put = who ? fetch(`${BASE}/videos/${v.id}/parts/1`, { method: "PUT", headers: { cookie: who.cookie, "x-ba": "1" }, body: data })
    : call(`/videos/${v.id}/parts/1`, { method: "PUT", body: data });
  assert.equal((await put).status, 200);
  const fin = who ? await who.post(`/videos/${v.id}/complete`) : await post(`/videos/${v.id}/complete`);
  assert.equal(fin.status, 200, await fin.text());
  return v;
}

test("能用 Mac 分析：上传、排队、限额、谁看得到", async () => {
  const user = async (email) => {
    const c = (await login(email)).cookie;
    return { ...as(c), cookie: c };
  };
  const boss = await user("boss@example.com");
  assert.equal((await (await boss.get("/me")).json()).owner, true);
  const ann = await user("ann@example.com");
  const vic = await user("vic@example.com");
  const carl = await user("carl@example.com");
  for (const [email, role] of [["ann@example.com", "viewer"], ["vic@example.com", "viewer"], ["carl@example.com", "admin"]])
    assert.equal((await boss.post("/admin/users", { email, status: "active", role })).status, 200);

  // 没开之前：普通用户不能上传
  assert.equal((await (await ann.get("/me")).json()).canAnalyze, false);
  assert.equal((await ann.post("/videos", { filename: "a.mp4", size: 10 })).status, 403);
  // 只有超级管理员能开（别的管理员不行）
  assert.equal((await carl.post("/admin/users", { email: "ann@example.com", canAnalyze: true })).status, 403);
  const users = (await (await boss.post("/admin/users", { email: "ann@example.com", canAnalyze: true })).json()).users;
  const a = users.find((u) => u.email === "ann@example.com");
  assert.deepEqual([a.role, a.status, a.canAnalyze, a.videos], ["viewer", "active", true, 0]);
  assert.equal((await (await ann.get("/me")).json()).canAnalyze, true);
  // 改别的（角色）不会把它关掉
  await boss.post("/admin/users", { email: "ann@example.com", role: "annotator" });
  assert.equal((await (await ann.get("/me")).json()).canAnalyze, true);

  // 超级管理员先传 O1，ann 传 A1，超级管理员再传 O2：Mac 先做 O1、O2，再做 A1
  const o1 = await upload(null, "o1.mp4");
  const a1 = await upload(ann, "a1.mp4");
  const o2 = await upload(null, "o2.mp4");
  const a2 = await upload(ann, "a2.mp4", false); // 还在上传：也占一个名额
  assert.equal(a2.uploader, "ann@example.com");
  const full = await upload(ann, "a3.mp4", false);
  assert.equal(full.status, 429);
  const mine = await (await ann.get(`/videos/${a1.id}`)).json();
  assert.deepEqual([mine.uploader, mine.job.status, mine.job.ahead], ["ann@example.com", "queued", 2]);
  assert.equal((await (await boss.get(`/videos/${o2.id}`)).json()).job.ahead, 1);
  const claim = async () => (await (await post("/node/claim", { node: "mac", tasks: ["process"] })).json()).video?.id;
  assert.equal(await claim(), o1.id);
  assert.equal((await (await ann.get(`/videos/${a1.id}`)).json()).job.ahead, 2); // O1 处理中 + O2 排在前面
  assert.equal(await claim(), o2.id);
  assert.equal(await claim(), a1.id);

  // 谁看得到：ann 看得到自己的两段，看不到超级管理员还没结果的；vic 看不到 ann 的
  const list = await (await ann.get("/videos")).json();
  assert.deepEqual(list.videos.map((v) => v.id).sort(), [a1.id, a2.id].sort());
  assert.equal(list.node.name, ""); // 知道 Mac 在不在线，不给名字
  assert.ok(list.node.seenAt);
  assert.deepEqual((await (await vic.get("/videos")).json()).videos, []);
  assert.equal((await vic.get(`/videos/${a1.id}`)).status, 404);
  assert.equal((await call(`/videos/${a1.id}/results/points.json`, { method: "PUT", headers: json, body: JSON.stringify({ rallies: [] }) })).status, 200);
  assert.equal((await vic.get(`/videos/${a1.id}/results/points.json`)).status, 404);
  assert.equal((await ann.get(`/videos/${a1.id}/results/points.json`)).status, 200);
  assert.equal((await vic.get(`/videos/${a1.id}/rally-labels`)).status, 404);
  // 超级管理员看得到所有人的
  const bossList = (await (await boss.get("/videos")).json()).videos.map((v) => v.id);
  for (const v of [o1, o2, a1, a2]) assert.ok(bossList.includes(v.id));

  // ann 只能改自己的
  assert.equal((await ann.send("PATCH", `/videos/${o1.id}`, { title: "x" })).status, 403);
  assert.equal((await ann.send("DELETE", `/videos/${o1.id}`)).status, 403);
  assert.equal((await ann.post(`/videos/${o1.id}/job/retry`)).status, 403);
  assert.equal((await ann.send("PATCH", `/videos/${a1.id}`, { title: "我的比赛" })).status, 200);
  assert.equal((await ann.put(`/videos/${a1.id}/players`, { names: { 1: "Ann" } })).status, 200);
  assert.equal((await ann.put(`/videos/${a1.id}/rally-labels`, { items: [{ frame: 5, winner: "near" }] })).status, 200);
  // 设置、Mac 状态、标注材料都不行
  for (const p of ["/status", "/labeling/videos"]) assert.equal((await ann.get(p)).status, 403, p);
  assert.equal((await ann.send("PUT", "/ai/settings", { defaultModel: "x" })).status, 403);
  assert.equal((await ann.post(`/videos/${a1.id}/annot`)).status, 403);
  assert.equal((await ann.get(`/videos/${a1.id}/file`)).status, 403);

  // 处理完、删掉没传完的那段：名额空出来，可以重跑
  assert.equal((await post(`/videos/${a1.id}/job`, { status: "done" })).status, 200);
  assert.equal((await ann.send("DELETE", `/videos/${a2.id}`)).status, 200);
  assert.equal((await ann.post(`/videos/${a1.id}/job/retry`, { only: "cards" })).status, 200);
  const a4 = await upload(ann, "a4.mp4");
  assert.equal((await upload(ann, "a5.mp4", false)).status, 429); // A1（重跑）+ A4
  assert.equal((await ann.post(`/videos/${a4.id}/job/retry`)).status, 200); // 已经在排队的不另算

  // AI 报告：只给默认模型和这个月的次数；用完了就不让生成
  const info = await (await ann.get("/ai")).json();
  assert.equal(info.models.length, 1);
  assert.equal(info.usage, undefined);
  assert.deepEqual(info.quota, { reports: { used: 0, limit: 10 }, asks: { used: 0, limit: 30 } });
  const review = { system: "s", user: "u", ask: "{question}", max_tokens: 1000 };
  assert.equal((await call(`/videos/${a1.id}/results/review.request.json`, { method: "PUT", headers: json, body: JSON.stringify(review) })).status, 200);
  const now = new Date().toISOString();
  await sql(`INSERT INTO ai_usage (user_id, video_id, kind, model, input_tokens, output_tokens, cost_usd, actor, created_at) VALUES ${
    Array.from({ length: 10 }, () => `('owner', '${a1.id}', 'report', 'm', 1, 1, 0.01, 'ann@example.com', '${now}')`).join(", ")}, ('owner', '${a1.id}', 'report', 'm', 1, 1, 0.01, 'ann@example.com', '2020-01-01T00:00:00.000Z')`);
  assert.deepEqual((await (await ann.get("/ai")).json()).quota.reports, { used: 10, limit: 10 });
  const no = await ann.post(`/videos/${a1.id}/review`, {});
  assert.equal(no.status, 429, await no.text());
  assert.equal((await ann.post(`/videos/${o1.id}/review`, {})).status, 403); // 别人的录像更不行
  const row = (await (await boss.get("/admin/users")).json()).users.find((u) => u.email === "ann@example.com");
  assert.deepEqual([row.videos, row.monthReports, row.monthAsks], [2, 10, 0]);

  // 关掉以后：自己的录像还能看，不能再改、再传
  await boss.post("/admin/users", { email: "ann@example.com", canAnalyze: false });
  assert.equal((await ann.get(`/videos/${a1.id}`)).status, 200);
  assert.equal((await ann.send("PATCH", `/videos/${a1.id}`, { title: "y" })).status, 403);
  assert.equal((await ann.post("/videos", { filename: "a.mp4", size: 10 })).status, 403);
});
