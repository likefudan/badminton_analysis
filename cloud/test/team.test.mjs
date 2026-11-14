// 集成测试：Google 登录（用假的 Google）、成员白名单、领任务、标注归属。
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "node:http";

const PORT = 8797;
const ORIGIN = `http://127.0.0.1:${PORT}`;
const BASE = `${ORIGIN}/api/v1`;
// wrangler dev 把请求的域名换成 wrangler.toml 里的网站域名，所以回调地址是它
const CALLBACK = "http://ba.llmat.dev/api/v1/auth/google/callback";
const TOKEN = "test-token";
const state = mkdtempSync(join(tmpdir(), "ba-team-"));
let server;

// 假的 Google：授权码就是邮箱，换回来的 id_token 里写着这个邮箱
const G_PORT = 8796;
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
  server = spawn("npx", ["wrangler", "dev", "--local", "--port", String(PORT), "--inspector-port", "9331", "--persist-to", state,
    "--var", `ACCESS_TOKEN:${TOKEN}`, "--var", "GOOGLE_CLIENT_ID:test-client", "--var", "GOOGLE_CLIENT_SECRET:test-secret",
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

// 一段已经准备好标注材料的录像：2 个回合（帧 10–12、50–53），45 拍击球
async function prepared() {
  const data = randomBytes(500);
  let v;
  for (let i = 0; i < 10 && !(v && v.id); i++) {
    // 本地模拟的 R2 刚启动时偶尔还没就绪
    if (i) await new Promise((r) => setTimeout(r, 500));
    v = await (await post("/videos", { filename: "team.mp4", size: data.length, contentType: "video/mp4", meta: { fps: 30 } })).json();
  }
  assert.ok(v.id, JSON.stringify(v));
  await call(`/videos/${v.id}/parts/1`, { method: "PUT", body: data });
  const done = await post(`/videos/${v.id}/complete`);
  assert.equal(done.status, 200, await done.text());
  const req = await post(`/videos/${v.id}/annot`);
  assert.equal(req.status, 200, await req.text());
  assert.equal((await (await post("/node/claim", { node: "mac", tasks: ["annotate"] })).json()).task, "annotate");
  const rallies = { rallies: [{ start: 10, offsets: [[0, 1], [1, 1], [2, 1]] }, { start: 50, offsets: [[0, 1], [1, 1], [2, 1], [3, 1]] }] };
  await call(`/videos/${v.id}/annot/files/state.json`, { method: "PUT", body: JSON.stringify(rallies) });
  await post(`/videos/${v.id}/annot/job`, { status: "ready" });
  await post(`/videos/${v.id}/strokes`);
  assert.equal((await (await post("/node/claim", { node: "mac", tasks: ["strokes"] })).json()).task, "strokes");
  const strokes = { strokes: Array.from({ length: 45 }, (_, i) => ({ frame: 100 + i * 10 })) };
  await call(`/videos/${v.id}/strokes/files/strokes.state.json`, { method: "PUT", body: JSON.stringify(strokes) });
  await post(`/videos/${v.id}/strokes/job`, { status: "ready" });
  return v.id;
}

test("Google 登录、白名单、领任务、标注归属、超时放回、重新准备", async () => {
  assert.deepEqual(await (await fetch(BASE + "/auth/config")).json(), { google: true });

  const vid = await prepared();

  // 登录出错都回到首页带上原因，不发会话
  const wrong = await fetch(`${BASE}/auth/google/callback?code=a@x.com&state=nope`, { redirect: "manual", headers: { cookie: "ba_oauth=other" } });
  assert.equal(wrong.headers.get("location"), "/?login_error=state");
  assert.equal(cookieFrom(wrong, "ba_sid"), "");
  for (const flag of ["aud", "unverified"]) {
    const r = await login(`mallory@x.com|${flag}`);
    assert.equal(r.location, "/?login_error=google");
    assert.equal(r.cookie, "");
  }

  // 以前（多人之前）用口令标的帧：没有署名
  const putLabel = (who, frame) => who.put(`/videos/${vid}/labels`, { items: [{ frame, visibility: 1, x: 5, y: 6, status: "human" }] });
  for (const frame of [10, 11, 12]) {
    assert.equal((await call(`/videos/${vid}/labels`, { method: "PUT", headers: json, body: JSON.stringify({ items: [{ frame, visibility: 1, x: 1, y: 2, status: "human" }] }) })).status, 200);
  }

  // 第一个人登录：还没被邀请，什么都看不了；输对口令成为管理员
  const a = await login("Alice@Example.com");
  assert.equal(a.location, "/");
  const alice = as(a.cookie);
  let me = await (await alice.get("/me")).json();
  assert.deepEqual([me.email, me.name, me.role, me.status, me.via], ["alice@example.com", "Alice", "annotator", "pending", "google"]);
  assert.equal((await alice.get("/tasks")).status, 403);
  assert.equal((await alice.get("/videos")).status, 403);
  assert.equal((await alice.post("/auth/claim-admin", { password: "nope" })).status, 403);
  // 没有 x-ba 头的写请求不认 cookie（别的网站借不了登录状态）
  assert.equal((await fetch(BASE + "/auth/claim-admin", { method: "POST", headers: { cookie: a.cookie, ...json }, body: JSON.stringify({ password: TOKEN }) })).status, 401);
  assert.equal((await alice.post("/auth/claim-admin", { password: TOKEN })).status, 200);
  me = await (await alice.get("/me")).json();
  assert.deepEqual([me.role, me.status], ["admin", "active"]);
  assert.equal((await alice.get("/videos")).status, 200);

  // 任务：2 个回合 + 击球 45 拍分成 20 和 25；已经标完的回合直接算完成
  let list = await (await alice.get("/tasks")).json();
  const brief = (l) => l.tasks.map((t) => [t.kind, t.start, t.end, t.items, t.done, t.status]);
  assert.deepEqual(brief(list), [
    ["shuttle", 10, 12, 3, 3, "approved"], ["shuttle", 50, 53, 4, 0, "open"],
    ["stroke", 100, 290, 20, 0, "open"], ["stroke", 300, 540, 25, 0, "open"],
  ]);
  assert.equal(list.tasks[0].fps, 30);
  const [, rally2, strokes1, strokes2] = list.tasks;

  // 邀请 bob；carol 自己登录后出现在申请里，管理员放行
  assert.equal((await alice.post("/admin/users", { email: "not-an-email" })).status, 400);
  assert.equal((await alice.post("/admin/users", { email: "Bob@Example.com" })).status, 200);
  const bob = as((await login("bob@example.com")).cookie);
  assert.deepEqual((({ role, status }) => [role, status])(await (await bob.get("/me")).json()), ["annotator", "active"]);
  const c = await login("carol@example.com");
  const carol = as(c.cookie);
  assert.equal((await carol.get("/tasks")).status, 403);
  let users = (await (await alice.get("/admin/users")).json()).users;
  assert.deepEqual(users.map((u) => [u.email, u.role, u.status]), [
    ["carol@example.com", "annotator", "pending"], ["alice@example.com", "admin", "active"], ["bob@example.com", "annotator", "active"],
  ]);
  assert.equal(users[1].frames, 3); // 以前没署名的标注记到第一个管理员名下
  assert.equal((await alice.post("/admin/users", { email: "carol@example.com", status: "active" })).status, 200);
  assert.equal((await carol.get("/tasks")).status, 200);

  // 标注员：看不到录像、设置、成员，不能改标注材料；没领任务读不了材料
  for (const path of ["/videos", `/videos/${vid}`, "/ai", "/status", "/admin/users"]) assert.equal((await bob.get(path)).status, 403, path);
  assert.equal((await bob.post("/admin/users", { email: "eve@example.com" })).status, 403);
  assert.equal((await bob.post(`/videos/${vid}/annot`)).status, 403);
  assert.equal((await bob.put(`/videos/${vid}/annot/files/state.json`, {})).status, 403);
  assert.equal((await bob.get(`/videos/${vid}/annot/files/state.json`)).status, 403);
  list = await (await bob.get("/tasks")).json();
  assert.deepEqual(brief(list).map((t) => t[5]), ["open", "open", "open"]); // 别人完成的不显示

  // 领取：同一个任务只有一个人领得到
  assert.equal((await bob.post(`/tasks/${rally2.id}/claim`)).status, 200);
  const lost = await carol.post(`/tasks/${rally2.id}/claim`);
  assert.equal(lost.status, 409);
  assert.match((await lost.json()).error, /别人领走/);
  assert.equal((await bob.get(`/videos/${vid}/annot/files/state.json`)).status, 200);
  assert.equal((await bob.get(`/videos/${vid}/labels`)).status, 200);
  assert.equal((await putLabel(bob, 50)).status, 200);
  assert.equal((await putLabel(bob, 10)).status, 403);        // 不在自己的任务里
  assert.equal((await putLabel(bob, 999)).status, 403);       // 不在任何任务里
  assert.equal((await putLabel(carol, 50)).status, 403);      // 没领任务
  assert.equal((await putLabel(alice, 51)).status, 409);      // 管理员也不能写别人正在标的
  assert.equal((await putLabel(alice, 10)).status, 200);      // 别的地方管理员可以改
  assert.equal((await bob.put(`/videos/${vid}/stroke-labels`, { items: [{ frame: 100, type: "杀球" }] })).status, 403);
  list = await (await carol.get("/tasks")).json();
  assert.ok(!list.tasks.some((t) => t.id === rally2.id)); // 别人领走的不显示

  // CSV 带标注人
  const csv = (await (await alice.get(`/videos/${vid}/labels?format=csv`)).text()).replace(/^﻿/, "").trim().split("\r\n");
  assert.equal(csv[0], "frame,visibility,x,y,status,annotator,updated_at");
  assert.match(csv[1], /^10,1,5,6,human,alice@example\.com,/);
  assert.match(csv[2], /^11,1,1,2,human,alice@example\.com,/);
  assert.match(csv[4], /^50,1,5,6,human,bob@example\.com,/);

  // 提交后不能再写；管理员退回后又可以；通过
  assert.equal((await carol.post(`/tasks/${rally2.id}/submit`)).status, 409);
  assert.equal((await bob.post(`/tasks/${rally2.id}/submit`)).status, 200);
  assert.equal((await putLabel(bob, 51)).status, 403);
  assert.equal((await bob.post(`/tasks/${rally2.id}/return`)).status, 403);
  assert.equal((await putLabel(alice, 51)).status, 200); // 提交之后管理员可以改
  list = await (await alice.post(`/tasks/${rally2.id}/return`)).json();
  let t = list.tasks.find((x) => x.id === rally2.id);
  assert.deepEqual([t.status, t.assignee.email, t.done], ["claimed", "bob@example.com", 2]);
  assert.equal((await putLabel(bob, 52)).status, 200);
  await bob.post(`/tasks/${rally2.id}/submit`);
  list = await (await alice.post(`/tasks/${rally2.id}/approve`)).json();
  assert.equal(list.tasks.find((x) => x.id === rally2.id).status, "approved");

  // 太久没保存：任务放回大厅，别人可以领；原来的人不能再写
  assert.equal((await bob.post(`/tasks/${strokes1.id}/claim`)).status, 200);
  const putStroke = (who, frame) => who.put(`/videos/${vid}/stroke-labels`, { items: [{ frame, type: "杀球", player_id: 1 }] });
  assert.equal((await putStroke(bob, 100)).status, 200);
  sql(`UPDATE tasks SET lease_until = '2000-01-01T00:00:00.000Z' WHERE id = ${strokes1.id}`);
  assert.equal((await putStroke(bob, 110)).status, 403);
  assert.equal((await carol.post(`/tasks/${strokes1.id}/claim`)).status, 200);
  assert.equal((await putStroke(carol, 110)).status, 200);
  const scsv = (await (await alice.get(`/videos/${vid}/stroke-labels?format=csv`)).text()).replace(/^﻿/, "").trim().split("\r\n");
  assert.equal(scsv[0], "frame,player_id,type,tags,note,annotator,updated_at");
  assert.match(scsv[1], /^100,1,杀球,,,bob@example\.com,/);
  assert.match(scsv[2], /^110,1,杀球,,,carol@example\.com,/);

  // 同时最多领 3 个
  // （carol 手上 1 个；再造不出 3 个任务，所以只检查放弃后可以重新领）
  assert.equal((await carol.post(`/tasks/${strokes2.id}/claim`)).status, 200);
  assert.equal((await carol.post(`/tasks/${strokes2.id}/release`)).status, 200);

  // 重新准备击球小视频（击球帧变了）：carol 手上的任务保留，没人领的按新材料重新分
  await post(`/videos/${vid}/strokes`);
  await post("/node/claim", { node: "mac", tasks: ["strokes"] });
  const again = { strokes: Array.from({ length: 30 }, (_, i) => ({ frame: 105 + i * 20 })) }; // 105 … 685
  await call(`/videos/${vid}/strokes/files/strokes.state.json`, { method: "PUT", body: JSON.stringify(again) });
  await new Promise((r) => setTimeout(r, 5)); // 准备完成的时间要比上次生成任务的时间晚
  await post(`/videos/${vid}/strokes/job`, { status: "ready" });
  list = await (await alice.get("/tasks")).json();
  assert.deepEqual(brief(list).filter((x) => x[0] === "stroke"), [
    ["stroke", 100, 290, 10, 2, "claimed"],   // 范围不变，里面现在有 10 拍
    ["stroke", 305, 685, 20, 0, "open"],
  ]);

  // 停用 bob：会话失效；删除 carol：手上的任务放回大厅，标注还在
  assert.equal((await alice.post("/admin/users", { email: "alice@example.com", status: "disabled" })).status, 409); // 不能停用自己
  assert.equal((await alice.post("/admin/users", { email: "bob@example.com", status: "disabled" })).status, 200);
  assert.equal((await bob.get("/me")).status, 401);
  users = (await (await alice.post("/admin/users/remove", { email: "carol@example.com" })).json()).users;
  assert.deepEqual(users.map((u) => [u.email, u.status, u.frames, u.strokes]), [["alice@example.com", "active", 4, 0], ["bob@example.com", "disabled", 2, 1]]);
  list = await (await alice.get("/tasks")).json();
  assert.equal(list.tasks.find((x) => x.id === strokes1.id).assignee, null);
  assert.ok(list.tasks.filter((x) => x.kind === "stroke").every((x) => x.status === "open"));

  // 用口令（管理员的老办法、Mac 节点）照常能用，标注记在第一个管理员名下
  assert.equal((await call("/tasks")).status, 200);
  assert.equal((await call(`/videos/${vid}/labels`, { method: "PUT", headers: json, body: JSON.stringify({ items: [{ frame: 53, visibility: 0, status: "human" }] }) })).status, 200);
  const last = (await (await call(`/videos/${vid}/labels?format=csv`)).text()).trim().split("\r\n").pop();
  assert.match(last, /^53,0,0,0,human,alice@example\.com,/);
  // 节点读到的标注格式没有变
  assert.deepEqual((await (await call(`/videos/${vid}/labels`)).json()).labels["50"], [1, 5, 6, "human"]);

  // 退出
  assert.equal((await alice.post("/auth/logout")).status, 200);
  assert.equal((await alice.get("/me")).status, 401);
});
