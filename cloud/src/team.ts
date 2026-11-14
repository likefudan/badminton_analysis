// 多人标注：Google 登录、成员（白名单）、领任务。
// 登录后浏览器拿到一个随机的会话 cookie（库里只存它的 SHA-256）。没被邀请的账号登录后是 pending，
// 管理员在「成员」页点一下就能放行。口令（ACCESS_TOKEN）仍然有效：Mac 节点用它，管理员也可以用它登录，
// 第一个管理员就是用 Google 登录后再输一次口令认领的。

import { type Env, HttpError, json, sha256, hex, sameSecret, annotKey, strokesKey } from "./index";

export type Who = {
  email: string;                 // 用口令登录时是空的
  name: string;
  picture: string;
  role: "admin" | "annotator";
  status: "active" | "pending" | "disabled";
  via: "token" | "google";
  mediaOnly?: boolean;           // 口令登录留下的只读 cookie：只用来取图片和视频，不算登录
};

const SID_COOKIE = "ba_sid";
const STATE_COOKIE = "ba_oauth";
const LEGACY_COOKIE = "ba_session";
const SESSION_MS = 30 * 24 * 3600 * 1000;
export const TASK_LEASE_MS = 2 * 3600 * 1000;  // 领了任务这么久没有保存，任务放回大厅
const MAX_HELD = 3;                            // 标注员同时最多领几个任务
const MAX_PENDING = 100;                       // 没被邀请的账号最多记这么多个
const STROKES_PER_TASK = 20;

const googleReady = (env: Env) => Boolean(env.GOOGLE_CLIENT_ID && env.GOOGLE_CLIENT_SECRET);
const cookieOf = (req: Request, name: string) => {
  const c = (req.headers.get("cookie") ?? "").split(/;\s*/).find((x) => x.startsWith(name + "="));
  return c ? c.slice(name.length + 1) : "";
};
const randomHex = (n: number) => hex(crypto.getRandomValues(new Uint8Array(n)));
const hashOf = async (text: string) => hex(await sha256(text));
const secureFlag = (req: Request) => (new URL(req.url).protocol === "https:" ? " Secure;" : "");

// ---------- 是谁 ----------

export async function identify(req: Request, env: Env): Promise<Who | null> {
  const tokenUser: Who = { email: "", name: "管理员", picture: "", role: "admin", status: "active", via: "token" };
  const header = req.headers.get("authorization") ?? "";
  if (header.startsWith("Bearer ")) {
    if (!env.ACCESS_TOKEN) return null;
    const got = header.slice(7);
    if (await sameSecret(got, env.ACCESS_TOKEN)) return tokenUser;
    return env.NODE_TOKEN && (await sameSecret(got, env.NODE_TOKEN)) ? tokenUser : null;
  }
  const sid = cookieOf(req, SID_COOKIE);
  if (sid) {
    // 别的网站发不出带这个头的请求（没有开放跨域），所以它们没法借 cookie 来改数据
    if (req.method !== "GET" && req.headers.get("x-ba") !== "1") return null;
    const row = await env.DB.prepare(
      `SELECT u.email, u.name, u.picture, u.role, u.status FROM sessions s JOIN users u ON u.email = s.email
       WHERE s.id_hash = ? AND s.expires_at > ?`,
    )
      .bind(await hashOf(sid), new Date().toISOString())
      .first<Omit<Who, "via">>();
    if (row) return { ...row, via: "google" };
  }
  // 用口令登录的网页：图片和视频靠这个只读 cookie（值由口令算出）
  const legacy = cookieOf(req, LEGACY_COOKIE);
  if (legacy && req.method === "GET" && env.ACCESS_TOKEN && (await sameSecret(legacy, hex(await sha256("ba-session:" + env.ACCESS_TOKEN))))) return { ...tokenUser, mediaOnly: true };
  return null;
}

export const isAdmin = (who: Who) => who.role === "admin" && who.status === "active";

export const meJson = (who: Who) => ({ email: who.email, name: who.name, picture: who.picture, role: who.role, status: who.status, via: who.via });

// 标注记在谁名下：用口令登录的算第一个管理员
async function annotatorOf(env: Env, who: Who): Promise<string> {
  if (who.email) return who.email;
  const admin = await env.DB.prepare("SELECT email FROM users WHERE role = 'admin' AND status = 'active' ORDER BY created_at LIMIT 1").first<{ email: string }>();
  return admin?.email ?? ""; // 还没有管理员：先空着，第一个管理员认领时补上
}

// ---------- Google 登录 ----------

const redirect = (to: string, cookies: string[] = []) => {
  const res = new Response(null, { status: 302, headers: { location: to, "cache-control": "no-store" } });
  for (const c of cookies) res.headers.append("set-cookie", c);
  return res;
};
const loginError = (code: string, cookies: string[] = []) => redirect("/?login_error=" + code, cookies);

function googleStart(req: Request, env: Env, url: URL): Response {
  if (!googleReady(env)) return loginError("not_configured");
  const state = randomHex(16);
  const to = new URL(env.GOOGLE_AUTH_URL || "https://accounts.google.com/o/oauth2/v2/auth");
  to.searchParams.set("client_id", env.GOOGLE_CLIENT_ID!);
  to.searchParams.set("redirect_uri", url.origin + "/api/v1/auth/google/callback");
  to.searchParams.set("response_type", "code");
  to.searchParams.set("scope", "openid email profile");
  to.searchParams.set("state", state);
  to.searchParams.set("prompt", "select_account");
  return redirect(to.toString(), [`${STATE_COOKIE}=${state}; Path=/api/v1/auth; HttpOnly;${secureFlag(req)} SameSite=Lax; Max-Age=600`]);
}

function decodeJwtPayload(token: string): Record<string, unknown> | null {
  try {
    const part = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const bin = atob(part + "=".repeat((4 - (part.length % 4)) % 4));
    return JSON.parse(new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0))));
  } catch {
    return null;
  }
}

async function googleCallback(req: Request, env: Env, url: URL): Promise<Response> {
  const clear = [`${STATE_COOKIE}=; Path=/api/v1/auth; HttpOnly;${secureFlag(req)} SameSite=Lax; Max-Age=0`];
  if (!googleReady(env)) return loginError("not_configured", clear);
  const state = url.searchParams.get("state") ?? "", code = url.searchParams.get("code") ?? "";
  const want = cookieOf(req, STATE_COOKIE);
  if (url.searchParams.get("error")) return loginError("cancelled", clear);
  if (!code || !state || !want || !(await sameSecret(state, want))) return loginError("state", clear);
  // 授权码直接向 Google 换 id_token（走 HTTPS，对方就是 Google，所以不用再验签名）
  const res = await fetch(env.GOOGLE_TOKEN_URL || "https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      code, client_id: env.GOOGLE_CLIENT_ID!, client_secret: env.GOOGLE_CLIENT_SECRET!,
      redirect_uri: url.origin + "/api/v1/auth/google/callback", grant_type: "authorization_code",
    }),
  });
  const data = (await res.json().catch(() => null)) as { id_token?: string } | null;
  const claims = res.ok && data?.id_token ? decodeJwtPayload(data.id_token) : null;
  if (!claims) {
    console.error("google token exchange failed", res.status);
    return loginError("google", clear);
  }
  const email = typeof claims.email === "string" ? claims.email.trim().toLowerCase() : "";
  const issOk = claims.iss === "https://accounts.google.com" || claims.iss === "accounts.google.com";
  if (!email || claims.aud !== env.GOOGLE_CLIENT_ID || !issOk || claims.email_verified !== true || Number(claims.exp) * 1000 < Date.now())
    return loginError("google", clear);
  const name = typeof claims.name === "string" ? claims.name.slice(0, 80) : "";
  const picture = typeof claims.picture === "string" && claims.picture.startsWith("https://") ? claims.picture.slice(0, 500) : "";
  const now = new Date();
  const known = await env.DB.prepare("SELECT email FROM users WHERE email = ?").bind(email).first();
  if (known) {
    await env.DB.prepare("UPDATE users SET name = CASE WHEN ? = '' THEN name ELSE ? END, picture = ?, last_login_at = ? WHERE email = ?")
      .bind(name, name, picture, now.toISOString(), email)
      .run();
  } else {
    const pending = await env.DB.prepare("SELECT COUNT(*) AS n FROM users WHERE status = 'pending'").first<{ n: number }>();
    if ((pending?.n ?? 0) >= MAX_PENDING) return loginError("full", clear);
    await env.DB.prepare("INSERT OR IGNORE INTO users (email, name, picture, role, status, created_at, last_login_at) VALUES (?, ?, ?, 'annotator', 'pending', ?, ?)")
      .bind(email, name, picture, now.toISOString(), now.toISOString())
      .run();
  }
  const sid = randomHex(32);
  await env.DB.batch([
    env.DB.prepare("DELETE FROM sessions WHERE expires_at < ?").bind(now.toISOString()),
    env.DB.prepare("INSERT INTO sessions (id_hash, email, created_at, expires_at) VALUES (?, ?, ?, ?)")
      .bind(await hashOf(sid), email, now.toISOString(), new Date(now.getTime() + SESSION_MS).toISOString()),
  ]);
  return redirect("/", [...clear, `${SID_COOKIE}=${sid}; Path=/api/; HttpOnly;${secureFlag(req)} SameSite=Lax; Max-Age=${SESSION_MS / 1000}`]);
}

async function logout(req: Request, env: Env): Promise<Response> {
  const sid = cookieOf(req, SID_COOKIE);
  if (sid) await env.DB.prepare("DELETE FROM sessions WHERE id_hash = ?").bind(await hashOf(sid)).run();
  const res = json({ ok: true });
  res.headers.append("set-cookie", `${SID_COOKIE}=; Path=/api/; HttpOnly;${secureFlag(req)} SameSite=Lax; Max-Age=0`);
  res.headers.append("set-cookie", `${LEGACY_COOKIE}=; Path=/api/; HttpOnly;${secureFlag(req)} SameSite=Strict; Max-Age=0`);
  return res;
}

// 用 Google 登录后输一次口令：这个账号成为管理员，以前没有署名的标注记到它名下
async function claimAdmin(req: Request, env: Env, who: Who): Promise<Response> {
  if (who.via !== "google") throw new HttpError(400, "先用 Google 登录");
  const body = (await req.json().catch(() => null)) as { password?: unknown } | null;
  const password = typeof body?.password === "string" ? body.password : "";
  if (!env.ACCESS_TOKEN || !password || !(await sameSecret(password, env.ACCESS_TOKEN))) throw new HttpError(403, "口令不对");
  await env.DB.batch([
    env.DB.prepare("UPDATE users SET role = 'admin', status = 'active' WHERE email = ?").bind(who.email),
    env.DB.prepare("UPDATE labels SET annotator = ? WHERE annotator = ''").bind(who.email),
    env.DB.prepare("UPDATE stroke_labels SET annotator = ? WHERE annotator = ''").bind(who.email),
  ]);
  return json({ ok: true });
}

// 不用登录就能访问的接口；不是这些就返回 null
export async function publicApi(req: Request, env: Env, path: string, url: URL): Promise<Response | null> {
  const m = req.method;
  if (path === "/auth/config" && m === "GET") return json({ google: googleReady(env) });
  if (path === "/auth/google/start" && m === "GET") return googleStart(req, env, url);
  if (path === "/auth/google/callback" && m === "GET") return googleCallback(req, env, url);
  if (path === "/auth/logout" && m === "POST") return logout(req, env);
  return null;
}

// ---------- 任务 ----------

type TaskRow = {
  id: number; video_id: string; kind: "shuttle" | "stroke"; idx: number; start_frame: number; end_frame: number; n_items: number;
  status: "open" | "claimed" | "submitted" | "approved"; assignee: string; claimed_at: string | null; lease_until: string | null;
  submitted_at: string | null; active: number; updated_at: string;
};

async function expireLeases(env: Env, now: string): Promise<void> {
  await env.DB.prepare("UPDATE tasks SET status = 'open', assignee = '', lease_until = NULL, updated_at = ? WHERE status = 'claimed' AND lease_until < ?")
    .bind(now, now)
    .run();
}

async function readJson(env: Env, key: string): Promise<any | null> {
  const obj = await env.BUCKET.get(key);
  if (!obj) return null;
  try {
    return JSON.parse(await obj.text());
  } catch {
    return null;
  }
}

// 把一段录像的任务和它现在的标注材料对齐。已经有人领过的任务保留；没人领的按现在的材料重新生成。
async function syncVideoTasks(env: Env, videoId: string, kind: "shuttle" | "stroke", stamp: string): Promise<void> {
  const now = new Date().toISOString();
  const existing = (await env.DB.prepare("SELECT * FROM tasks WHERE video_id = ? AND kind = ?").bind(videoId, kind).all<TaskRow>()).results;
  const stmts: D1PreparedStatement[] = [];
  const insert = (idx: number, start: number, end: number, n: number, done: boolean) =>
    stmts.push(
      env.DB.prepare(
        `INSERT OR IGNORE INTO tasks (video_id, kind, idx, start_frame, end_frame, n_items, status, active, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)`,
      ).bind(videoId, kind, idx, start, end, n, done ? "approved" : "open", now),
    );
  if (kind === "shuttle") {
    const state = await readJson(env, annotKey(videoId, "state.json"));
    const rallies: { start: number; n: number }[] = Array.isArray(state?.rallies)
      ? state.rallies.map((r: any) => ({ start: Number(r?.start), n: Array.isArray(r?.offsets) ? r.offsets.length : 0 })).filter((r: any) => Number.isInteger(r.start) && r.n > 0)
      : [];
    const labelled = (await env.DB.prepare("SELECT frame FROM labels WHERE video_id = ? ORDER BY frame").bind(videoId).all<{ frame: number }>()).results.map((r) => r.frame);
    const byStart = new Map(existing.map((t) => [t.start_frame, t]));
    const keep = new Set<number>();
    rallies.forEach((r, idx) => {
      const old = byStart.get(r.start), end = r.start + r.n - 1;
      if (old) {
        keep.add(old.id);
        stmts.push(env.DB.prepare("UPDATE tasks SET idx = ?, end_frame = ?, n_items = ?, active = 1, updated_at = ? WHERE id = ?").bind(idx, end, r.n, now, old.id));
      } else {
        // 生成任务之前就已经标完的回合（以前自己标的）直接算完成
        insert(idx, r.start, end, r.n, labelled.filter((f) => f >= r.start && f <= end).length >= r.n);
      }
    });
    for (const t of existing) {
      if (keep.has(t.id)) continue;
      stmts.push(t.status === "open" ? env.DB.prepare("DELETE FROM tasks WHERE id = ?").bind(t.id) : env.DB.prepare("UPDATE tasks SET active = 0, updated_at = ? WHERE id = ?").bind(now, t.id));
    }
    stmts.push(env.DB.prepare("UPDATE videos SET annot_tasks_at = ? WHERE id = ?").bind(stamp, videoId));
  } else {
    const state = await readJson(env, strokesKey(videoId, "strokes.state.json"));
    const frames: number[] = Array.isArray(state?.strokes)
      ? [...new Set<number>(state.strokes.map((s: any) => Number(s?.frame)).filter((f: number) => Number.isInteger(f) && f >= 0))].sort((a, b) => a - b)
      : [];
    const labelled = new Set((await env.DB.prepare("SELECT frame FROM stroke_labels WHERE video_id = ?").bind(videoId).all<{ frame: number }>()).results.map((r) => r.frame));
    const held = existing.filter((t) => t.status !== "open").sort((a, b) => a.start_frame - b.start_frame);
    for (const t of existing) if (t.status === "open") stmts.push(env.DB.prepare("DELETE FROM tasks WHERE id = ?").bind(t.id));
    for (const t of held) {
      const n = frames.filter((f) => f >= t.start_frame && f <= t.end_frame).length;
      stmts.push(env.DB.prepare("UPDATE tasks SET n_items = ?, active = ?, updated_at = ? WHERE id = ?").bind(n || t.n_items, n ? 1 : 0, now, t.id));
    }
    // 没被保留的任务盖住的拍，按时间顺序每 20 拍一组；被保留的任务隔开的两边不合成一组
    const runs: number[][] = [[]];
    let h = 0;
    for (const f of frames) {
      while (h < held.length && held[h].end_frame < f) {
        h++;
        if (runs[runs.length - 1].length) runs.push([]);
      }
      if (h < held.length && f >= held[h].start_frame) continue;
      runs[runs.length - 1].push(f);
    }
    let idx = held.length;
    for (const run of runs) {
      for (let i = 0; i < run.length; ) {
        let n = Math.min(STROKES_PER_TASK, run.length - i);
        if (run.length - i - n < STROKES_PER_TASK / 2) n = run.length - i; // 剩下不到半组就并进来
        const part = run.slice(i, i + n);
        insert(idx++, part[0], part[part.length - 1], part.length, part.every((f) => labelled.has(f)));
        i += n;
      }
    }
    stmts.push(env.DB.prepare("UPDATE videos SET strokes_tasks_at = ? WHERE id = ?").bind(stamp, videoId));
  }
  for (let i = 0; i < stmts.length; i += 400) await env.DB.batch(stmts.slice(i, i + 400));
}

async function syncTasks(env: Env): Promise<void> {
  const rows = await env.DB.prepare(
    `SELECT id, annot_status, annot_updated_at, annot_tasks_at, strokes_status, strokes_updated_at, strokes_tasks_at FROM videos
     WHERE (annot_status = 'ready' AND (annot_tasks_at IS NULL OR annot_tasks_at < COALESCE(annot_updated_at, '')))
        OR (strokes_status = 'ready' AND (strokes_tasks_at IS NULL OR strokes_tasks_at < COALESCE(strokes_updated_at, '')))`,
  ).all<{ id: string; annot_status: string; annot_updated_at: string | null; annot_tasks_at: string | null; strokes_status: string; strokes_updated_at: string | null; strokes_tasks_at: string | null }>();
  for (const v of rows.results) {
    if (v.annot_status === "ready" && (v.annot_tasks_at === null || v.annot_tasks_at < (v.annot_updated_at ?? ""))) await syncVideoTasks(env, v.id, "shuttle", v.annot_updated_at ?? "");
    if (v.strokes_status === "ready" && (v.strokes_tasks_at === null || v.strokes_tasks_at < (v.strokes_updated_at ?? ""))) await syncVideoTasks(env, v.id, "stroke", v.strokes_updated_at ?? "");
  }
}

async function listTasks(env: Env, who: Who): Promise<Response> {
  const now = new Date().toISOString();
  await syncTasks(env);
  await expireLeases(env, now);
  const admin = isAdmin(who);
  const me = await annotatorOf(env, who);
  const rows = await env.DB.prepare(
    `SELECT t.*, v.title, v.filename, v.meta, COALESCE(u.name, '') AS assignee_name,
       CASE t.kind WHEN 'shuttle' THEN (SELECT COUNT(*) FROM labels l WHERE l.video_id = t.video_id AND l.frame BETWEEN t.start_frame AND t.end_frame)
                   ELSE (SELECT COUNT(*) FROM stroke_labels l WHERE l.video_id = t.video_id AND l.frame BETWEEN t.start_frame AND t.end_frame) END AS done
     FROM tasks t JOIN videos v ON v.id = t.video_id LEFT JOIN users u ON u.email = t.assignee
     WHERE t.active = 1 AND ((t.kind = 'shuttle' AND v.annot_status = 'ready') OR (t.kind = 'stroke' AND v.strokes_status = 'ready'))
       AND (? = 1 OR t.status = 'open' OR t.assignee = ?)
     ORDER BY v.created_at DESC, t.kind, t.start_frame`,
  )
    .bind(admin ? 1 : 0, me)
    .all<TaskRow & { title: string; filename: string; meta: string; assignee_name: string; done: number }>();
  const fpsOf = (meta: string) => {
    try {
      const fps = Number(JSON.parse(meta).fps);
      return fps > 0 ? fps : null;
    } catch {
      return null;
    }
  };
  return json({
    now,
    me,
    maxHeld: admin ? null : MAX_HELD,
    tasks: rows.results.map((t) => ({
      id: t.id, videoId: t.video_id, video: t.title || t.filename, kind: t.kind, idx: t.idx, start: t.start_frame, end: t.end_frame,
      items: t.n_items, done: Math.min(t.done, t.n_items), fps: fpsOf(t.meta), status: t.status, mine: t.assignee !== "" && t.assignee === me,
      assignee: t.assignee ? { email: t.assignee, name: t.assignee_name } : null,
      leaseUntil: t.lease_until, submittedAt: t.submitted_at,
    })),
  });
}

async function getTask(env: Env, id: number): Promise<TaskRow> {
  const t = await env.DB.prepare("SELECT * FROM tasks WHERE id = ? AND active = 1").bind(id).first<TaskRow>();
  if (!t) throw new HttpError(404, "没有这个任务");
  return t;
}

async function taskAction(req: Request, env: Env, who: Who, id: number, action: string): Promise<Response> {
  const now = new Date(), iso = now.toISOString(), lease = new Date(now.getTime() + TASK_LEASE_MS).toISOString();
  await expireLeases(env, iso);
  const t = await getTask(env, id);
  const me = await annotatorOf(env, who), admin = isAdmin(who);
  const mine = t.assignee === me && t.assignee !== "";
  if (action === "claim") {
    if (!admin) {
      const held = await env.DB.prepare("SELECT COUNT(*) AS n FROM tasks WHERE assignee = ? AND status = 'claimed' AND active = 1").bind(me).first<{ n: number }>();
      if ((held?.n ?? 0) >= MAX_HELD && !(mine && t.status === "claimed")) throw new HttpError(409, `手上已经有 ${MAX_HELD} 个任务，先提交或放弃一个`);
    }
    // 只有还没人领的才领得到：两个人同时点，只有一个成功
    const res = await env.DB.prepare(
      "UPDATE tasks SET status = 'claimed', assignee = ?, claimed_at = ?, lease_until = ?, updated_at = ? WHERE id = ? AND (status = 'open' OR (status = 'claimed' AND assignee = ?))",
    )
      .bind(me, iso, lease, iso, id, me)
      .run();
    if (!res.meta.changes) throw new HttpError(409, "这个任务刚被别人领走了");
  } else if (action === "release") {
    if (t.status !== "claimed" || !(mine || admin)) throw new HttpError(409, "这个任务现在不在你手上");
    await env.DB.prepare("UPDATE tasks SET status = 'open', assignee = '', lease_until = NULL, updated_at = ? WHERE id = ?").bind(iso, id).run();
  } else if (action === "submit") {
    if (t.status !== "claimed" || !mine) throw new HttpError(409, "这个任务现在不在你手上（可能太久没保存，已经放回大厅）");
    await env.DB.prepare("UPDATE tasks SET status = 'submitted', submitted_at = ?, lease_until = NULL, updated_at = ? WHERE id = ?").bind(iso, iso, id).run();
  } else if (action === "approve") {
    if (!admin) throw new HttpError(403, "只有管理员可以");
    if (t.status !== "submitted") throw new HttpError(409, "这个任务还没提交");
    await env.DB.prepare("UPDATE tasks SET status = 'approved', updated_at = ? WHERE id = ?").bind(iso, id).run();
  } else if (action === "return") {
    // 退回给原来的人重标；原来没有人（以前自己标的）就放回大厅
    if (!admin) throw new HttpError(403, "只有管理员可以");
    if (t.status !== "submitted" && t.status !== "approved") throw new HttpError(409, "这个任务还没提交");
    const body = (await req.json().catch(() => null)) as { open?: unknown } | null;
    if (t.assignee && body?.open !== true)
      await env.DB.prepare("UPDATE tasks SET status = 'claimed', lease_until = ?, updated_at = ? WHERE id = ?").bind(lease, iso, id).run();
    else await env.DB.prepare("UPDATE tasks SET status = 'open', assignee = '', lease_until = NULL, updated_at = ? WHERE id = ?").bind(iso, id).run();
  } else throw new HttpError(404, "没有这个接口");
  return listTasks(env, who);
}

// 标注员只能读自己领过任务的录像的标注材料
export async function requireTaskOn(env: Env, who: Who, videoId: string): Promise<void> {
  const row = await env.DB.prepare("SELECT 1 AS ok FROM tasks WHERE video_id = ? AND assignee = ? AND active = 1 LIMIT 1").bind(videoId, who.email).first();
  if (!row) throw new HttpError(403, "先在任务大厅领这段录像的任务");
}

// 写标注之前：标注员只能写自己手上任务范围里的帧；管理员不能写别人正在标的帧。
// 顺便给碰到的任务续期。返回这些标注记在谁名下。
export async function guardLabelWrite(env: Env, who: Who, videoId: string, kind: "shuttle" | "stroke", frames: number[]): Promise<string> {
  const now = new Date(), iso = now.toISOString();
  const me = await annotatorOf(env, who), admin = isAdmin(who);
  await expireLeases(env, iso);
  const tasks = (await env.DB.prepare("SELECT * FROM tasks WHERE video_id = ? AND kind = ? AND active = 1").bind(videoId, kind).all<TaskRow>()).results;
  const touched = new Set<number>();
  for (const f of frames) {
    const t = tasks.find((x) => f >= x.start_frame && f <= x.end_frame);
    if (admin) {
      if (t && t.status === "claimed" && t.assignee !== me) throw new HttpError(409, `这一段 ${t.assignee} 正在标`);
    } else if (!t || t.status !== "claimed" || t.assignee !== me) {
      throw new HttpError(403, "这一段不在你手上的任务里（可能太久没保存，已经放回大厅）");
    }
    if (t && t.status === "claimed" && t.assignee === me) touched.add(t.id);
  }
  if (touched.size) {
    const lease = new Date(now.getTime() + TASK_LEASE_MS).toISOString();
    await env.DB.batch([...touched].map((id) => env.DB.prepare("UPDATE tasks SET lease_until = ?, updated_at = ? WHERE id = ?").bind(lease, iso, id)));
  }
  return me;
}

// ---------- 成员（管理员） ----------

const EMAIL = /^[^\s@]{1,64}@[^\s@]{1,190}\.[^\s@]{2,}$/;

async function listUsers(env: Env): Promise<Response> {
  const users = await env.DB.prepare(
    `SELECT u.email, u.name, u.picture, u.role, u.status, u.created_at, u.last_login_at,
       (SELECT COUNT(*) FROM labels l WHERE l.annotator = u.email) AS frames,
       (SELECT COUNT(*) FROM stroke_labels l WHERE l.annotator = u.email) AS strokes,
       (SELECT COUNT(*) FROM tasks t WHERE t.assignee = u.email AND t.status = 'claimed' AND t.active = 1) AS held,
       (SELECT COUNT(*) FROM tasks t WHERE t.assignee = u.email AND t.status IN ('submitted', 'approved')) AS finished
     FROM users u ORDER BY CASE u.status WHEN 'pending' THEN 0 WHEN 'active' THEN 1 ELSE 2 END, u.created_at`,
  ).all<Record<string, unknown>>();
  return json({
    users: users.results.map((u) => ({
      email: u.email, name: u.name, picture: u.picture, role: u.role, status: u.status, createdAt: u.created_at, lastLoginAt: u.last_login_at,
      frames: u.frames, strokes: u.strokes, held: u.held, finished: u.finished,
    })),
  });
}

async function releaseTasksOf(env: Env, email: string): Promise<D1PreparedStatement[]> {
  const now = new Date().toISOString();
  return [
    env.DB.prepare("UPDATE tasks SET status = 'open', assignee = '', lease_until = NULL, updated_at = ? WHERE assignee = ? AND status = 'claimed'").bind(now, email),
    env.DB.prepare("DELETE FROM sessions WHERE email = ?").bind(email),
  ];
}

async function saveUser(req: Request, env: Env, who: Who, remove: boolean): Promise<Response> {
  const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
  const email = typeof body?.email === "string" ? body.email.trim().toLowerCase() : "";
  if (!EMAIL.test(email)) throw new HttpError(400, "邮箱不对");
  const old = await env.DB.prepare("SELECT role, status FROM users WHERE email = ?").bind(email).first<{ role: string; status: string }>();
  const role = body?.role === "admin" || body?.role === "annotator" ? body.role : old?.role ?? "annotator";
  const status = remove ? "disabled" : body?.status === "active" || body?.status === "disabled" ? body.status : old && old.status !== "pending" ? old.status : "active";
  const losesAdmin = old?.role === "admin" && old.status === "active" && (role !== "admin" || status !== "active");
  if (losesAdmin) {
    if (email === who.email) throw new HttpError(409, "不能停用或降级自己");
    const admins = await env.DB.prepare("SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND status = 'active'").first<{ n: number }>();
    if ((admins?.n ?? 0) <= 1) throw new HttpError(409, "至少要留一个管理员");
  }
  if (remove) {
    if (!old) throw new HttpError(404, "没有这个成员");
    await env.DB.batch([env.DB.prepare("DELETE FROM users WHERE email = ?").bind(email), ...(await releaseTasksOf(env, email))]);
  } else if (old) {
    await env.DB.batch([
      env.DB.prepare("UPDATE users SET role = ?, status = ? WHERE email = ?").bind(role, status, email),
      ...(status === "disabled" ? await releaseTasksOf(env, email) : []),
    ]);
  } else {
    await env.DB.prepare("INSERT INTO users (email, role, status, created_at) VALUES (?, ?, ?, ?)").bind(email, role, status, new Date().toISOString()).run();
  }
  return listUsers(env);
}

// 登录后的接口；不是这里管的就返回 null
export async function teamApi(req: Request, env: Env, path: string, who: Who): Promise<Response | null> {
  const m = req.method;
  if (path === "/auth/claim-admin" && m === "POST") return claimAdmin(req, env, who);
  if (!path.startsWith("/tasks") && !path.startsWith("/admin/")) return null;
  if (who.status !== "active") throw new HttpError(403, "这个账号还没有被邀请");
  if (path === "/tasks" && m === "GET") return listTasks(env, who);
  const match = path.match(/^\/tasks\/(\d{1,12})\/(claim|release|submit|approve|return)$/);
  if (match && m === "POST") return taskAction(req, env, who, Number(match[1]), match[2]);
  if (!isAdmin(who)) throw new HttpError(403, "只有管理员可以");
  if (path === "/admin/users" && m === "GET") return listUsers(env);
  if (path === "/admin/users" && m === "POST") return saveUser(req, env, who, false);
  if (path === "/admin/users/remove" && m === "POST") return saveUser(req, env, who, true);
  return null;
}
