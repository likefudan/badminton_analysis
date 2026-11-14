import { identify, publicApi, teamApi, requireTaskOn, guardLabelWrite, isAdmin, meJson, type Who } from "./team";

// ba.llmat.dev 后端。阶段 0：录像分片上传到 R2，可断点续传。
// 接口在 /api/v1 下，用 Bearer 口令登录；网页、将来的 App 和 Mac 处理节点共用。

export interface Env {
  BUCKET: R2Bucket;
  DB: D1Database;
  ASSETS: Fetcher;
  ACCESS_TOKEN: string;
  NODE_TOKEN?: string; // Mac 处理节点自己的钥匙，和登录口令分开；没设置就只认登录口令
  // AI 报告用的模型。只有 key 是秘密（secret），其余写在 wrangler.toml 的 [vars] 里。没有 key 就不出报告。
  AI_API_KEY?: string;
  ANTHROPIC_API_KEY?: string; // 旧名字，等同于 AI_API_KEY
  AI_ENDPOINT_ANTHROPIC?: string; // Claude 走的地址（后面补 /v1/messages），默认 https://api.anthropic.com
  AI_ENDPOINT_OPENAI?: string; // 其它模型走的地址（后面补 /chat/completions）
  AI_MODELS?: string; // 可选：JSON 数组 [{id, label, format}]，替换内置的模型列表
  LABEL_SYNC_DELAY_S?: string;
  // Google 登录（OAuth 客户端的 ID 和密钥）。没有设置时网站只能用口令登录。
  GOOGLE_CLIENT_ID?: string;
  GOOGLE_CLIENT_SECRET?: string;
  ADMIN_GOOGLE_EMAIL?: string; // 超级管理员：用这个 Google 邮箱登录就是管理员，别人不能停用或降级它
  GOOGLE_AUTH_URL?: string;  // 只在测试里改：假的 Google
  GOOGLE_TOKEN_URL?: string;
}

// 片越小，网络中断时白传的越少。R2 要求除最后一片外每片一样大、每片至少 5 MiB、最多 10000 片，
// 所以默认 8 MiB，文件超过 80 GB 时按 8 MiB 的整数倍放大。Worker 单个请求上限约 100 MB。
const PART_SIZE = 8 * 1024 * 1024;
const MAX_PARTS = 10000;
function partSizeFor(size: number): number {
  return PART_SIZE * Math.max(1, Math.ceil(size / (PART_SIZE * MAX_PARTS)));
}
const MAX_SIZE = 200 * 1024 ** 3;
const USER_ID = "owner"; // 第一版只有一个用户
const LEASE_MS = 10 * 60 * 1000;        // 处理节点这么久没有消息，任务可以被重新领取
const KEEP_RESULT_BYTES = 10 * 1024 ** 3; // 云端只留最近这么多结果，更早的只在 Mac 上
const SESSION_COOKIE = "ba_session";
// 网页上停手这么久（默认 60 秒）后，Mac 节点把标注同步回去；测试里设成 0
const labelSyncDelay = (env: Env) => (env.LABEL_SYNC_DELAY_S !== undefined && env.LABEL_SYNC_DELAY_S !== "" ? Number(env.LABEL_SYNC_DELAY_S) : 60) * 1000;

type VideoRow = {
  id: string;
  user_id: string;
  filename: string;
  size: number;
  content_type: string;
  part_size: number;
  r2_key: string;
  upload_id: string;
  status: "uploading" | "uploaded";
  meta: string;
  created_at: string;
  completed_at: string | null;
  job_status: "none" | "queued" | "processing" | "done" | "failed" | "archived";
  job_stage: string;
  job_error: string;
  job_node: string;
  job_lease_until: string | null;
  job_updated_at: string | null;
  options: string;
  original_deleted_at: string | null;
  job_progress: string;
  job_started_at: string | null;
  title: string;
  annot_status: "none" | "queued" | "preparing" | "ready" | "failed";
  annot_stage: string;
  annot_error: string;
  annot_options: string;
  annot_lease_until: string | null;
  annot_updated_at: string | null;
  labels_changed_at: string | null;
  labels_synced_at: string | null;
  strokes_status: "none" | "queued" | "preparing" | "ready" | "failed";
  strokes_stage: string;
  strokes_error: string;
  strokes_lease_until: string | null;
  strokes_updated_at: string | null;
  stroke_labels_changed_at: string | null;
  stroke_labels_synced_at: string | null;
};

type ResultRow = { video_id: string; name: string; r2_key: string; size: number; content_type: string; created_at: string };

export class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });

export async function sha256(text: string): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
}

export const hex = (bytes: Uint8Array) => [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");

// 网页里的 <img> 和 <video> 带不了 Authorization 头，所以登录后发一个只有浏览器自己带的 cookie。
// 它的值由口令算出，不是口令本身；改口令后旧 cookie 自动失效。
async function sessionValue(env: Env): Promise<string> {
  return hex(await sha256("ba-session:" + env.ACCESS_TOKEN));
}

// 先各自做 SHA-256 再逐字节比较，比较时间不随口令内容变化
export async function sameSecret(got: string, want: string): Promise<boolean> {
  const [a, b] = await Promise.all([sha256(got), sha256(want)]);
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
  return diff === 0;
}

async function startSession(req: Request, env: Env): Promise<Response> {
  const secure = new URL(req.url).protocol === "https:" ? " Secure;" : "";
  const res = json({ ok: true });
  res.headers.set(
    "set-cookie",
    `${SESSION_COOKIE}=${await sessionValue(env)}; Path=/api/; HttpOnly;${secure} SameSite=Strict; Max-Age=31536000`,
  );
  return res;
}

function partCount(v: Pick<VideoRow, "size" | "part_size">): number {
  return Math.max(1, Math.ceil(v.size / v.part_size));
}

function expectedPartSize(v: VideoRow, n: number): number {
  return n < partCount(v) ? v.part_size : v.size - v.part_size * (partCount(v) - 1);
}

async function getVideo(env: Env, id: string): Promise<VideoRow> {
  const row = await env.DB.prepare("SELECT * FROM videos WHERE id = ? AND user_id = ?")
    .bind(id, USER_ID)
    .first<VideoRow>();
  if (!row) throw new HttpError(404, "没有这个录像");
  return row;
}

async function describe(env: Env, v: VideoRow) {
  const parts = await env.DB.prepare(
    "SELECT part_number, size FROM video_parts WHERE video_id = ? ORDER BY part_number",
  )
    .bind(v.id)
    .all<{ part_number: number; size: number }>();
  const results = await env.DB.prepare("SELECT name, size, content_type FROM results WHERE video_id = ? ORDER BY name")
    .bind(v.id)
    .all<Pick<ResultRow, "name" | "size" | "content_type">>();
  const usage = await env.DB.prepare(
    "SELECT video_id, kind, model, input_tokens, output_tokens, cost_usd, created_at FROM ai_usage WHERE video_id = ? ORDER BY id DESC LIMIT 50",
  )
    .bind(v.id)
    .all<UsageRow>();
  const labelCounts = await env.DB.prepare("SELECT status, COUNT(*) AS n FROM labels WHERE video_id = ? GROUP BY status")
    .bind(v.id)
    .all<{ status: string; n: number }>();
  const strokeCount = await env.DB.prepare("SELECT COUNT(*) AS n FROM stroke_labels WHERE video_id = ?").bind(v.id).first<{ n: number }>();
  return {
    strokes: {
      status: v.strokes_status,
      stage: v.strokes_stage,
      error: v.strokes_error,
      updatedAt: v.strokes_updated_at,
      labels: strokeCount?.n ?? 0,
      labelsChangedAt: v.stroke_labels_changed_at,
      labelsSyncedAt: v.stroke_labels_synced_at,
    },
    annot: {
      status: v.annot_status,
      stage: v.annot_stage,
      error: v.annot_error,
      options: JSON.parse(v.annot_options || "{}"),
      updatedAt: v.annot_updated_at,
      labels: Object.fromEntries(labelCounts.results.map((c) => [c.status, c.n])),
      labelsChangedAt: v.labels_changed_at,
      labelsSyncedAt: v.labels_synced_at,
    },
    ai: usage.results.reverse().map(usageJson),
    job: {
      status: v.job_status,
      stage: v.job_stage,
      error: v.job_error,
      node: v.job_node,
      updatedAt: v.job_updated_at,
      startedAt: v.job_started_at,
      progress: v.job_progress ? JSON.parse(v.job_progress) : null,
    },
    options: JSON.parse(v.options),
    originalDeleted: v.original_deleted_at !== null,
    results: results.results.map((r) => ({ name: r.name, size: r.size, contentType: r.content_type })),
    id: v.id,
    title: v.title,
    filename: v.filename,
    size: v.size,
    contentType: v.content_type,
    partSize: v.part_size,
    partCount: partCount(v),
    status: v.status,
    meta: JSON.parse(v.meta),
    createdAt: v.created_at,
    completedAt: v.completed_at,
    uploadedParts: parts.results.map((p) => p.part_number),
    uploadedBytes: parts.results.reduce((s, p) => s + p.size, 0),
  };
}

async function createVideo(req: Request, env: Env): Promise<Response> {
  const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
  const filename = typeof body?.filename === "string" ? body.filename.slice(0, 200) : "";
  const size = typeof body?.size === "number" ? body.size : NaN;
  const contentType =
    typeof body?.contentType === "string" && body.contentType ? body.contentType.slice(0, 100) : "application/octet-stream";
  if (!filename) throw new HttpError(400, "缺少文件名");
  if (!Number.isInteger(size) || size <= 0 || size > MAX_SIZE) throw new HttpError(400, "文件大小不对");
  const meta = JSON.stringify(body?.meta && typeof body.meta === "object" ? body.meta : {});
  if (meta.length > 4000) throw new HttpError(400, "meta 太长");
  const options = JSON.stringify(cleanOptions(body?.options));
  const title = typeof body?.title === "string" ? body.title.trim().slice(0, 80) : "";

  const id = crypto.randomUUID();
  const ext = (filename.match(/\.([A-Za-z0-9]{1,5})$/)?.[1] ?? "bin").toLowerCase();
  const key = `originals/${USER_ID}/${id}.${ext}`;
  const upload = await env.BUCKET.createMultipartUpload(key, { httpMetadata: { contentType } });
  await env.DB.prepare(
    `INSERT INTO videos (id, user_id, filename, size, content_type, part_size, r2_key, upload_id, status, meta, options, title, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'uploading', ?, ?, ?, ?)`,
  )
    .bind(id, USER_ID, filename, size, contentType, partSizeFor(size), key, upload.uploadId, meta, options, title, new Date().toISOString())
    .run();
  return json(await describe(env, await getVideo(env, id)), 201);
}

async function uploadPart(req: Request, env: Env, id: string, n: number): Promise<Response> {
  const v = await getVideo(env, id);
  if (v.status !== "uploading") throw new HttpError(409, "这个录像已经传完");
  if (!Number.isInteger(n) || n < 1 || n > partCount(v)) throw new HttpError(400, "分片编号不对");
  const want = expectedPartSize(v, n);
  const got = Number(req.headers.get("content-length"));
  if (got !== want) throw new HttpError(400, `第 ${n} 片应该是 ${want} 字节，收到 ${got}`);
  if (!req.body) throw new HttpError(400, "没有内容");

  const upload = env.BUCKET.resumeMultipartUpload(v.r2_key, v.upload_id);
  const part = await upload.uploadPart(n, req.body);
  await env.DB.prepare(
    `INSERT INTO video_parts (video_id, part_number, etag, size) VALUES (?, ?, ?, ?)
     ON CONFLICT (video_id, part_number) DO UPDATE SET etag = excluded.etag, size = excluded.size`,
  )
    .bind(id, n, part.etag, want)
    .run();
  return json({ partNumber: n, size: want });
}

async function completeVideo(env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  if (v.status === "uploaded") return json(await describe(env, v));
  const parts = await env.DB.prepare(
    "SELECT part_number, etag FROM video_parts WHERE video_id = ? ORDER BY part_number",
  )
    .bind(id)
    .all<{ part_number: number; etag: string }>();
  const have = new Set(parts.results.map((p) => p.part_number));
  const missing: number[] = [];
  for (let n = 1; n <= partCount(v); n++) if (!have.has(n)) missing.push(n);
  if (missing.length) return json({ error: "还有分片没传", missing }, 409);

  const upload = env.BUCKET.resumeMultipartUpload(v.r2_key, v.upload_id);
  const object = await upload.complete(parts.results.map((p) => ({ partNumber: p.part_number, etag: p.etag })));
  if (object.size !== v.size) {
    await env.BUCKET.delete(v.r2_key);
    await env.DB.prepare("DELETE FROM videos WHERE id = ?").bind(id).run();
    throw new HttpError(500, `合并后大小 ${object.size} 和原文件 ${v.size} 不一致，已删除，请重新上传`);
  }
  const now = new Date().toISOString();
  await env.DB.prepare(
    "UPDATE videos SET status = 'uploaded', completed_at = ?, job_status = 'queued', job_stage = '', job_updated_at = ? WHERE id = ?",
  )
    .bind(now, now, id)
    .run();
  return json(await describe(env, await getVideo(env, id)));
}

async function renameVideo(req: Request, env: Env, id: string): Promise<Response> {
  await getVideo(env, id);
  const body = (await req.json().catch(() => null)) as { title?: unknown } | null;
  if (typeof body?.title !== "string") throw new HttpError(400, "缺少标题");
  await env.DB.prepare("UPDATE videos SET title = ? WHERE id = ?").bind(body.title.trim().slice(0, 80), id).run();
  return json(await describe(env, await getVideo(env, id)));
}

async function deleteVideo(env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  if (v.status === "uploading") {
    await env.BUCKET.resumeMultipartUpload(v.r2_key, v.upload_id)
      .abort()
      .catch(() => {});
  } else {
    await env.BUCKET.delete(v.r2_key);
  }
  await deleteResults(env, id);
  await deleteAnnotFiles(env, id);
  await deletePrefix(env, strokesKey(id, ""));
  await env.DB.batch([
    env.DB.prepare("DELETE FROM tasks WHERE video_id = ?").bind(id),
    env.DB.prepare("DELETE FROM stroke_labels WHERE video_id = ?").bind(id),
    env.DB.prepare("DELETE FROM labels WHERE video_id = ?").bind(id),
    env.DB.prepare("DELETE FROM video_parts WHERE video_id = ?").bind(id),
    env.DB.prepare("DELETE FROM videos WHERE id = ?").bind(id),
  ]);
  return json({ deleted: id });
}

// 下载原片（支持 Range）。Mac 处理节点取原片、以及测试时核对内容用。
async function downloadVideo(req: Request, env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  if (v.status !== "uploaded") throw new HttpError(409, "还没传完");
  if (v.original_deleted_at) throw new HttpError(410, "原片已经被 Mac 取走，云端不再保留");
  return serveObject(req, env, v.r2_key);
}

async function serveObject(req: Request, env: Env, key: string): Promise<Response> {
  const object = await env.BUCKET.get(key, { range: req.headers });
  if (!object) throw new HttpError(404, "文件不在了");
  const headers = new Headers({ "accept-ranges": "bytes", etag: object.httpEtag, "cache-control": "private, no-store" });
  object.writeHttpMetadata(headers);
  const range = object.range as { offset?: number; length?: number } | undefined;
  if (req.headers.has("range") && range && range.offset !== undefined) {
    const length = range.length ?? object.size - range.offset;
    headers.set("content-range", `bytes ${range.offset}-${range.offset + length - 1}/${object.size}`);
    headers.set("content-length", String(length));
    return new Response(object.body, { status: 206, headers });
  }
  headers.set("content-length", String(object.size));
  return new Response(object.body, { headers });
}

// ---------- 处理任务 ----------

function cleanOptions(raw: unknown): Record<string, unknown> {
  const o = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const out: Record<string, unknown> = { mode: o.mode === "doubles" ? "doubles" : "singles" };
  if (typeof o.names === "string" && o.names.trim()) out.names = o.names.trim().slice(0, 200);
  return out;
}

// 节点每次来问任务或汇报进度都记一笔，网页据此显示“Mac 在线 / 离线”。
async function nodeSeen(env: Env, name: string, now: Date): Promise<void> {
  await env.DB.prepare(
    "INSERT INTO nodes (user_id, name, seen_at) VALUES (?, ?, ?) ON CONFLICT (user_id, name) DO UPDATE SET seen_at = excluded.seen_at",
  )
    .bind(USER_ID, name, now.toISOString())
    .run();
}

// 处理节点领取一个任务：排队中的，或者领取后节点失联（租约过期）的。一条语句完成，不会被两个节点同时领到。
async function claimJob(req: Request, env: Env): Promise<Response> {
  const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
  const node = typeof body?.node === "string" ? body.node.slice(0, 80) : "node";
  // 新的节点会说自己能做哪些事；旧节点不说，只给它分析任务
  const tasks = Array.isArray(body?.tasks) ? (body.tasks as unknown[]) : ["process"];
  const now = new Date();
  await nodeSeen(env, node, now);
  if (tasks.includes("process")) {
    const row = await env.DB.prepare(
      `UPDATE videos SET job_status = 'processing', job_stage = '已领取', job_error = '', job_progress = '', job_node = ?,
         job_lease_until = ?, job_updated_at = ?, job_started_at = ?
       WHERE id = (
         SELECT id FROM videos WHERE user_id = ? AND status = 'uploaded'
           AND (job_status = 'queued' OR (job_status = 'processing' AND job_lease_until < ?))
         ORDER BY completed_at LIMIT 1)
       RETURNING *`,
    )
      .bind(node, new Date(now.getTime() + LEASE_MS).toISOString(), now.toISOString(), now.toISOString(), USER_ID, now.toISOString())
      .first<VideoRow>();
    if (row) return json({ video: await describe(env, row), task: "process" });
  }
  // 没有录像要分析时：先切标注用的图，再把网页上的标注同步回 Mac
  if (tasks.includes("annotate")) {
    const a = await env.DB.prepare(
      `UPDATE videos SET annot_status = 'preparing', annot_stage = '已领取', annot_error = '', annot_lease_until = ?, annot_updated_at = ?
       WHERE id = (
         SELECT id FROM videos WHERE user_id = ? AND status = 'uploaded'
           AND (annot_status = 'queued' OR (annot_status = 'preparing' AND annot_lease_until < ?))
         ORDER BY annot_updated_at LIMIT 1)
       RETURNING *`,
    )
      .bind(new Date(now.getTime() + LEASE_MS).toISOString(), now.toISOString(), USER_ID, now.toISOString())
      .first<VideoRow>();
    if (a) return json({ video: await describe(env, a), task: "annotate" });
  }
  if (tasks.includes("strokes")) {
    const a = await env.DB.prepare(
      `UPDATE videos SET strokes_status = 'preparing', strokes_stage = '已领取', strokes_error = '', strokes_lease_until = ?, strokes_updated_at = ?
       WHERE id = (
         SELECT id FROM videos WHERE user_id = ? AND status = 'uploaded'
           AND (strokes_status = 'queued' OR (strokes_status = 'preparing' AND strokes_lease_until < ?))
         ORDER BY strokes_updated_at LIMIT 1)
       RETURNING *`,
    )
      .bind(new Date(now.getTime() + LEASE_MS).toISOString(), now.toISOString(), USER_ID, now.toISOString())
      .first<VideoRow>();
    if (a) return json({ video: await describe(env, a), task: "strokes" });
  }
  if (tasks.includes("stroke-labels")) {
    const l = await env.DB.prepare(
      `SELECT * FROM videos WHERE user_id = ? AND stroke_labels_changed_at IS NOT NULL
         AND (stroke_labels_synced_at IS NULL OR stroke_labels_synced_at < stroke_labels_changed_at) AND stroke_labels_changed_at < ?
       ORDER BY stroke_labels_changed_at LIMIT 1`,
    )
      .bind(USER_ID, new Date(now.getTime() - labelSyncDelay(env)).toISOString())
      .first<VideoRow>();
    if (l) return json({ video: await describe(env, l), task: "stroke-labels" });
  }
  if (tasks.includes("labels")) {
    // 停手一分钟后再同步，标注过程中不用每一下都同步
    const l = await env.DB.prepare(
      `SELECT * FROM videos WHERE user_id = ? AND labels_changed_at IS NOT NULL
         AND (labels_synced_at IS NULL OR labels_synced_at < labels_changed_at) AND labels_changed_at < ?
       ORDER BY labels_changed_at LIMIT 1`,
    )
      .bind(USER_ID, new Date(now.getTime() - labelSyncDelay(env)).toISOString())
      .first<VideoRow>();
    if (l) return json({ video: await describe(env, l), task: "labels" });
  }
  return json({ video: null });
}

// 节点汇报进度（同时续租）或结束任务。
async function updateJob(req: Request, env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
  const status = body?.status;
  if (status !== "processing" && status !== "done" && status !== "failed") throw new HttpError(400, "status 不对");
  if (v.job_status !== "processing") throw new HttpError(409, "这个任务现在不在处理中");
  const stage = typeof body?.stage === "string" ? body.stage.slice(0, 200) : v.job_stage;
  const error = typeof body?.error === "string" ? body.error.slice(0, 4000) : "";
  // 步骤进度原样保存（节点写、网页读），只限制大小
  let progress = v.job_progress;
  if (body?.progress && typeof body.progress === "object") {
    const text = JSON.stringify(body.progress);
    if (text.length <= 16000) progress = text;
  }
  const now = new Date();
  await env.DB.prepare(
    "UPDATE videos SET job_status = ?, job_stage = ?, job_error = ?, job_progress = ?, job_lease_until = ?, job_updated_at = ? WHERE id = ?",
  )
    .bind(status, stage, error, progress, new Date(now.getTime() + LEASE_MS).toISOString(), now.toISOString(), id)
    .run();
  if (v.job_node) await nodeSeen(env, v.job_node, now);
  if (status === "done") await enforceRetention(env);
  return json(await describe(env, await getVideo(env, id)));
}

// 重新处理（失败后，或者改了单打/双打、球员名字之后）。节点本地留着原片，所以云端原片删了也能重跑。
async function retryJob(req: Request, env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  if (v.status !== "uploaded") throw new HttpError(409, "还没传完");
  const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
  const options = body?.options ? JSON.stringify(cleanOptions(body.options)) : v.options;
  await deleteResults(env, id); // 旧结果不留：换了单打/双打后文件不一样，混在一起会看错
  await env.DB.prepare(
    "UPDATE videos SET job_status = 'queued', job_stage = '', job_error = '', job_progress = '', job_lease_until = NULL, options = ?, job_updated_at = ? WHERE id = ?",
  )
    .bind(options, new Date().toISOString(), id)
    .run();
  return json(await describe(env, await getVideo(env, id)));
}

// 节点下载完原片后调用：云端立刻删掉原片。
async function releaseOriginal(env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  if (v.status !== "uploaded") throw new HttpError(409, "还没传完");
  if (!v.original_deleted_at) {
    await env.BUCKET.delete(v.r2_key);
    await env.DB.prepare("UPDATE videos SET original_deleted_at = ? WHERE id = ?").bind(new Date().toISOString(), id).run();
  }
  return json({ released: id });
}

// ---------- 结果文件 ----------

const RESULT_TYPES: Record<string, string> = {
  png: "image/png", jpg: "image/jpeg", json: "application/json", mp4: "video/mp4",
  html: "text/html; charset=utf-8", md: "text/markdown; charset=utf-8", txt: "text/plain; charset=utf-8", csv: "text/csv; charset=utf-8",
};
const resultType = (name: string) => RESULT_TYPES[name.split(".").pop()!.toLowerCase()] ?? "application/octet-stream";
const resultKey = (id: string, name: string) => `results/${USER_ID}/${id}/${name}`;

async function recordResult(env: Env, id: string, name: string, size: number): Promise<Response> {
  await env.DB.prepare(
    `INSERT INTO results (video_id, name, r2_key, size, content_type, created_at) VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT (video_id, name) DO UPDATE SET size = excluded.size, content_type = excluded.content_type, created_at = excluded.created_at`,
  )
    .bind(id, name, resultKey(id, name), size, resultType(name), new Date().toISOString())
    .run();
  return json({ name, size });
}

async function putResult(req: Request, env: Env, id: string, name: string): Promise<Response> {
  await getVideo(env, id);
  const size = Number(req.headers.get("content-length"));
  if (!req.body || !Number.isInteger(size) || size <= 0) throw new HttpError(400, "没有内容");
  const object = await env.BUCKET.put(resultKey(id, name), req.body, { httpMetadata: { contentType: resultType(name) } });
  return recordResult(env, id, name, object.size);
}

// 大的结果文件（视频）分片传。uploadId 和各片的 etag 由节点自己记着，服务器不存状态。
async function resultMultipart(req: Request, env: Env, id: string, name: string, url: URL): Promise<Response> {
  await getVideo(env, id);
  return multipart(req, env, resultKey(id, name), resultType(name), url, (size) => recordResult(env, id, name, size));
}

async function multipart(req: Request, env: Env, key: string, contentType: string, url: URL, done: (size: number) => Promise<Response>): Promise<Response> {
  const uploadId = url.searchParams.get("uploadId");
  if (req.method === "POST" && !uploadId) {
    const upload = await env.BUCKET.createMultipartUpload(key, { httpMetadata: { contentType } });
    return json({ uploadId: upload.uploadId }, 201);
  }
  if (!uploadId) throw new HttpError(400, "缺少 uploadId");
  const upload = env.BUCKET.resumeMultipartUpload(key, uploadId);
  if (req.method === "PUT") {
    const n = Number(url.searchParams.get("part"));
    if (!Number.isInteger(n) || n < 1 || n > MAX_PARTS || !req.body) throw new HttpError(400, "分片不对");
    const part = await upload.uploadPart(n, req.body);
    return json({ partNumber: n, etag: part.etag });
  }
  const body = (await req.json().catch(() => null)) as { parts?: { partNumber: number; etag: string }[] } | null;
  if (!body?.parts?.length) throw new HttpError(400, "缺少分片列表");
  const object = await upload.complete(body.parts);
  return done(object.size);
}

async function getResult(req: Request, env: Env, id: string, name: string): Promise<Response> {
  const v = await getVideo(env, id);
  const res = await serveObject(req, env, resultKey(v.id, name));
  // 报告是程序生成的 HTML：不让它跑脚本，也不让它被当成本站页面的一部分
  res.headers.set("content-security-policy", "sandbox; default-src 'none'; img-src data: 'self'; style-src 'unsafe-inline'");
  res.headers.set("x-content-type-options", "nosniff");
  // ?download=1：让浏览器存成文件（手机上「下载」），文件名带上录像标题
  if (new URL(req.url).searchParams.get("download")) {
    const stem = (v.title || v.filename.replace(/\.[^.]*$/, "")).replace(/[\\/:*?"<>|\r\n]+/g, " ").trim().slice(0, 80) || "录像";
    const file = `${stem}_${name}`;
    const ascii = file.replace(/[^\x20-\x7e]/g, "_").replace(/["\\]/g, "_");
    res.headers.set("content-disposition", `attachment; filename="${ascii}"; filename*=UTF-8''${encodeURIComponent(file)}`);
    res.headers.delete("content-security-policy"); // 存成文件，不会当网页打开；sandbox 会让浏览器丢掉文件名
  }
  return res;
}

// ---------- AI 报告 ----------
// 节点上传 review.request.json（badminton review 写的：system、user、追问模板），里面只有统计数字和文字。
// 这里拿它去问模型，回答存成 review.md；追问的问答追加到 review.qa.md。key 只在 Worker 里。
// 每次调用的 token 用量记在 ai_usage 表里，按设置里的单价估算价格。

interface ReviewRequest {
  model?: string;
  max_tokens: number;
  system: string;
  user: string;
  ask: string; // 含 {question}
  title?: string;
}

// 可以选的模型：id 是部署名。Claude 走 Anthropic 的 Messages 接口，其余走 OpenAI 的 Chat Completions 接口。
interface Model {
  id: string;
  label: string;
  format: "anthropic" | "openai";
  in?: number; // 默认单价，美元 / 100 万 token；设置里填了就用设置里的
  out?: number;
}
// 默认单价是 2026 年 11 月从第三方价格汇总网站查到的公开价，不是 Azure 账单上的实际价格；查不到的留空。
const DEFAULT_MODELS: Model[] = [
  { id: "claude-sonnet-5-5", label: "Claude Sonnet 5.5", format: "anthropic", in: 2, out: 10 },
  { id: "claude-opus-5-5", label: "Claude Opus 5.5", format: "anthropic", in: 4, out: 20 },
  { id: "claude-haiku-5-5-2", label: "Claude Haiku 5.5", format: "anthropic", in: 0.1, out: 0.5 },
  { id: "claude-fable-5-1", label: "Claude Fable 5.1", format: "anthropic", in: 10, out: 50 },
  { id: "gpt-6.1-sol", label: "GPT-6.1 Sol", format: "openai", in: 2, out: 10 },
  { id: "gpt-6.1-terra", label: "GPT-6.1 Terra", format: "openai" },
  { id: "gpt-6-luna", label: "GPT-6 Luna", format: "openai", in: 0.1, out: 0.5 },
  { id: "grok-4.6", label: "Grok 4.6", format: "openai" },
];

function models(env: Env): Model[] {
  if (!env.AI_MODELS) return DEFAULT_MODELS;
  try {
    const list = JSON.parse(env.AI_MODELS) as Model[];
    if (Array.isArray(list) && list.every((m) => m && typeof m.id === "string")) {
      return list.map((m) => ({ id: m.id, label: m.label || m.id, format: m.format === "openai" ? "openai" : "anthropic", in: price(m.in), out: price(m.out) }));
    }
  } catch {}
  return DEFAULT_MODELS;
}

const aiKey = (env: Env) => env.AI_API_KEY || env.ANTHROPIC_API_KEY || "";

function modelUrl(env: Env, m: Model): string {
  const base = (m.format === "openai" ? env.AI_ENDPOINT_OPENAI || "https://api.openai.com/v1" : env.AI_ENDPOINT_ANTHROPIC || "https://api.anthropic.com").replace(/\/+$/, "");
  return base + (m.format === "openai" ? "/chat/completions" : "/v1/messages");
}

type Prices = Record<string, { in?: number; out?: number }>; // 美元 / 100 万 token
interface AiSettings {
  defaultModel: string;
  prices: Prices;
}

async function aiSettings(env: Env): Promise<AiSettings> {
  const row = await env.DB.prepare("SELECT value FROM settings WHERE user_id = ? AND key = 'ai'").bind(USER_ID).first<{ value: string }>();
  let saved: Partial<AiSettings> = {};
  try {
    saved = row ? (JSON.parse(row.value) as Partial<AiSettings>) : {};
  } catch {}
  const list = models(env);
  const defaultModel = list.some((m) => m.id === saved.defaultModel) ? saved.defaultModel! : (list[0]?.id ?? "");
  return { defaultModel, prices: saved.prices && typeof saved.prices === "object" ? saved.prices : {} };
}

function modelPrice(m: Model, saved: Prices): { in: number | null; out: number | null } {
  return { in: saved[m.id]?.in ?? m.in ?? null, out: saved[m.id]?.out ?? m.out ?? null };
}

function price(v: unknown): number | undefined {
  return typeof v === "number" && isFinite(v) && v >= 0 && v < 100000 ? v : undefined;
}

async function saveAiSettings(req: Request, env: Env): Promise<Response> {
  const body = (await req.json().catch(() => null)) as Partial<AiSettings> | null;
  const now = await aiSettings(env);
  const list = models(env);
  if (typeof body?.defaultModel === "string") {
    if (!list.some((m) => m.id === body.defaultModel)) throw new HttpError(400, "没有这个模型");
    now.defaultModel = body.defaultModel;
  }
  if (body?.prices && typeof body.prices === "object") {
    for (const m of list) {
      const p = (body.prices as Prices)[m.id];
      if (p === undefined) continue;
      const cleaned = { in: price(p?.in), out: price(p?.out) };
      if (cleaned.in === undefined && cleaned.out === undefined) delete now.prices[m.id];
      else now.prices[m.id] = cleaned;
    }
  }
  await env.DB.prepare("INSERT INTO settings (user_id, key, value) VALUES (?, 'ai', ?) ON CONFLICT (user_id, key) DO UPDATE SET value = excluded.value")
    .bind(USER_ID, JSON.stringify(now))
    .run();
  return aiInfo(env);
}

type UsageRow = { video_id: string; kind: string; model: string; input_tokens: number; output_tokens: number; cost_usd: number | null; created_at: string };
const usageJson = (u: UsageRow) => ({
  kind: u.kind, model: u.model, inputTokens: u.input_tokens, outputTokens: u.output_tokens, costUsd: u.cost_usd, at: u.created_at,
});

// 设置页：能选哪些模型、默认模型、单价、累计用量
async function aiInfo(env: Env): Promise<Response> {
  const settings = await aiSettings(env);
  const totals = await env.DB.prepare(
    `SELECT model, COUNT(*) AS calls, SUM(input_tokens) AS input, SUM(output_tokens) AS output, SUM(cost_usd) AS cost,
            SUM(cost_usd IS NULL) AS unpriced, MIN(created_at) AS since
     FROM ai_usage WHERE user_id = ? GROUP BY model ORDER BY calls DESC`,
  )
    .bind(USER_ID)
    .all<{ model: string; calls: number; input: number; output: number; cost: number | null; unpriced: number; since: string }>();
  return json({
    available: Boolean(aiKey(env)),
    models: models(env).map((m) => ({ id: m.id, label: m.label, priceIn: modelPrice(m, settings.prices).in, priceOut: modelPrice(m, settings.prices).out })),
    defaultModel: settings.defaultModel,
    usage: totals.results.map((t) => ({
      model: t.model, calls: t.calls, inputTokens: t.input, outputTokens: t.output, costUsd: t.cost, unpricedCalls: t.unpriced, since: t.since,
    })),
  });
}

async function resultText(env: Env, id: string, name: string): Promise<string | null> {
  const object = await env.BUCKET.get(resultKey(id, name));
  return object ? object.text() : null;
}

async function saveResultText(env: Env, id: string, name: string, text: string): Promise<void> {
  const object = await env.BUCKET.put(resultKey(id, name), text, { httpMetadata: { contentType: resultType(name) } });
  await recordResult(env, id, name, object.size);
}

async function reviewRequest(env: Env, id: string): Promise<ReviewRequest> {
  if (!aiKey(env)) throw new HttpError(503, "网站还没有配置模型的 API key（AI_API_KEY），出不了 AI 报告");
  const text = await resultText(env, id, "review.request.json");
  let r: ReviewRequest | null = null;
  try {
    r = text ? (JSON.parse(text) as ReviewRequest) : null;
  } catch {}
  if (!r || typeof r.system !== "string" || typeof r.user !== "string")
    throw new HttpError(409, "这局还没有 AI 报告要用的统计摘要，请先重新处理");
  return r;
}

async function pickModel(env: Env, wanted: unknown): Promise<Model> {
  const list = models(env);
  const id = typeof wanted === "string" && wanted ? wanted : (await aiSettings(env)).defaultModel;
  const m = list.find((x) => x.id === id);
  if (!m) throw new HttpError(400, "没有这个模型：" + String(id).slice(0, 60));
  return m;
}

type Message = { role: "user" | "assistant"; content: string };
interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

// 流式读取：长回答不会因为连接长时间没有数据被掐断
async function askModel(env: Env, m: Model, r: ReviewRequest, messages: Message[]): Promise<Answer> {
  const key = aiKey(env);
  const maxTokens = Math.min(Math.max(Number(r.max_tokens) || 8000, 256), 16000);
  const openai = m.format === "openai";
  const res = await fetch(modelUrl(env, m), {
    method: "POST",
    headers: openai
      ? { "content-type": "application/json", "api-key": key, authorization: "Bearer " + key }
      : { "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01" },
    body: JSON.stringify(
      openai
        ? {
            model: m.id, max_completion_tokens: maxTokens, stream: true, stream_options: { include_usage: true },
            messages: [{ role: "system", content: r.system }, ...messages],
          }
        : { model: m.id, max_tokens: maxTokens, system: r.system, messages, stream: true },
    ),
  });
  if (!res.ok || !res.body) {
    let detail = "";
    try {
      detail = ((await res.json()) as { error?: { message?: string } }).error?.message ?? "";
    } catch {}
    throw new HttpError(502, `调用 ${m.label} 失败（${res.status}）${detail ? "：" + detail.slice(0, 300) : ""}`);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  let text = "";
  let stop = "";
  let input = 0;
  let output = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    const lines = buf.split("\n");
    buf = lines.pop()!;
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      let ev: any;
      try {
        ev = JSON.parse(line.slice(5));
      } catch {
        continue; // 包括 OpenAI 格式结尾的 [DONE]
      }
      if (ev.error) throw new HttpError(502, `${m.label} 出错：` + String(ev.error?.message ?? "").slice(0, 300));
      if (openai) {
        const choice = ev.choices?.[0];
        if (typeof choice?.delta?.content === "string") text += choice.delta.content;
        if (choice?.finish_reason) stop = choice.finish_reason;
        if (ev.usage) {
          input = Number(ev.usage.prompt_tokens) || 0;
          // 有的模型把思考用的 token 只算在 total 里，不算在 completion 里；计费是按 total 的
          output = Math.max(Number(ev.usage.completion_tokens) || 0, (Number(ev.usage.total_tokens) || 0) - input);
        }
      } else if (ev.type === "message_start") input = Number(ev.message?.usage?.input_tokens) || 0;
      else if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") text += ev.delta.text;
      else if (ev.type === "message_delta") {
        if (ev.delta?.stop_reason) stop = ev.delta.stop_reason;
        if (ev.usage?.output_tokens != null) output = Number(ev.usage.output_tokens) || 0;
        if (ev.usage?.input_tokens != null) input = Number(ev.usage.input_tokens) || input;
      }
    }
  }
  text = text.trim();
  if (stop === "refusal" || stop === "content_filter") throw new HttpError(502, "模型拒绝回答这个请求");
  if (!text) throw new HttpError(502, "模型没有返回文字");
  if (stop === "max_tokens" || stop === "length") text += "\n\n（回答到长度上限被截断）";
  return { text, inputTokens: input, outputTokens: output };
}

async function recordUsage(env: Env, id: string, kind: "report" | "ask", m: Model, a: Answer): Promise<UsageRow> {
  const p = modelPrice(m, (await aiSettings(env)).prices);
  const cost = p.in !== null && p.out !== null ? (a.inputTokens * p.in + a.outputTokens * p.out) / 1e6 : null;
  const row: UsageRow = {
    video_id: id, kind, model: m.id, input_tokens: a.inputTokens, output_tokens: a.outputTokens, cost_usd: cost, created_at: new Date().toISOString(),
  };
  await env.DB.prepare("INSERT INTO ai_usage (user_id, video_id, kind, model, input_tokens, output_tokens, cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
    .bind(USER_ID, id, kind, m.id, a.inputTokens, a.outputTokens, cost, row.created_at)
    .run();
  return row;
}

async function makeReview(req: Request, env: Env, id: string): Promise<Response> {
  await getVideo(env, id);
  const body = (await req.json().catch(() => null)) as { model?: unknown } | null;
  const r = await reviewRequest(env, id);
  const m = await pickModel(env, body?.model);
  const answer = await askModel(env, m, r, [{ role: "user", content: r.user }]);
  await recordUsage(env, id, "report", m, answer);
  const note = `> 由 ${m.label} 根据程序算出的统计数字生成，它没有看视频。数字以上面的表和卡片为准。`;
  await saveResultText(env, id, "review.md", `${answer.text}\n\n${note}\n`);
  await deleteResult(env, id, "review.qa.md"); // 追问是接着旧报告问的，报告换了就不再对得上
  return json(await describe(env, await getVideo(env, id)));
}

async function askReview(req: Request, env: Env, id: string): Promise<Response> {
  await getVideo(env, id);
  const body = (await req.json().catch(() => null)) as { question?: unknown; model?: unknown } | null;
  const question = typeof body?.question === "string" ? body.question.trim().slice(0, 1000) : "";
  if (!question) throw new HttpError(400, "请写下要问的问题");
  const r = await reviewRequest(env, id);
  const m = await pickModel(env, body?.model);
  const report = await resultText(env, id, "review.md");
  const earlier = (await resultText(env, id, "review.qa.md")) ?? "";
  const messages: Message[] = [{ role: "user", content: r.user }];
  if (report) messages.push({ role: "assistant", content: report });
  else messages[0].content += "\n\n（先不用写点评，等我提问。）";
  if (!report) messages.push({ role: "assistant", content: "好的，请问。" });
  const ask = (q: string) => (r.ask || "追问：{question}").replace("{question}", q);
  // 之前的问答也带上，这样可以接着上一个问题问
  for (const qa of earlier.matchAll(/^## 问：(.*)\n\n([\s\S]*?)(?=\n\n## 问：|\s*$)/gm)) {
    messages.push({ role: "user", content: ask(qa[1]) }, { role: "assistant", content: qa[2].trim() });
  }
  messages.push({ role: "user", content: ask(question) });
  const answer = await askModel(env, m, r, messages);
  await recordUsage(env, id, "ask", m, answer);
  const oneLine = question.replace(/\s+/g, " ");
  await saveResultText(env, id, "review.qa.md", `${earlier}## 问：${oneLine}\n\n${answer.text.replace(/^## /gm, "### ")}\n\n`);
  return json({ answer: answer.text, video: await describe(env, await getVideo(env, id)) });
}

async function deleteResult(env: Env, id: string, name: string): Promise<void> {
  await env.BUCKET.delete(resultKey(id, name));
  await env.DB.prepare("DELETE FROM results WHERE video_id = ? AND name = ?").bind(id, name).run();
}

async function deleteResults(env: Env, id: string): Promise<void> {
  const rows = await env.DB.prepare("SELECT r2_key FROM results WHERE video_id = ?").bind(id).all<{ r2_key: string }>();
  const keys = rows.results.map((r) => r.r2_key);
  for (let i = 0; i < keys.length; i += 500) await env.BUCKET.delete(keys.slice(i, i + 500));
  await env.DB.prepare("DELETE FROM results WHERE video_id = ?").bind(id).run();
}

// 云端只留最近约 10 GB 的结果：从最新的一局往回数，超出的整局清掉（Mac 上还有）。最新一局总是保留。
async function enforceRetention(env: Env): Promise<void> {
  const rows = await env.DB.prepare(
    `SELECT v.id AS id, COALESCE(SUM(r.size), 0) AS bytes FROM videos v JOIN results r ON r.video_id = v.id
     WHERE v.user_id = ? GROUP BY v.id ORDER BY v.completed_at DESC`,
  )
    .bind(USER_ID)
    .all<{ id: string; bytes: number }>();
  let total = 0;
  for (const [i, row] of rows.results.entries()) {
    total += row.bytes;
    if (i === 0 || total <= KEEP_RESULT_BYTES) continue;
    await deleteResults(env, row.id);
    await env.DB.prepare("UPDATE videos SET job_status = 'archived', job_stage = '云端结果已清理，Mac 上还有' WHERE id = ?")
      .bind(row.id)
      .run();
  }
}

// ---------- 手机上标注球的位置 ----------
// 网页点「标注球位置」→ Mac 节点挑模型最没把握的几个回合、切成图片（每个回合一个文件，state.json 记每帧的起止位置
// 和模型的预测）传到 annot/ 下 → 网页按 Range 一次取几帧 → 每标一帧存进 labels 表 → 节点同步回 .labels.csv。

export const annotKey = (id: string, name: string) => `annot/${USER_ID}/${id}/${name}`;
const LABEL_STATUSES = new Set(["human", "bulk", "skip"]); // 网页只写人看过的；machine 只在 Mac 上
const MAX_LABEL_ITEMS = 5000;

async function requestAnnot(req: Request, env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  if (v.status !== "uploaded") throw new HttpError(409, "还没传完");
  if (v.annot_status === "preparing") throw new HttpError(409, "Mac 正在准备，稍等");
  const body = (await req.json().catch(() => null)) as { fresh?: unknown } | null;
  const options = JSON.stringify({ fresh: body?.fresh === true });
  await env.DB.prepare(
    "UPDATE videos SET annot_status = 'queued', annot_stage = '', annot_error = '', annot_options = ?, annot_lease_until = NULL, annot_updated_at = ? WHERE id = ?",
  )
    .bind(options, new Date().toISOString(), id)
    .run();
  return json(await describe(env, await getVideo(env, id)));
}

// 删掉切好的图（标注本身留着）
async function deleteAnnot(env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  if (v.annot_status === "preparing") throw new HttpError(409, "Mac 正在准备，稍等");
  await deleteAnnotFiles(env, id);
  await env.DB.prepare("UPDATE videos SET annot_status = 'none', annot_stage = '', annot_error = '', annot_updated_at = ? WHERE id = ?")
    .bind(new Date().toISOString(), id)
    .run();
  return json(await describe(env, await getVideo(env, id)));
}

async function deleteAnnotFiles(env: Env, id: string): Promise<void> {
  await deletePrefix(env, annotKey(id, ""));
}

async function deletePrefix(env: Env, prefix: string): Promise<void> {
  for (let cursor: string | undefined; ; ) {
    const list = await env.BUCKET.list({ prefix, cursor });
    if (list.objects.length) await env.BUCKET.delete(list.objects.map((o) => o.key));
    if (!list.truncated) break;
    cursor = list.cursor;
  }
}

// 节点汇报准备进度（同时续租），或以 ready / failed 结束
async function updateAnnotJob(req: Request, env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
  const status = body?.status;
  if (status !== "preparing" && status !== "ready" && status !== "failed") throw new HttpError(400, "status 不对");
  if (v.annot_status !== "preparing") throw new HttpError(409, "这个标注任务现在不在准备中");
  const stage = typeof body?.stage === "string" ? body.stage.slice(0, 200) : v.annot_stage;
  const error = typeof body?.error === "string" ? body.error.slice(0, 4000) : "";
  const now = new Date();
  await env.DB.prepare("UPDATE videos SET annot_status = ?, annot_stage = ?, annot_error = ?, annot_lease_until = ?, annot_updated_at = ? WHERE id = ?")
    .bind(status, stage, error, new Date(now.getTime() + LEASE_MS).toISOString(), now.toISOString(), id)
    .run();
  return json(await describe(env, await getVideo(env, id)));
}

const annotType = (name: string) => (name.endsWith(".json") ? "application/json" : "application/octet-stream");

async function putAnnotFile(req: Request, env: Env, id: string, name: string): Promise<Response> {
  await getVideo(env, id);
  const size = Number(req.headers.get("content-length"));
  if (!req.body || !Number.isInteger(size) || size <= 0) throw new HttpError(400, "没有内容");
  const object = await env.BUCKET.put(annotKey(id, name), req.body, { httpMetadata: { contentType: annotType(name) } });
  return json({ name, size: object.size });
}

async function annotMultipart(req: Request, env: Env, id: string, name: string, url: URL): Promise<Response> {
  await getVideo(env, id);
  return multipart(req, env, annotKey(id, name), annotType(name), url, async (size) => json({ name, size }));
}

async function getAnnotFile(req: Request, env: Env, id: string, name: string): Promise<Response> {
  const v = await getVideo(env, id);
  return serveObject(req, env, annotKey(v.id, name));
}

const csvName = (v: VideoRow) => (v.filename.replace(/\.[^.]*$/, "") || "video").replace(/[^A-Za-z0-9._-]/g, "_");
const csvFile = (lines: string[], name: string) =>
  new Response("\ufeff" + lines.join("\r\n") + "\r\n", {
    headers: { "content-type": "text/csv; charset=utf-8", "content-disposition": `attachment; filename="${name}"`, "cache-control": "no-store" },
  });

// ?format=csv：网页上下载，比 Mac 上的 .labels.csv 多两列：谁标的、什么时候标的
async function getLabels(env: Env, id: string, url: URL): Promise<Response> {
  const v = await getVideo(env, id);
  if (url.searchParams.get("format") === "csv") {
    const all = await env.DB.prepare("SELECT frame, visibility, x, y, status, annotator, updated_at FROM labels WHERE video_id = ? ORDER BY frame")
      .bind(id)
      .all<Record<string, unknown>>();
    return csvFile(
      ["frame,visibility,x,y,status,annotator,updated_at", ...all.results.map((r) => [r.frame, r.visibility, r.x, r.y, r.status, r.annotator, r.updated_at].map(csvCell).join(","))],
      csvName(v) + ".labels.csv",
    );
  }
  const rows = await env.DB.prepare("SELECT frame, visibility, x, y, status FROM labels WHERE video_id = ? ORDER BY frame")
    .bind(id)
    .all<{ frame: number; visibility: number; x: number; y: number; status: string }>();
  return json({
    labels: Object.fromEntries(rows.results.map((r) => [r.frame, [r.visibility, r.x, r.y, r.status]])),
    changedAt: v.labels_changed_at,
  });
}

// items: [{frame, visibility, x, y, status}] 或 [{frame, delete: true}]。先全部检查，有一条不对就一条都不写。
async function putLabels(req: Request, env: Env, id: string, who: Who): Promise<Response> {
  await getVideo(env, id);
  const body = (await req.json().catch(() => null)) as { items?: unknown } | null;
  const items = Array.isArray(body?.items) ? (body.items as Record<string, unknown>[]) : null;
  if (!items || !items.length || items.length > MAX_LABEL_ITEMS) throw new HttpError(400, "标注列表不对");
  const now = new Date().toISOString();
  const annotator = await guardLabelWrite(env, who, id, "shuttle", items.map((it) => Number(it?.frame)).filter(Number.isInteger));
  const stmts: D1PreparedStatement[] = [];
  for (const it of items) {
    const frame = Number(it?.frame);
    if (!Number.isInteger(frame) || frame < 0 || frame > 10_000_000) throw new HttpError(400, "帧号不对");
    if (it.delete === true) {
      stmts.push(env.DB.prepare("DELETE FROM labels WHERE video_id = ? AND frame = ?").bind(id, frame));
      continue;
    }
    const status = String(it.status ?? "human");
    if (!LABEL_STATUSES.has(status)) throw new HttpError(400, "标注状态不对：" + status.slice(0, 20));
    const visible = status !== "skip" && Boolean(it.visibility);
    const x = visible ? Number(it.x) : 0;
    const y = visible ? Number(it.y) : 0;
    if (!isFinite(x) || !isFinite(y) || x < 0 || y < 0 || x > 20000 || y > 20000) throw new HttpError(400, "位置不对");
    stmts.push(
      env.DB.prepare(
        `INSERT INTO labels (video_id, frame, visibility, x, y, status, updated_at, annotator) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (video_id, frame) DO UPDATE SET visibility = excluded.visibility, x = excluded.x, y = excluded.y,
           status = excluded.status, updated_at = excluded.updated_at, annotator = excluded.annotator`,
      ).bind(id, frame, visible ? 1 : 0, Math.round(x * 100) / 100, Math.round(y * 100) / 100, status, now, annotator),
    );
  }
  stmts.push(env.DB.prepare("UPDATE videos SET labels_changed_at = ? WHERE id = ?").bind(now, id));
  for (let i = 0; i < stmts.length; i += 400) await env.DB.batch(stmts.slice(i, i + 400));
  return json({ ok: true, changedAt: now });
}

// 节点同步完：记下同步到了哪一次改动（同步期间又有新改动的话，下次再同步）
async function labelsSynced(req: Request, env: Env, id: string): Promise<Response> {
  await getVideo(env, id);
  const body = (await req.json().catch(() => null)) as { changedAt?: unknown } | null;
  const at = typeof body?.changedAt === "string" ? body.changedAt : new Date(0).toISOString();
  await env.DB.prepare("UPDATE videos SET labels_synced_at = ? WHERE id = ?").bind(at, id).run();
  return json({ ok: true });
}

// ---------- 手机上标注击球类型 ----------
// 网页点「准备击球类型标注」→ Mac 节点跑 badminton strokes，把每一拍前后 1.4 秒剪成一段（一个文件，一段段首尾相接）
// 传到 strokes/ 下 → 网页逐拍看、选主类型和附加标记 → 存进 stroke_labels → 节点写回 .strokes.labels.csv。

export const strokesKey = (id: string, name: string) => `strokes/${USER_ID}/${id}/${name}`;
const STROKE_TYPES = new Set(["放网", "挡网", "杀球", "点杀", "挑球", "防守挑球", "高远球", "平高球", "平抽", "后场抽平", "吊球",
  "过渡吊球", "推球", "扑球", "防守抽球", "勾对角", "发短球", "发长球", "未知", "不是击球"]);
const STROKE_TAGS = ["劈杀", "劈吊", "收吊", "搓球", "直线", "对角"];

async function requestStrokes(env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  if (v.status !== "uploaded") throw new HttpError(409, "还没传完");
  if (v.strokes_status === "preparing") throw new HttpError(409, "Mac 正在准备，稍等");
  await env.DB.prepare(
    "UPDATE videos SET strokes_status = 'queued', strokes_stage = '', strokes_error = '', strokes_lease_until = NULL, strokes_updated_at = ? WHERE id = ?",
  )
    .bind(new Date().toISOString(), id)
    .run();
  return json(await describe(env, await getVideo(env, id)));
}

async function deleteStrokes(env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  if (v.strokes_status === "preparing") throw new HttpError(409, "Mac 正在准备，稍等");
  await deletePrefix(env, strokesKey(id, ""));
  await env.DB.prepare("UPDATE videos SET strokes_status = 'none', strokes_stage = '', strokes_error = '', strokes_updated_at = ? WHERE id = ?")
    .bind(new Date().toISOString(), id)
    .run();
  return json(await describe(env, await getVideo(env, id)));
}

async function updateStrokesJob(req: Request, env: Env, id: string): Promise<Response> {
  const v = await getVideo(env, id);
  const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
  const status = body?.status;
  if (status !== "preparing" && status !== "ready" && status !== "failed") throw new HttpError(400, "status 不对");
  if (v.strokes_status !== "preparing") throw new HttpError(409, "这个任务现在不在准备中");
  const stage = typeof body?.stage === "string" ? body.stage.slice(0, 200) : v.strokes_stage;
  const error = typeof body?.error === "string" ? body.error.slice(0, 4000) : "";
  const now = new Date();
  await env.DB.prepare("UPDATE videos SET strokes_status = ?, strokes_stage = ?, strokes_error = ?, strokes_lease_until = ?, strokes_updated_at = ? WHERE id = ?")
    .bind(status, stage, error, new Date(now.getTime() + LEASE_MS).toISOString(), now.toISOString(), id)
    .run();
  return json(await describe(env, await getVideo(env, id)));
}

const strokesType = (name: string) => (name.endsWith(".json") ? "application/json" : name.endsWith(".mp4") ? "video/mp4" : "application/octet-stream");

async function putStrokesFile(req: Request, env: Env, id: string, name: string): Promise<Response> {
  await getVideo(env, id);
  const size = Number(req.headers.get("content-length"));
  if (!req.body || !Number.isInteger(size) || size <= 0) throw new HttpError(400, "没有内容");
  const object = await env.BUCKET.put(strokesKey(id, name), req.body, { httpMetadata: { contentType: strokesType(name) } });
  return json({ name, size: object.size });
}

type StrokeLabelRow = { frame: number; player_id: string; type: string; tags: string; note: string };
const strokeLabelJson = (r: StrokeLabelRow) => ({
  player_id: r.player_id === "" ? null : /^\d+$/.test(r.player_id) ? Number(r.player_id) : r.player_id,
  type: r.type, tags: r.tags ? r.tags.split(";") : [], note: r.note,
});
const csvCell = (v: unknown) => {
  const t = v === null || v === undefined ? "" : String(v);
  return /[",\n\r]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
};

// ?format=csv：网页上直接下载（前几列和 Mac 上的 .strokes.labels.csv 一样，后面多两列：谁标的、什么时候标的）
async function getStrokeLabels(env: Env, id: string, url: URL): Promise<Response> {
  const v = await getVideo(env, id);
  const rows = await env.DB.prepare("SELECT frame, player_id, type, tags, note, annotator, updated_at FROM stroke_labels WHERE video_id = ? ORDER BY frame")
    .bind(id)
    .all<StrokeLabelRow & { annotator: string; updated_at: string }>();
  if (url.searchParams.get("format") === "csv") {
    return csvFile(
      ["frame,player_id,type,tags,note,annotator,updated_at", ...rows.results.map((r) => [r.frame, r.player_id, r.type, r.tags, r.note, r.annotator, r.updated_at].map(csvCell).join(","))],
      csvName(v) + ".strokes.labels.csv",
    );
  }
  return json({ labels: Object.fromEntries(rows.results.map((r) => [r.frame, strokeLabelJson(r)])), changedAt: v.stroke_labels_changed_at });
}

// items: [{frame, player_id, type, tags: [...], note}] 或 [{frame, delete: true}]。先全部检查。
async function putStrokeLabels(req: Request, env: Env, id: string, who: Who): Promise<Response> {
  await getVideo(env, id);
  const body = (await req.json().catch(() => null)) as { items?: unknown } | null;
  const items = Array.isArray(body?.items) ? (body.items as Record<string, unknown>[]) : null;
  if (!items || !items.length || items.length > MAX_LABEL_ITEMS) throw new HttpError(400, "标注列表不对");
  const now = new Date().toISOString();
  const annotator = await guardLabelWrite(env, who, id, "stroke", items.map((it) => Number(it?.frame)).filter(Number.isInteger));
  const stmts: D1PreparedStatement[] = [];
  for (const it of items) {
    const frame = Number(it?.frame);
    if (!Number.isInteger(frame) || frame < 0 || frame > 10_000_000) throw new HttpError(400, "帧号不对");
    if (it.delete === true) {
      stmts.push(env.DB.prepare("DELETE FROM stroke_labels WHERE video_id = ? AND frame = ?").bind(id, frame));
      continue;
    }
    const type = String(it.type ?? "");
    if (!STROKE_TYPES.has(type)) throw new HttpError(400, "击球类型不对：" + type.slice(0, 20));
    const tags = Array.isArray(it.tags) ? it.tags.map(String) : [];
    if (tags.some((t) => !STROKE_TAGS.includes(t))) throw new HttpError(400, "附加标记不对");
    const ordered = STROKE_TAGS.filter((t) => tags.includes(t)).join(";");
    const pid = it.player_id === null || it.player_id === undefined ? "" : String(it.player_id).slice(0, 20);
    const note = typeof it.note === "string" ? it.note.trim().slice(0, 300) : "";
    stmts.push(
      env.DB.prepare(
        `INSERT INTO stroke_labels (video_id, frame, player_id, type, tags, note, updated_at, annotator) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (video_id, frame) DO UPDATE SET player_id = excluded.player_id, type = excluded.type, tags = excluded.tags,
           note = excluded.note, updated_at = excluded.updated_at, annotator = excluded.annotator`,
      ).bind(id, frame, pid, type, ordered, note, now, annotator),
    );
  }
  stmts.push(env.DB.prepare("UPDATE videos SET stroke_labels_changed_at = ? WHERE id = ?").bind(now, id));
  for (let i = 0; i < stmts.length; i += 400) await env.DB.batch(stmts.slice(i, i + 400));
  return json({ ok: true, changedAt: now });
}

async function strokeLabelsSynced(req: Request, env: Env, id: string): Promise<Response> {
  await getVideo(env, id);
  const body = (await req.json().catch(() => null)) as { changedAt?: unknown } | null;
  const at = typeof body?.changedAt === "string" ? body.changedAt : new Date(0).toISOString();
  await env.DB.prepare("UPDATE videos SET stroke_labels_synced_at = ? WHERE id = ?").bind(at, id).run();
  return json({ ok: true });
}

async function listVideos(env: Env): Promise<Response> {
  const rows = await env.DB.prepare("SELECT * FROM videos WHERE user_id = ? ORDER BY created_at DESC LIMIT 100")
    .bind(USER_ID)
    .all<VideoRow>();
  const node = await env.DB.prepare("SELECT name, seen_at FROM nodes WHERE user_id = ? ORDER BY seen_at DESC LIMIT 1")
    .bind(USER_ID)
    .first<{ name: string; seen_at: string }>();
  return json({
    videos: await Promise.all(rows.results.map((v) => describe(env, v))),
    node: node ? { name: node.name, seenAt: node.seen_at } : null,
    now: new Date().toISOString(), // 手机时钟可能不准，时间差都按服务器时间算
  });
}

// ---------- Mac 状态 ----------

// 节点上报自己的状态：原样保存（节点写、网页读），只限制大小。
// 「标注 › 素材」页：每段录像的标注材料准备得怎么样、标了多少、任务进度。不带分析结果。
async function listLabelingVideos(env: Env): Promise<Response> {
  const rows = await env.DB.prepare("SELECT * FROM videos WHERE user_id = ? AND status = 'uploaded' ORDER BY created_at DESC LIMIT 100")
    .bind(USER_ID)
    .all<VideoRow>();
  const counts = await env.DB.prepare("SELECT video_id, kind, status, COUNT(*) AS n FROM tasks WHERE active = 1 GROUP BY video_id, kind, status")
    .all<{ video_id: string; kind: string; status: string; n: number }>();
  const node = await env.DB.prepare("SELECT name, seen_at FROM nodes WHERE user_id = ? ORDER BY seen_at DESC LIMIT 1")
    .bind(USER_ID)
    .first<{ name: string; seen_at: string }>();
  const tasksOf = (id: string, kind: string) =>
    Object.fromEntries(counts.results.filter((c) => c.video_id === id && c.kind === kind).map((c) => [c.status, c.n]));
  return json({
    videos: await Promise.all(
      rows.results.map(async (v) => {
        const d = await describe(env, v);
        return {
          id: d.id, title: d.title, filename: d.filename, meta: d.meta, createdAt: d.createdAt, analyzed: ["done", "failed", "archived"].includes(d.job.status),
          annot: { ...d.annot, tasks: tasksOf(v.id, "shuttle") }, strokes: { ...d.strokes, tasks: tasksOf(v.id, "stroke") },
        };
      }),
    ),
    node: node ? { name: node.name, seenAt: node.seen_at } : null,
    now: new Date().toISOString(),
  });
}

async function putNodeStatus(req: Request, env: Env): Promise<Response> {
  const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
  const node = typeof body?.node === "string" && body.node ? body.node.slice(0, 80) : "node";
  if (!body?.status || typeof body.status !== "object") throw new HttpError(400, "缺少 status");
  const text = JSON.stringify(body.status);
  if (text.length > 200_000) throw new HttpError(413, "状态太大");
  const now = new Date().toISOString();
  await env.DB.prepare(
    `INSERT INTO nodes (user_id, name, seen_at, status, status_at) VALUES (?, ?, ?, ?, ?)
     ON CONFLICT (user_id, name) DO UPDATE SET seen_at = excluded.seen_at, status = excluded.status, status_at = excluded.status_at`,
  )
    .bind(USER_ID, node, now, text, now)
    .run();
  return json({ ok: true });
}

// 「Mac 状态」页要的全部内容：节点上报的状态，加上云端自己知道的作业（正在做的、排队的、最近失败的）。
async function macStatus(env: Env): Promise<Response> {
  const nodes = await env.DB.prepare("SELECT name, seen_at, status, status_at FROM nodes WHERE user_id = ? ORDER BY seen_at DESC LIMIT 5")
    .bind(USER_ID)
    .all<{ name: string; seen_at: string; status: string; status_at: string | null }>();
  const rows = await env.DB.prepare(
    `SELECT * FROM videos WHERE user_id = ? AND status = 'uploaded' AND (
       job_status IN ('queued', 'processing', 'failed') OR annot_status IN ('queued', 'preparing', 'failed')
       OR strokes_status IN ('queued', 'preparing', 'failed'))
     ORDER BY completed_at LIMIT 200`,
  )
    .bind(USER_ID)
    .all<VideoRow>();
  const now = new Date();
  type Job = { videoId: string; name: string; kind: string; stage: string; error: string; updatedAt: string | null; startedAt?: string | null; node?: string; stale?: boolean };
  const active: Job[] = [];
  const queued: Job[] = [];
  const failed: Job[] = [];
  for (const v of rows.results) {
    const kinds = [
      { kind: "process", status: v.job_status, busy: "processing", stage: v.job_stage, error: v.job_error, at: v.job_updated_at, lease: v.job_lease_until },
      { kind: "annotate", status: v.annot_status, busy: "preparing", stage: v.annot_stage, error: v.annot_error, at: v.annot_updated_at, lease: v.annot_lease_until },
      { kind: "strokes", status: v.strokes_status, busy: "preparing", stage: v.strokes_stage, error: v.strokes_error, at: v.strokes_updated_at, lease: v.strokes_lease_until },
    ];
    for (const k of kinds) {
      const job: Job = { videoId: v.id, name: v.title || v.filename, kind: k.kind, stage: k.stage, error: k.error.slice(0, 1500), updatedAt: k.at };
      if (k.status === "queued") queued.push(job);
      else if (k.status === "failed") failed.push(job);
      else if (k.status === k.busy) {
        // 租约过期 = 节点领了任务之后失联，下次有节点来问时会被重新领取
        job.stale = k.lease !== null && k.lease < now.toISOString();
        if (k.kind === "process") {
          job.startedAt = v.job_started_at;
          job.node = v.job_node;
        }
        active.push(job);
      }
    }
  }
  failed.sort((a, b) => (b.updatedAt ?? "").localeCompare(a.updatedAt ?? ""));
  return json({
    now: now.toISOString(),
    nodes: nodes.results.map((n) => {
      let status: unknown = null;
      try {
        status = n.status ? JSON.parse(n.status) : null;
      } catch {}
      return { name: n.name, seenAt: n.seen_at, statusAt: n.status_at, status };
    }),
    active,
    queued,
    failed: failed.slice(0, 10),
  });
}

// 标注员（不是管理员）能用的接口：自己领过任务的录像的标注材料和标注
const ANNOTATOR_PATHS = /^\/videos\/([0-9a-f-]{36})\/(annot\/files\/[^/]+|strokes\/files\/[^/]+|labels|stroke-labels)$/;

async function api(req: Request, env: Env, path: string, url: URL): Promise<Response> {
  const open = await publicApi(req, env, path, url);
  if (open) return open;
  const who = await identify(req, env);
  if (!who || (who.mediaOnly && path === "/me")) throw new HttpError(401, "没有登录");
  const m = req.method;
  if (path === "/me" && m === "GET") return json({ user: USER_ID, partSize: PART_SIZE, review: Boolean(aiKey(env)), ...meJson(who) });
  const team = await teamApi(req, env, path, who);
  if (team) return team;
  if (!who.owner) {
    // 不是超级管理员：只有标注这一边。管理员能看所有录像的标注材料和进度，标注员只能读自己领过任务的录像的
    if (who.status !== "active") throw new HttpError(403, "这个账号还没有被邀请");
    if (isAdmin(who) && path === "/labeling/videos" && m === "GET") return listLabelingVideos(env);
    const allowed = path.match(ANNOTATOR_PATHS);
    const write = allowed && m === "PUT" && !allowed[2].includes("/");
    if (!allowed || !(m === "GET" || write)) throw new HttpError(403, isAdmin(who) ? "只有超级管理员可以" : "只有管理员可以");
    if (!isAdmin(who)) await requireTaskOn(env, who, allowed[1]);
  }
  if (path === "/labeling/videos" && m === "GET") return listLabelingVideos(env);
  if (path === "/videos" && m === "POST") return createVideo(req, env);
  if (path === "/videos" && m === "GET") return listVideos(env);
  let match = path.match(/^\/videos\/([0-9a-f-]{36})$/);
  if (match && m === "GET") return json(await describe(env, await getVideo(env, match[1])));
  if (match && m === "PATCH") return renameVideo(req, env, match[1]);
  if (match && m === "DELETE") return deleteVideo(env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/parts\/(\d{1,5})$/);
  if (match && m === "PUT") return uploadPart(req, env, match[1], Number(match[2]));
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/complete$/);
  if (match && m === "POST") return completeVideo(env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/file$/);
  if (match && m === "GET") return downloadVideo(req, env, match[1]);
  if (path === "/session" && m === "POST") return startSession(req, env);
  if (path === "/ai" && m === "GET") return aiInfo(env);
  if (path === "/ai/settings" && m === "PUT") return saveAiSettings(req, env);
  if (path === "/node/claim" && m === "POST") return claimJob(req, env);
  if (path === "/node/status" && m === "POST") return putNodeStatus(req, env);
  if (path === "/status" && m === "GET") return macStatus(env);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/job$/);
  if (match && m === "POST") return updateJob(req, env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/job\/retry$/);
  if (match && m === "POST") return retryJob(req, env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/review$/);
  if (match && m === "POST") return makeReview(req, env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/review\/ask$/);
  if (match && m === "POST") return askReview(req, env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/original\/release$/);
  if (match && m === "POST") return releaseOriginal(env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/results\/([A-Za-z0-9][A-Za-z0-9._-]{0,119})$/);
  if (match && m === "GET") return getResult(req, env, match[1], match[2]);
  if (match && m === "PUT") return putResult(req, env, match[1], match[2]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/results\/([A-Za-z0-9][A-Za-z0-9._-]{0,119})\/multipart$/);
  if (match && (m === "POST" || m === "PUT")) return resultMultipart(req, env, match[1], match[2], url);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/annot$/);
  if (match && m === "POST") return requestAnnot(req, env, match[1]);
  if (match && m === "DELETE") return deleteAnnot(env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/annot\/job$/);
  if (match && m === "POST") return updateAnnotJob(req, env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/annot\/files\/([A-Za-z0-9][A-Za-z0-9._-]{0,119})$/);
  if (match && m === "GET") return getAnnotFile(req, env, match[1], match[2]);
  if (match && m === "PUT") return putAnnotFile(req, env, match[1], match[2]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/annot\/files\/([A-Za-z0-9][A-Za-z0-9._-]{0,119})\/multipart$/);
  if (match && (m === "POST" || m === "PUT")) return annotMultipart(req, env, match[1], match[2], url);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/labels$/);
  if (match && m === "GET") return getLabels(env, match[1], url);
  if (match && m === "PUT") return putLabels(req, env, match[1], who);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/labels\/synced$/);
  if (match && m === "POST") return labelsSynced(req, env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/strokes$/);
  if (match && m === "POST") return requestStrokes(env, match[1]);
  if (match && m === "DELETE") return deleteStrokes(env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/strokes\/job$/);
  if (match && m === "POST") return updateStrokesJob(req, env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/strokes\/files\/([A-Za-z0-9][A-Za-z0-9._-]{0,119})$/);
  if (match && m === "GET") return serveObject(req, env, strokesKey((await getVideo(env, match[1])).id, match[2]));
  if (match && m === "PUT") return putStrokesFile(req, env, match[1], match[2]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/strokes\/files\/([A-Za-z0-9][A-Za-z0-9._-]{0,119})\/multipart$/);
  if (match && (m === "POST" || m === "PUT")) {
    await getVideo(env, match[1]);
    const name = match[2];
    return multipart(req, env, strokesKey(match[1], name), strokesType(name), url, async (size) => json({ name, size }));
  }
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/stroke-labels$/);
  if (match && m === "GET") return getStrokeLabels(env, match[1], url);
  if (match && m === "PUT") return putStrokeLabels(req, env, match[1], who);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/stroke-labels\/synced$/);
  if (match && m === "POST") return strokeLabelsSynced(req, env, match[1]);
  throw new HttpError(404, "没有这个接口");
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (!url.pathname.startsWith("/api/v1/")) return env.ASSETS.fetch(req);
    try {
      return await api(req, env, url.pathname.slice("/api/v1".length), url);
    } catch (e) {
      if (e instanceof HttpError) return json({ error: e.message }, e.status);
      console.error(e);
      return json({ error: "服务器出错：" + (e instanceof Error ? e.message : String(e)) }, 500);
    }
  },
} satisfies ExportedHandler<Env>;
