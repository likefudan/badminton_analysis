// ba.llmat.dev 后端。阶段 0：录像分片上传到 R2，可断点续传。
// 接口在 /api/v1 下，用 Bearer 口令登录；网页、将来的 App 和 Mac 处理节点共用。

export interface Env {
  BUCKET: R2Bucket;
  DB: D1Database;
  ASSETS: Fetcher;
  ACCESS_TOKEN: string;
  NODE_TOKEN?: string; // Mac 处理节点自己的钥匙，和登录口令分开；没设置就只认登录口令
  ANTHROPIC_API_KEY?: string; // AI 报告用；没设置就不出报告
  ANTHROPIC_BASE_URL?: string; // 只在测试里改
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
};

type ResultRow = { video_id: string; name: string; r2_key: string; size: number; content_type: string; created_at: string };

class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });

async function sha256(text: string): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
}

const hex = (bytes: Uint8Array) => [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");

// 网页里的 <img> 和 <video> 带不了 Authorization 头，所以登录后发一个只有浏览器自己带的 cookie。
// 它的值由口令算出，不是口令本身；改口令后旧 cookie 自动失效。
async function sessionValue(env: Env): Promise<string> {
  return hex(await sha256("ba-session:" + env.ACCESS_TOKEN));
}

// 先各自做 SHA-256 再逐字节比较，比较时间不随口令内容变化
async function sameSecret(got: string, want: string): Promise<boolean> {
  const [a, b] = await Promise.all([sha256(got), sha256(want)]);
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
  return diff === 0;
}

async function authorized(req: Request, env: Env): Promise<boolean> {
  if (!env.ACCESS_TOKEN) return false;
  const header = req.headers.get("authorization") ?? "";
  if (header.startsWith("Bearer ")) {
    const got = header.slice(7);
    if (await sameSecret(got, env.ACCESS_TOKEN)) return true;
    return env.NODE_TOKEN ? sameSecret(got, env.NODE_TOKEN) : false;
  }
  // cookie 只用于读取（GET）：别的网站没法借它来改数据
  if (req.method !== "GET") return false;
  const cookie = (req.headers.get("cookie") ?? "").split(/;\s*/).find((c) => c.startsWith(SESSION_COOKIE + "="));
  return cookie ? sameSecret(cookie.slice(SESSION_COOKIE.length + 1), await sessionValue(env)) : false;
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
  return {
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

  const id = crypto.randomUUID();
  const ext = (filename.match(/\.([A-Za-z0-9]{1,5})$/)?.[1] ?? "bin").toLowerCase();
  const key = `originals/${USER_ID}/${id}.${ext}`;
  const upload = await env.BUCKET.createMultipartUpload(key, { httpMetadata: { contentType } });
  await env.DB.prepare(
    `INSERT INTO videos (id, user_id, filename, size, content_type, part_size, r2_key, upload_id, status, meta, options, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'uploading', ?, ?, ?)`,
  )
    .bind(id, USER_ID, filename, size, contentType, partSizeFor(size), key, upload.uploadId, meta, options, new Date().toISOString())
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
  await env.DB.batch([
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
  const now = new Date();
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
  await nodeSeen(env, node, now);
  return json({ video: row ? await describe(env, row) : null });
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
  const key = resultKey(id, name);
  const uploadId = url.searchParams.get("uploadId");
  if (req.method === "POST" && !uploadId) {
    const upload = await env.BUCKET.createMultipartUpload(key, { httpMetadata: { contentType: resultType(name) } });
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
  return recordResult(env, id, name, object.size);
}

async function getResult(req: Request, env: Env, id: string, name: string): Promise<Response> {
  const v = await getVideo(env, id);
  const res = await serveObject(req, env, resultKey(v.id, name));
  // 报告是程序生成的 HTML：不让它跑脚本，也不让它被当成本站页面的一部分
  res.headers.set("content-security-policy", "sandbox; default-src 'none'; img-src data: 'self'; style-src 'unsafe-inline'");
  res.headers.set("x-content-type-options", "nosniff");
  return res;
}

// ---------- AI 报告 ----------
// 节点上传 review.request.json（badminton review 写的：system、user、追问模板、模型），里面只有统计数字和文字。
// 这里拿它去问 Anthropic API，回答存成 review.md；追问的问答追加到 review.qa.md。key 只在 Worker 里。

interface ReviewRequest {
  model: string;
  max_tokens: number;
  system: string;
  user: string;
  ask: string; // 含 {question}
  title?: string;
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
  if (!env.ANTHROPIC_API_KEY) throw new HttpError(503, "网站还没有配置 ANTHROPIC_API_KEY，出不了 AI 报告");
  const text = await resultText(env, id, "review.request.json");
  let r: ReviewRequest | null = null;
  try {
    r = text ? (JSON.parse(text) as ReviewRequest) : null;
  } catch {}
  if (!r || typeof r.system !== "string" || typeof r.user !== "string" || typeof r.model !== "string")
    throw new HttpError(409, "这局还没有 AI 报告要用的统计摘要，请先重新处理");
  return r;
}

type Message = { role: "user" | "assistant"; content: string };

// 流式读取：长回答不会因为连接长时间没有数据被掐断
async function askClaude(env: Env, r: ReviewRequest, messages: Message[]): Promise<string> {
  const res = await fetch((env.ANTHROPIC_BASE_URL || "https://api.anthropic.com") + "/v1/messages", {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": env.ANTHROPIC_API_KEY!, "anthropic-version": "2023-06-01" },
    body: JSON.stringify({
      model: r.model,
      max_tokens: Math.min(Math.max(Number(r.max_tokens) || 8000, 256), 16000),
      system: r.system,
      messages,
      stream: true,
    }),
  });
  if (!res.ok || !res.body) {
    let detail = "";
    try {
      detail = ((await res.json()) as { error?: { message?: string } }).error?.message ?? "";
    } catch {}
    throw new HttpError(502, `调用 Anthropic API 失败（${res.status}）${detail ? "：" + detail.slice(0, 300) : ""}`);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  let text = "";
  let stop = "";
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
        continue;
      }
      if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") text += ev.delta.text;
      else if (ev.type === "message_delta" && ev.delta?.stop_reason) stop = ev.delta.stop_reason;
      else if (ev.type === "error") throw new HttpError(502, "Anthropic API 出错：" + String(ev.error?.message ?? "").slice(0, 300));
    }
  }
  text = text.trim();
  if (stop === "refusal") throw new HttpError(502, "模型拒绝回答这个请求");
  if (!text) throw new HttpError(502, "模型没有返回文字");
  if (stop === "max_tokens") text += "\n\n（回答到长度上限被截断）";
  return text;
}

async function makeReview(env: Env, id: string): Promise<Response> {
  await getVideo(env, id);
  const r = await reviewRequest(env, id);
  const body = await askClaude(env, r, [{ role: "user", content: r.user }]);
  const note = `> 由 ${r.model} 根据程序算出的统计数字生成，它没有看视频。数字以上面的表和卡片为准。`;
  await saveResultText(env, id, "review.md", `${body}\n\n${note}\n`);
  await deleteResult(env, id, "review.qa.md"); // 追问是接着旧报告问的，报告换了就不再对得上
  return json(await describe(env, await getVideo(env, id)));
}

async function askReview(req: Request, env: Env, id: string): Promise<Response> {
  await getVideo(env, id);
  const body = (await req.json().catch(() => null)) as { question?: unknown } | null;
  const question = typeof body?.question === "string" ? body.question.trim().slice(0, 1000) : "";
  if (!question) throw new HttpError(400, "请写下要问的问题");
  const r = await reviewRequest(env, id);
  const report = await resultText(env, id, "review.md");
  const earlier = (await resultText(env, id, "review.qa.md")) ?? "";
  const messages: Message[] = [{ role: "user", content: r.user }];
  if (report) messages.push({ role: "assistant", content: report });
  else messages[0].content += "\n\n（先不用写点评，等我提问。）";
  if (!report) messages.push({ role: "assistant", content: "好的，请问。" });
  const ask = (q: string) => (r.ask || "追问：{question}").replace("{question}", q);
  // 之前的问答也带上，这样可以接着上一个问题问
  for (const m of earlier.matchAll(/^## 问：(.*)\n\n([\s\S]*?)(?=\n\n## 问：|\s*$)/gm)) {
    messages.push({ role: "user", content: ask(m[1]) }, { role: "assistant", content: m[2].trim() });
  }
  messages.push({ role: "user", content: ask(question) });
  const answer = await askClaude(env, r, messages);
  const oneLine = question.replace(/\s+/g, " ");
  await saveResultText(env, id, "review.qa.md", `${earlier}## 问：${oneLine}\n\n${answer.replace(/^## /gm, "### ")}\n\n`);
  return json({ answer, video: await describe(env, await getVideo(env, id)) });
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

async function api(req: Request, env: Env, path: string, url: URL): Promise<Response> {
  if (!(await authorized(req, env))) throw new HttpError(401, "口令不对");
  const m = req.method;
  if (path === "/me" && m === "GET") return json({ user: USER_ID, partSize: PART_SIZE, review: Boolean(env.ANTHROPIC_API_KEY) });
  if (path === "/videos" && m === "POST") return createVideo(req, env);
  if (path === "/videos" && m === "GET") return listVideos(env);
  let match = path.match(/^\/videos\/([0-9a-f-]{36})$/);
  if (match && m === "GET") return json(await describe(env, await getVideo(env, match[1])));
  if (match && m === "DELETE") return deleteVideo(env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/parts\/(\d{1,5})$/);
  if (match && m === "PUT") return uploadPart(req, env, match[1], Number(match[2]));
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/complete$/);
  if (match && m === "POST") return completeVideo(env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/file$/);
  if (match && m === "GET") return downloadVideo(req, env, match[1]);
  if (path === "/session" && m === "POST") return startSession(req, env);
  if (path === "/node/claim" && m === "POST") return claimJob(req, env);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/job$/);
  if (match && m === "POST") return updateJob(req, env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/job\/retry$/);
  if (match && m === "POST") return retryJob(req, env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/review$/);
  if (match && m === "POST") return makeReview(env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/review\/ask$/);
  if (match && m === "POST") return askReview(req, env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/original\/release$/);
  if (match && m === "POST") return releaseOriginal(env, match[1]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/results\/([A-Za-z0-9][A-Za-z0-9._-]{0,119})$/);
  if (match && m === "GET") return getResult(req, env, match[1], match[2]);
  if (match && m === "PUT") return putResult(req, env, match[1], match[2]);
  match = path.match(/^\/videos\/([0-9a-f-]{36})\/results\/([A-Za-z0-9][A-Za-z0-9._-]{0,119})\/multipart$/);
  if (match && (m === "POST" || m === "PUT")) return resultMultipart(req, env, match[1], match[2], url);
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
