// ba.llmat.dev 后端。阶段 0：录像分片上传到 R2，可断点续传。
// 接口在 /api/v1 下，用 Bearer 口令登录；网页、将来的 App 和 Mac 处理节点共用。

export interface Env {
  BUCKET: R2Bucket;
  DB: D1Database;
  ASSETS: Fetcher;
  ACCESS_TOKEN: string;
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
};

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

// 先各自做 SHA-256 再逐字节比较，比较时间不随口令内容变化
async function authorized(req: Request, env: Env): Promise<boolean> {
  if (!env.ACCESS_TOKEN) return false;
  const header = req.headers.get("authorization") ?? "";
  if (!header.startsWith("Bearer ")) return false;
  const [a, b] = await Promise.all([sha256(header.slice(7)), sha256(env.ACCESS_TOKEN)]);
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
  return diff === 0;
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
  return {
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

  const id = crypto.randomUUID();
  const ext = (filename.match(/\.([A-Za-z0-9]{1,5})$/)?.[1] ?? "bin").toLowerCase();
  const key = `originals/${USER_ID}/${id}.${ext}`;
  const upload = await env.BUCKET.createMultipartUpload(key, { httpMetadata: { contentType } });
  await env.DB.prepare(
    `INSERT INTO videos (id, user_id, filename, size, content_type, part_size, r2_key, upload_id, status, meta, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'uploading', ?, ?)`,
  )
    .bind(id, USER_ID, filename, size, contentType, partSizeFor(size), key, upload.uploadId, meta, new Date().toISOString())
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
  await env.DB.prepare("UPDATE videos SET status = 'uploaded', completed_at = ? WHERE id = ?")
    .bind(new Date().toISOString(), id)
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
  const object = await env.BUCKET.get(v.r2_key, { range: req.headers });
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

async function listVideos(env: Env): Promise<Response> {
  const rows = await env.DB.prepare("SELECT * FROM videos WHERE user_id = ? ORDER BY created_at DESC LIMIT 100")
    .bind(USER_ID)
    .all<VideoRow>();
  return json({ videos: await Promise.all(rows.results.map((v) => describe(env, v))) });
}

async function api(req: Request, env: Env, path: string): Promise<Response> {
  if (!(await authorized(req, env))) throw new HttpError(401, "口令不对");
  const m = req.method;
  if (path === "/me" && m === "GET") return json({ user: USER_ID, partSize: PART_SIZE });
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
  throw new HttpError(404, "没有这个接口");
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (!url.pathname.startsWith("/api/v1/")) return env.ASSETS.fetch(req);
    try {
      return await api(req, env, url.pathname.slice("/api/v1".length));
    } catch (e) {
      if (e instanceof HttpError) return json({ error: e.message }, e.status);
      console.error(e);
      return json({ error: "服务器出错：" + (e instanceof Error ? e.message : String(e)) }, 500);
    }
  },
} satisfies ExportedHandler<Env>;
