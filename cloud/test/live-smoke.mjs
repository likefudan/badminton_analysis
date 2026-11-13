// 对线上网站走一遍真实的分片上传：2 片（32 MiB + 1 MiB），合并、下载核对、删除。
// 用法：SITE=https://ba.llmat.dev BA_ACCESS_TOKEN=口令 node test/live-smoke.mjs
import assert from "node:assert/strict";
import { createHash, randomBytes } from "node:crypto";

const BASE = `${process.env.SITE}/api/v1`;
const auth = { authorization: `Bearer ${process.env.BA_ACCESS_TOKEN}` };
const call = (path, init = {}) => fetch(BASE + path, { ...init, headers: { ...auth, ...(init.headers || {}) } });
const sha = (b) => createHash("sha256").update(b).digest("hex");
// 核对状态码，失败时带上返回内容；成功时返回解析后的 JSON
async function expect(res, status, what) {
  const text = await res.text();
  assert.equal(res.status, status, `${what}: ${text}`);
  return text ? JSON.parse(text) : null;
}

const data = randomBytes(33 * 1024 * 1024);
const created = await call("/videos", {
  method: "POST", headers: { "content-type": "application/json" },
  body: JSON.stringify({ filename: "deploy-smoke.bin", size: data.length, contentType: "application/octet-stream", meta: {} }),
});
const video = await expect(created, 201, "新建上传");
try {
  assert.equal(video.partCount, 2);
  const started = Date.now();
  for (let n = 1; n <= video.partCount; n++) {
    const body = data.subarray((n - 1) * video.partSize, Math.min(n * video.partSize, data.length));
    const res = await call(`/videos/${video.id}/parts/${n}`, { method: "PUT", body });
    await expect(res, 200, `第 ${n} 片`);
  }
  const seconds = (Date.now() - started) / 1000;
  const done = await call(`/videos/${video.id}/complete`, { method: "POST" });
  assert.equal((await expect(done, 200, "合并")).status, "uploaded");

  const file = await call(`/videos/${video.id}/file`);
  assert.equal(file.status, 200);
  assert.equal(sha(Buffer.from(await file.arrayBuffer())), sha(data));

  const ranged = await call(`/videos/${video.id}/file`, { headers: { range: "bytes=100-199" } });
  assert.equal(ranged.status, 206);
  assert.deepEqual(Buffer.from(await ranged.arrayBuffer()), data.subarray(100, 200));
  console.log(`分片上传、合并、下载核对通过：33 MiB 上传用时 ${seconds.toFixed(1)} 秒`);
} finally {
  const del = await call(`/videos/${video.id}`, { method: "DELETE" });
  assert.equal(del.status, 200, "删除测试文件失败");
  console.log("测试文件已删除");
}
