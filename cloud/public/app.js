// 手机网页：把录像切片传到 /api/v1（可断点续传），看每一局的处理状态和分析结果。
"use strict";

const API = "/api/v1";
const CONCURRENCY = 3;
const MAX_TRIES = 8;          // 服务器明确报错时最多试几次；网络问题会一直重试
const STALL_MS = 30 * 1000;   // 一片这么久没有任何进展就算卡住，断开重传
const HIDDEN_KICK_MS = 3000;  // 离开页面超过这么久，回来后把在途的请求断开重传
const $ = (id) => document.getElementById(id);

let token = "";
try { token = localStorage.getItem("ba.token") || ""; } catch {}

let file = null;
let meta = {};
let run = null; // 当前上传：{ video, controller, paused, ... }
let wakeLock = null;
let hiddenCount = 0;
let hiddenSince = 0;
let hiddenTotalMs = 0;

// ---------- 小工具 ----------

function fmtBytes(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " MB";
  return Math.round(n / 1e3) + " KB";
}
function fmtDur(s) {
  if (!isFinite(s)) return "—";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h} 小时 ${m} 分` : m ? `${m} 分 ${sec} 秒` : `${sec} 秒`;
}
const fmtSpeed = (bytesPerSec) => (bytesPerSec * 8 / 1e6).toFixed(1) + " Mbps";

function log(msg) {
  const t = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  const el = $("log");
  el.textContent += `${t} ${msg}\n`;
  el.scrollTop = el.scrollHeight;
}

async function api(path, options = {}) {
  const res = await fetch(API + path, {
    ...options,
    headers: { authorization: "Bearer " + token, ...(options.headers || {}) },
  });
  let data = null;
  try { data = await res.json(); } catch {}
  if (!res.ok) {
    const err = new Error((data && data.error) || "HTTP " + res.status);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

// ---------- 登录 ----------

async function tryLogin() {
  if (!token) return showLogin("");
  try {
    await api("/me");
    await api("/session", { method: "POST" }); // 图片和视频靠 cookie 读取
    $("login").hidden = true;
    $("app").hidden = false;
    route();
    refreshList();
    loadAi();
  } catch (e) {
    showLogin(e.status === 401 ? "口令不对" : "连不上服务器：" + e.message);
  }
}
function showLogin(msg) {
  $("login").hidden = false;
  $("app").hidden = true;
  $("loginMsg").textContent = msg;
}
$("loginBtn").onclick = () => {
  token = $("token").value.trim();
  try { localStorage.setItem("ba.token", token); } catch {}
  tryLogin();
};
$("logout").onclick = () => {
  token = "";
  try { localStorage.removeItem("ba.token"); } catch {}
  showLogin("");
};

// ---------- 读视频参数 ----------

// 用 <video> 读分辨率和时长；播放一小段，用相邻帧的时间差估计帧率。
function probeVideo(f) {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    const url = URL.createObjectURL(f);
    const out = { width: 0, height: 0, duration: NaN, fps: NaN };
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      v.pause();
      v.removeAttribute("src");
      v.load();
      URL.revokeObjectURL(url);
      resolve(out);
    };
    v.muted = true;
    v.playsInline = true;
    v.preload = "metadata";
    v.onerror = finish;
    v.onloadedmetadata = () => {
      out.width = v.videoWidth;
      out.height = v.videoHeight;
      out.duration = v.duration;
      if (!("requestVideoFrameCallback" in v)) return finish();
      const times = [];
      const onFrame = (_now, m) => {
        times.push(m.mediaTime);
        if (times.length >= 40) {
          const deltas = [];
          for (let i = 1; i < times.length; i++) if (times[i] > times[i - 1]) deltas.push(times[i] - times[i - 1]);
          deltas.sort((a, b) => a - b);
          // 掉帧会让间隔变大，所以取偏小的分位数
          if (deltas.length) out.fps = 1 / deltas[Math.floor(deltas.length * 0.25)];
          return finish();
        }
        v.requestVideoFrameCallback(onFrame);
      };
      v.requestVideoFrameCallback(onFrame);
      v.play().catch(finish);
    };
    setTimeout(finish, 6000);
    v.src = url;
  });
}

function fileKey(f) {
  return `ba.upload.${f.name}.${f.size}.${f.lastModified}`;
}

$("file").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  if (run && !run.finished) return log("正在上传，先暂停再换文件");
  file = f;
  $("info").hidden = false;
  $("progress").hidden = true;
  $("upDone").hidden = true;
  $("upTitle").value = "";
  $("iName").textContent = f.name;
  $("iSize").textContent = fmtBytes(f.size);
  $("iRes").textContent = $("iFps").textContent = $("iDur").textContent = $("iRate").textContent = "读取中";
  $("verdict").textContent = "";
  $("startBtn").disabled = true;
  log(`选了 ${f.name}，${fmtBytes(f.size)}，类型 ${f.type || "未知"}`);

  const p = await probeVideo(f);
  const perMin = p.duration > 0 ? f.size / (p.duration / 60) : NaN;
  meta = {
    width: p.width, height: p.height,
    duration: isFinite(p.duration) ? Math.round(p.duration * 100) / 100 : null,
    fps: isFinite(p.fps) ? Math.round(p.fps * 10) / 10 : null,
    lastModified: f.lastModified,
    userAgent: navigator.userAgent,
  };
  $("iRes").textContent = p.width ? `${p.width} × ${p.height}` : "读不出";
  $("iFps").textContent = isFinite(p.fps) ? p.fps.toFixed(1) : "读不出";
  $("iDur").textContent = fmtDur(p.duration);
  $("iRate").textContent = isFinite(perMin) ? fmtBytes(perMin) + "/分钟" : "—";

  const is4k = Math.max(p.width, p.height) >= 3840;
  const is60 = isFinite(p.fps) && p.fps > 50;
  const v = $("verdict");
  if (!p.width) { v.className = "warn"; v.textContent = "浏览器读不出这个视频的参数，仍然可以上传。"; }
  else if (is4k && is60) { v.className = "ok"; v.textContent = "是 4K60 原片，可以上传。"; }
  else if (is4k && !isFinite(p.fps)) { v.className = "warn"; v.textContent = "是 4K，帧率读不出。看“每分钟大小”：4K60 大约 400 MB/分钟。"; }
  else { v.className = "bad"; v.textContent = "不是 4K60，可能在选择时被系统压缩了。分析需要原片，建议换一种方式重新选。"; }
  log(`参数：${p.width}×${p.height}，约 ${isFinite(p.fps) ? p.fps.toFixed(1) : "?"} fps，${fmtDur(p.duration)}，${isFinite(perMin) ? fmtBytes(perMin) : "?"}/分钟`);

  let saved = null;
  try { saved = localStorage.getItem(fileKey(f)); } catch {}
  $("startBtn").textContent = saved ? "继续上传" : "开始上传";
  $("startBtn").disabled = false;
  $("startBtn").hidden = false;
  $("pauseBtn").hidden = true;
};

// ---------- 上传 ----------

async function requestWakeLock() {
  if (!("wakeLock" in navigator)) { $("pWake").textContent = "浏览器不支持"; return; }
  try {
    wakeLock = await navigator.wakeLock.request("screen");
    $("pWake").textContent = "已开启";
    wakeLock.addEventListener("release", () => { $("pWake").textContent = "已释放"; });
  } catch (e) {
    $("pWake").textContent = "申请失败";
  }
}

document.addEventListener("visibilitychange", () => {
  if (!run || run.finished) return;
  if (document.hidden) {
    hiddenCount++;
    hiddenSince = Date.now();
    log("页面进入后台");
  } else {
    const ms = hiddenSince ? Date.now() - hiddenSince : 0;
    hiddenTotalMs += ms;
    hiddenSince = 0;
    log(`页面回到前台，离开了 ${fmtDur(ms / 1000)}`);
    if (!run.paused) requestWakeLock();
    // iOS 在后台会冻结或掐断连接，回来后旧请求常常既不成功也不报错
    if (ms >= HIDDEN_KICK_MS) kick("离开过页面");
  }
  $("pHidden").textContent = `${hiddenCount} 次，共 ${fmtDur(hiddenTotalMs / 1000)}`;
});

// 把在途的请求断开，让它们马上重传（已确认的片不受影响）
function kick(reason) {
  if (!run || run.finished || run.paused) return;
  for (const xhr of run.xhrs) { xhr.kickReason = reason; xhr.abort(); }
}
window.addEventListener("online", () => { if (run && !run.finished) { log("网络恢复"); kick("网络恢复"); } });
window.addEventListener("offline", () => { if (run && !run.finished) log("网络断开，等恢复后自动继续"); });

function putPart(video, n, signal, onProgress) {
  // 用 XMLHttpRequest 而不是 fetch，是为了拿到上传进度
  return new Promise((resolve, reject) => {
    const start = (n - 1) * video.partSize;
    const blob = file.slice(start, Math.min(start + video.partSize, file.size));
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `${API}/videos/${video.id}/parts/${n}`);
    xhr.setRequestHeader("authorization", "Bearer " + token);
    xhr.setRequestHeader("content-type", "application/octet-stream");
    // 卡住检测：没有新的进度就断开。字节发完后等服务器回应，多给一倍时间
    let lastMove = Date.now(), lastLoaded = -1, allSent = false;
    const watchdog = setInterval(() => {
      if (document.hidden) { lastMove = Date.now(); return; } // 后台时计时器不准，回前台时统一处理
      if (Date.now() - lastMove > (allSent ? STALL_MS * 2 : STALL_MS)) { xhr.kickReason = "卡住了"; xhr.abort(); }
    }, 2000);
    const cleanup = () => { clearInterval(watchdog); run && run.xhrs.delete(xhr); };
    xhr.upload.onprogress = (e) => {
      if (e.loaded !== lastLoaded) { lastLoaded = e.loaded; lastMove = Date.now(); }
      allSent = e.loaded >= blob.size;
      onProgress(e.loaded);
    };
    xhr.onloadend = cleanup;
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve(blob.size);
      let msg = "HTTP " + xhr.status;
      try { msg = JSON.parse(xhr.responseText).error || msg; } catch {}
      const err = new Error(msg);
      err.status = xhr.status;
      reject(err);
    };
    xhr.onerror = () => reject(new Error("网络中断"));
    xhr.ontimeout = () => reject(new Error("超时"));
    xhr.onabort = () => reject(xhr.kickReason && !signal.aborted
      ? Object.assign(new Error(xhr.kickReason), { kicked: true })
      : Object.assign(new Error("已暂停"), { aborted: true }));
    xhr.timeout = 10 * 60 * 1000;
    signal.addEventListener("abort", () => xhr.abort(), { once: true });
    run.xhrs.add(xhr);
    xhr.send(blob);
  });
}

const sleep = (ms, signal) => new Promise((resolve) => {
  const t = setTimeout(resolve, ms);
  signal.addEventListener("abort", () => { clearTimeout(t); resolve(); }, { once: true });
});
// 断网时等到恢复（或暂停）为止
const untilOnline = (signal) => new Promise((resolve) => {
  if (navigator.onLine !== false) return resolve();
  const done = () => { window.removeEventListener("online", done); resolve(); };
  window.addEventListener("online", done);
  signal.addEventListener("abort", done, { once: true });
});

async function startUpload() {
  if (!file) return;
  const key = fileKey(file);
  let video = null;
  let savedId = null;
  try { savedId = localStorage.getItem(key); } catch {}
  if (savedId) {
    try {
      video = await api("/videos/" + savedId);
      if (video.size !== file.size) video = null;
    } catch (e) {
      if (e.status !== 404) return log("读不到上次的进度：" + e.message);
    }
  }
  if (video && video.status === "uploaded") {
    log("这个文件之前已经传完了");
    try { localStorage.removeItem(key); } catch {}
    return refreshList();
  }
  if (!video) {
    video = await api("/videos", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        filename: file.name, size: file.size, contentType: file.type, meta, title: $("upTitle").value.trim(),
        options: { mode: document.querySelector('input[name="mode"]:checked').value },
      }),
    });
    try { localStorage.setItem(key, video.id); } catch {}
    log(`新建上传，共 ${video.partCount} 片，每片 ${fmtBytes(video.partSize)}`);
  } else {
    log(`继续上传，已有 ${video.uploadedParts.length}/${video.partCount} 片`);
  }

  const done = new Set(video.uploadedParts);
  const todo = [];
  for (let n = 1; n <= video.partCount; n++) if (!done.has(n)) todo.push(n);

  const controller = new AbortController();
  run = {
    video, controller, paused: false, finished: false,
    confirmed: video.uploadedBytes, // 服务器已确认的字节
    inflight: new Map(),            // 正在传的片 -> 已发送字节
    xhrs: new Set(),                // 在途的请求，卡住时用来断开
    sentThisRun: 0, startedAt: Date.now(), retries: 0, samples: [],
  };
  hiddenCount = 0; hiddenTotalMs = 0; hiddenSince = 0;
  $("pHidden").textContent = "0 次";
  $("pRetries").textContent = "0";
  $("progress").hidden = false;
  $("pTitle").textContent = "上传中";
  $("startBtn").hidden = true;
  $("pauseBtn").hidden = false;
  requestWakeLock();
  const timer = setInterval(render, 500);
  const r = run;

  async function worker() {
    while (todo.length && !controller.signal.aborted) {
      const n = todo.shift();
      for (let attempt = 1, httpFails = 0; ; attempt++) {
        try {
          r.inflight.set(n, 0);
          const size = await putPart(video, n, controller.signal, (loaded) => r.inflight.set(n, loaded));
          r.inflight.delete(n);
          r.confirmed += size;
          r.sentThisRun += size;
          break;
        } catch (e) {
          r.inflight.delete(n);
          if (e.aborted || controller.signal.aborted) { todo.unshift(n); return; }
          if (e.status === 401 || e.status === 404 || e.status === 409) throw e;
          // 服务器明确报错才计入放弃次数；断网、卡住、切后台一直重试
          if (e.status && ++httpFails >= MAX_TRIES) throw e;
          r.retries++;
          $("pRetries").textContent = String(r.retries);
          if (e.kicked) {
            log(`第 ${n} 片${e.message}，马上重传`);
            attempt = 0;
          } else {
            const wait = Math.min(15000, 1000 * 2 ** Math.min(attempt - 1, 4));
            log(`第 ${n} 片失败（${e.message}），${wait / 1000} 秒后重试`);
            await sleep(wait, controller.signal);
          }
          await untilOnline(controller.signal);
        }
      }
    }
  }

  try {
    await Promise.all(Array.from({ length: CONCURRENCY }, worker));
    if (controller.signal.aborted) {
      log("已暂停");
      $("pTitle").textContent = "已暂停";
      return;
    }
    $("pTitle").textContent = "正在合并";
    const result = await api(`/videos/${video.id}/complete`, { method: "POST" });
    const secs = (Date.now() - r.startedAt) / 1000;
    r.finished = true;
    try { localStorage.removeItem(key); } catch {}
    $("pTitle").textContent = "上传完成，已经排队等 Mac 处理";
    $("upDone").hidden = false;
    log(`完成：${fmtBytes(result.size)}，本次用时 ${fmtDur(secs)}，平均 ${fmtSpeed(r.sentThisRun / Math.max(secs, 1))}，重试 ${r.retries} 次，进入后台 ${hiddenCount} 次`);
    refreshList();
  } catch (e) {
    controller.abort();
    $("pTitle").textContent = "上传出错";
    log("上传出错：" + e.message + "。可以点“继续上传”。");
  } finally {
    clearInterval(timer);
    render();
    if (wakeLock) wakeLock.release().catch(() => {});
    $("pauseBtn").hidden = true;
    if (!r.finished) {
      r.finished = true;
      $("startBtn").textContent = "继续上传";
      $("startBtn").hidden = false;
    }
  }
}

function render() {
  if (!run) return;
  const r = run;
  let flying = 0;
  for (const v of r.inflight.values()) flying += v;
  const sent = Math.min(r.video.size, r.confirmed + flying);
  const now = Date.now();
  r.samples.push([now, r.sentThisRun + flying]);
  while (r.samples.length > 2 && now - r.samples[0][0] > 10000) r.samples.shift();
  const [t0, b0] = r.samples[0];
  const speed = now > t0 ? Math.max(0, (r.sentThisRun + flying - b0) / ((now - t0) / 1000)) : 0;
  const elapsed = (now - r.startedAt) / 1000;
  const avg = (r.sentThisRun + flying) / Math.max(elapsed, 1);
  $("pBar").style.width = (sent / r.video.size * 100).toFixed(1) + "%";
  $("pBytes").textContent = `${fmtBytes(sent)} / ${fmtBytes(r.video.size)}（${(sent / r.video.size * 100).toFixed(1)}%）`;
  $("pSpeed").textContent = fmtSpeed(speed);
  $("pAvg").textContent = fmtSpeed(avg);
  $("pEta").textContent = speed > 0 ? fmtDur((r.video.size - sent) / speed) : "—";
  $("pElapsed").textContent = fmtDur(elapsed);
}

$("startBtn").onclick = () => startUpload().catch((e) => log("出错：" + e.message));
$("pauseBtn").onclick = () => { if (run) { run.paused = true; run.controller.abort(); } };

// ---------- 录像列表和结果 ----------

let videos = [];
let openId = null;     // 正在看哪一段
let shownKey = "";     // 结果区已经画过的内容，没变化就不重画（不然视频会被打断）
let pollTimer = null;
let node = null;       // Mac 处理节点：{ name, seenAt }
let clockSkew = 0;     // 服务器时间 - 手机时间（毫秒），时间差都按服务器时间算
let filter = "all";
let tab = "overview";
let ai = null;         // GET /ai：能选的模型、默认模型、单价、累计用量
const serverNow = () => Date.now() + clockSkew;
const NODE_ONLINE_MS = 90 * 1000; // 节点空闲时每 15 秒来问一次任务，处理中每 20 秒汇报一次

// 节点上报之后又过了多久（秒）：用来让“已用时间”和“预计剩余”在两次上报之间继续走
const sinceReport = (j) => (j && j.updatedAt ? Math.max(0, (serverNow() - new Date(j.updatedAt)) / 1000) : 0);
const fmtAbout = (s) => (s < 50 ? `约 ${Math.max(5, Math.round(s / 5) * 5)} 秒` : `约 ${Math.max(1, Math.round(s / 60))} 分钟`);
const nodeOnline = () => node && serverNow() - new Date(node.seenAt) < NODE_ONLINE_MS;
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};
const titleOf = (v) => v.title || v.filename;
const resultUrl = (v, name) => `${API}/videos/${v.id}/results/${encodeURIComponent(name)}`;
// 录像是什么时候拍的：手机上文件的时间，没有就用上传时间
const shotAt = (v) => new Date((v.meta && v.meta.lastModified) || v.createdAt);
const fmtDate = (d) => d.toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
const fmtClock = (s) => {
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
};

function queueText(v) {
  const ahead = videos.filter((x) => x.id !== v.id && x.status === "uploaded" && x.job &&
    (x.job.status === "processing" || (x.job.status === "queued" && x.completedAt < v.completedAt))).length;
  const where = ahead ? `排队中，前面还有 ${ahead} 局` : "排队中，下一个就是它";
  if (!node) return where + "。Mac 上的处理程序还没连接过";
  return where + (nodeOnline() ? "" : "。Mac 现在离线，上线后自动开始");
}

function statusText(v) {
  if (v.status !== "uploaded") return `上传中 ${(v.uploadedBytes / v.size * 100).toFixed(0)}%`;
  const j = v.job || {};
  if (j.status === "queued") return queueText(v);
  if (j.status === "processing") {
    const eta = j.progress && j.progress.etaSeconds != null ? Math.max(0, j.progress.etaSeconds - sinceReport(j)) : null;
    return "处理中：" + (j.stage || "") + (eta != null ? `，${fmtAbout(eta).replace("约", "约剩")}` : "");
  }
  if (j.status === "done") return "已完成";
  if (j.status === "failed") return "处理失败";
  if (j.status === "archived") return "云端结果已清理，Mac 上还有";
  return "已上传";
}
const busy = (v) => v.status === "uploaded" && v.job && (v.job.status === "queued" || v.job.status === "processing");
const annotBusy = (v) => (v.annot && (v.annot.status === "queued" || v.annot.status === "preparing")) ||
  (v.strokes && (v.strokes.status === "queued" || v.strokes.status === "preparing"));


// 卡片右下角的小标签：一眼看出这段录像到哪一步了
function badgeOf(v) {
  if (v.status !== "uploaded") {
    const mine = run && !run.finished && run.video.id === v.id;
    return [mine ? "busy" : "warn", mine ? `上传中 ${(v.uploadedBytes / v.size * 100).toFixed(0)}%` : `没传完 ${(v.uploadedBytes / v.size * 100).toFixed(0)}%`];
  }
  const j = v.job || {};
  if (j.status === "queued") return ["busy", "排队中"];
  if (j.status === "processing") return ["busy", "处理中" + (j.progress && j.progress.etaSeconds != null ? `，${fmtAbout(Math.max(0, j.progress.etaSeconds - sinceReport(j))).replace("约", "约剩")}` : "")];
  if (j.status === "done") return ["done", "已完成"];
  if (j.status === "failed") return ["failed", "处理失败"];
  if (j.status === "archived") return ["", "云端结果已清理"];
  return ["", "已上传"];
}
const FILTERS = [
  ["all", "全部", () => true],
  ["busy", "进行中", (v) => v.status !== "uploaded" || busy(v)],
  ["done", "已完成", (v) => v.status === "uploaded" && v.job && v.job.status === "done"],
  ["failed", "失败", (v) => v.status === "uploaded" && v.job && v.job.status === "failed"],
];

function coverOf(v) {
  const have = new Set(v.results.map((r) => r.name));
  for (const name of ["cover.jpg", "players.jpg", "court.jpg", "court.rejected.jpg"]) if (have.has(name)) return resultUrl(v, name);
  return null;
}

// 整段处理的进度（0–100），画在封面底部
function overallPercent(v) {
  if (v.status !== "uploaded") return v.uploadedBytes / v.size * 100;
  const p = v.job && v.job.status === "processing" && v.job.progress;
  if (!p || !Array.isArray(p.steps) || !p.steps.length) return null;
  const done = p.steps.filter((s) => s.status === "done" || s.status === "skipped").length;
  const running = p.steps.find((s) => s.status === "running");
  return (done + (running && running.percent ? running.percent / 100 : 0)) / p.steps.length * 100;
}

function renderList() {
  const chips = $("filters");
  chips.textContent = "";
  for (const [key, label, test] of FILTERS) {
    const n = videos.filter(test).length;
    if (key !== "all" && !n) continue;
    const b = el("button", key === filter ? "on" : "", label);
    b.append(el("span", "n", String(n)));
    b.onclick = () => { filter = key; renderList(); };
    chips.append(b);
  }
  chips.hidden = videos.length < 2;
  const box = $("list");
  box.textContent = "";
  const test = (FILTERS.find((f) => f[0] === filter) || FILTERS[0])[2];
  const shown = videos.filter(test);
  if (!videos.length) box.append(el("div", "empty", "还没有录像。点上面的「上传录像」，传完后家里的 Mac 会自动分析。"));
  else if (!shown.length) box.append(el("div", "empty", "这一类里没有录像"));
  for (const v of shown) {
    const card = el("div", "vcard");
    const cover = el("div", "cover");
    const src = coverOf(v);
    if (src) cover.style.backgroundImage = `url("${src}")`;
    else cover.append(el("div", "ph", v.status !== "uploaded" ? "上传中" : busy(v) ? "等待处理" : "没有封面"));
    if (v.meta && v.meta.duration) cover.append(el("span", "len", fmtClock(v.meta.duration)));
    const pct = overallPercent(v);
    if (pct != null) {
      const bar = el("div", "pbar");
      const fill = el("div");
      fill.style.width = pct.toFixed(1) + "%";
      bar.append(fill);
      cover.append(bar);
    }
    const body = el("div", "body");
    body.append(el("div", "name", titleOf(v)));
    body.append(el("div", "sub", `${fmtDate(shotAt(v))} · ${v.options && v.options.mode === "doubles" ? "双打" : "单打"} · ${fmtBytes(v.size)}`));
    const [cls, text] = badgeOf(v);
    body.append(el("span", "badge " + cls, text));
    card.append(cover, body);
    card.onclick = () => { location.hash = "#/v/" + v.id; };
    box.append(card);
  }
  renderNodeLine();
}

async function refreshList() {
  try {
    const data = await api("/videos");
    videos = data.videos;
    node = data.node;
    if (data.now) clockSkew = new Date(data.now) - Date.now();
  } catch (e) {
    $("list").textContent = "";
    $("list").append(el("div", "empty", "读取失败：" + e.message));
    return;
  }
  renderList();
  if (openId) renderDetail();
  clearTimeout(pollTimer);
  if (videos.some(busy) || videos.some(annotBusy) || (run && !run.finished)) pollTimer = setTimeout(refreshList, 3000);
}

function renderNodeLine() {
  let text = "家里的 Mac 还没有连接过";
  if (node) {
    const ago = Math.max(0, (serverNow() - new Date(node.seenAt)) / 1000);
    text = nodeOnline() ? `家里的 Mac 在线（${node.name}），传完的录像会自动处理` : `家里的 Mac 离线（${node.name}），最后在线是 ${fmtDur(ago)}前；上线后自动继续`;
  }
  for (const id of ["nodeLine", "nodeLine2"]) {
    const line = $(id);
    line.textContent = "";
    line.append(el("span", "dot " + (node ? (nodeOnline() ? "on" : "off") : "")), text);
  }
}

const STEP_ICON = { done: "✓", running: "●", pending: "○", skipped: "–", failed: "✕" };

// 步骤列表：每步的状态、用时、进度条和预计时间。每秒重画一次，让时间一直在走。
function renderProgress(v) {
  const box = $("dProgress");
  box.textContent = "";
  const j = v.job || {};
  const p = j.progress;
  if (v.status !== "uploaded" || !p || !Array.isArray(p.steps)) return;
  const live = j.status === "processing";
  const extra = live ? sinceReport(j) : 0;
  const sum = document.createElement("p");
  sum.className = "sum";
  if (live) {
    const eta = p.etaSeconds != null ? Math.max(0, p.etaSeconds - extra) : null;
    sum.textContent = `已用 ${fmtDur(p.elapsedSeconds + extra)}` +
      (eta != null ? ` · 预计还要${fmtAbout(eta)}` : " · 总时间暂时估不出（这台 Mac 多处理几局后会有）");
    if (extra > 90) sum.textContent += ` · Mac 已经 ${fmtDur(extra)}没有消息`;
  } else {
    const total = p.steps.reduce((t, s) => t + (s.seconds || 0), 0);
    sum.textContent = (j.status === "failed" ? "失败前用了 " : "处理用时 ") + fmtDur(total) +
      (p.videoSeconds ? `（录像 ${fmtDur(p.videoSeconds)}）` : "");
  }
  box.append(sum);
  const ul = document.createElement("ul");
  ul.className = "steps";
  for (const s of p.steps) {
    const li = document.createElement("li");
    li.className = s.status;
    const ico = document.createElement("span");
    ico.className = "ico";
    ico.textContent = STEP_ICON[s.status] || "○";
    const lab = document.createElement("span");
    lab.className = "lab";
    lab.textContent = s.label + (s.status === "running" && s.percent != null ? ` ${s.percent}%` : "");
    const time = document.createElement("span");
    time.className = "time";
    if (s.status === "done" || s.status === "failed") time.textContent = fmtDur(s.seconds || 0);
    else if (s.status === "skipped") time.textContent = "跳过";
    else if (s.status === "running") {
      const left = s.etaSeconds != null ? Math.max(0, s.etaSeconds - extra) : null;
      time.textContent = `已 ${fmtDur((s.seconds || 0) + extra)}` + (left != null ? `，${fmtAbout(left).replace("约", "约剩")}` : "");
    } else if (s.estimateSeconds != null) time.textContent = fmtAbout(s.estimateSeconds);
    li.append(ico, lab, time);
    if (s.status === "running" && s.percent != null) {
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("div");
      fill.style.width = s.percent + "%";
      bar.append(fill);
      li.append(bar);
    }
    if (s.detail) {
      const note = document.createElement("span");
      note.className = "note";
      note.textContent = s.detail;
      li.append(note);
    }
    ul.append(li);
  }
  box.append(ul);
}
setInterval(() => {
  if (document.hidden) return;
  const v = openId && videos.find((x) => x.id === openId);
  if (v && busy(v)) renderProgress(v);
}, 1000);


// ---------- 页面切换：#/ 首页，#/upload 上传，#/v/<id> 一段录像，#/settings 设置 ----------

function route() {
  const h = location.hash || "#/";
  const m = h.match(/^#\/v\/([0-9a-f-]{36})$/);
  const lm = h.match(/^#\/label\/([0-9a-f-]{36})$/);
  const sm = h.match(/^#\/strokes\/([0-9a-f-]{36})$/);
  const name = m ? "vDetail" : lm ? "vLabel" : sm ? "vStrokes" : h === "#/upload" ? "vUpload" : h === "#/settings" ? "vSettings" : "vHome";
  for (const v of document.querySelectorAll(".view")) v.hidden = v.id !== name;
  document.body.classList.toggle("labeling", Boolean(lm || sm));
  document.querySelector("main").classList.toggle("wide", Boolean(lm || sm));
  if (m) openDetail(m[1]);
  else closeDetail();
  if (sm) {
    $("sBack").href = "#/v/" + sm[1];
    Strokes.open(sm[1]);
  } else Strokes.close();
  if (lm) {
    $("lBack").href = "#/v/" + lm[1];
    Label.open(lm[1]);
  }
  if (name === "vUpload" && !$("upDone").hidden) {
    // 上一段已经传完：回到空白的上传页，方便接着传下一段
    file = null;
    $("file").value = "";
    $("info").hidden = $("progress").hidden = $("upDone").hidden = true;
  }
  if (name === "vSettings") loadAi();
  if (name === "vHome") refreshList();
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", () => { if (token && !$("app").hidden) route(); });

function openDetail(id) {
  if (openId === id) return;
  openId = id;
  shownKey = "";
  reviewKey = "";
  reviewError = "";
  for (const pane of ["dStats", "dReview", "tVideos", "tCards"]) $(pane).textContent = "";
  $("dRenameBox").hidden = true;
  $("dDelete").textContent = "删除这段录像";
  delete $("dDelete").dataset.armed;
  const v = videos.find((x) => x.id === id);
  // 还在处理或者失败了：先看处理情况；有结果了：先看概览
  setTab(v && v.job && v.job.status === "done" ? "overview" : v ? "process" : "overview");
  if (v) renderDetail();
  else refreshList();
}
function closeDetail() {
  openId = null;
  for (const pane of ["tVideos", "tCards"]) $(pane).textContent = ""; // 停掉正在放的视频
}

const TABS = { overview: "tOverview", videos: "tVideos", cards: "tCards", annot: "tAnnot", process: "tProcess" };
function setTab(name) {
  tab = name;
  for (const b of $("tabs").children) b.classList.toggle("on", b.dataset.tab === name);
  for (const [key, id] of Object.entries(TABS)) $(id).hidden = key !== name;
  if (name !== "videos") for (const v of $("tVideos").querySelectorAll("video")) v.pause();
}
for (const b of $("tabs").children) b.onclick = () => setTab(b.dataset.tab);

// 结果文件放在哪个标签里、叫什么
const RESULT_ORDER = [
  ["videos", /^combined\.mp4$/, () => "羽毛球轨迹 + 骨架"],
  ["videos", /^skeleton\.mp4$/, () => "骨架视频"],
  ["videos", /^rallies\.mp4$/, () => "只有回合的精简版"],
  ["videos", /^longest\.mp4$/, () => "最长的几个回合"],
  ["cards", /^movement\.(.+)\.png$/, (m, names) => "跑动卡片 · " + (names[m[1]] || m[1] + " 号")],
  ["cards", /^positioning\.(near|far)\.png$/, (m) => "站位卡片 · " + (m[1] === "near" ? "近端" : "远端")],
  ["cards", /^players\.(png|jpg)$/, () => "球员编号"],
  ["cards", /^court\.(png|jpg)$/, () => "球场标定（绿线应该压在白线上）"],
  ["cards", /^court\.rejected\.(png|jpg)$/, () => "自动标定找到的最好结果（红线，没有采用）"],
];

async function renderDetail() {
  const v = videos.find((x) => x.id === openId);
  if (!v) { location.hash = "#/"; return; }
  $("dTitle").textContent = titleOf(v);
  const m = v.meta || {};
  $("dMeta").textContent = [
    v.title ? v.filename : "", fmtDate(shotAt(v)), m.duration ? fmtDur(m.duration) : "",
    m.width ? `${m.width}×${m.height}${m.fps ? ` ${Math.round(m.fps)} 帧` : ""}` : "", fmtBytes(v.size),
    v.options && v.options.mode === "doubles" ? "双打" : "单打",
  ].filter(Boolean).join(" · ");
  const st = $("dStatus");
  st.textContent = statusText(v);
  st.className = v.job && v.job.status === "failed" ? "bad" : v.job && v.job.status === "done" ? "ok" : "quiet";
  st.hidden = Boolean(v.job && v.job.status === "done");
  renderProgress(v);
  $("dError").hidden = !(v.job && v.job.error);
  $("dError").textContent = (v.job && v.job.error) || "";
  $("dRetry").hidden = !(v.status === "uploaded" && v.job && ["failed", "done", "archived"].includes(v.job.status));
  // 处理中不能改（Mac 正在跑）；排队中、完成、失败、已清理都可以改
  $("dMode").hidden = !(v.status === "uploaded" && v.job && v.job.status !== "processing");
  $("dMode").textContent = v.options && v.options.mode === "doubles" ? "改成单打重跑" : "改成双打重跑";

  renderReview(v);
  renderAnnot(v);
  const key = v.id + "|" + v.results.filter((r) => !r.name.startsWith("review.")).map((r) => r.name + ":" + r.size).join("|");
  if (key === shownKey) return;
  shownKey = key;
  const have = new Set(v.results.map((r) => r.name));
  const stats = $("dStats");
  stats.textContent = "";
  let names = {};
  if (have.has("movement.json")) {
    try {
      const mv = await (await fetch(resultUrl(v, "movement.json"))).json();
      if (openId !== v.id || key !== shownKey) return;
      stats.append(statsTable(mv));
      for (const p of mv.players || []) if (p.name) names[p.id] = p.name;
    } catch {}
  }
  const panes = { videos: $("tVideos"), cards: $("tCards") };
  for (const pane of Object.values(panes)) pane.textContent = "";
  for (const [where, pattern, title] of RESULT_ORDER) {
    for (const r of v.results) {
      const match = r.name.match(pattern);
      if (!match) continue;
      const h = el("h3", "", title(match, names));
      panes[where].append(h);
      if (r.name.endsWith(".mp4")) {
        const video = document.createElement("video");
        video.controls = true;
        video.playsInline = true;
        video.preload = "metadata";
        video.src = resultUrl(v, r.name);
        panes[where].append(video, saveRow(v, r));
      } else {
        const img = document.createElement("img");
        img.loading = "lazy";
        img.alt = h.textContent;
        img.src = resultUrl(v, r.name);
        panes[where].append(img);
      }
    }
  }
  const waiting = v.job && v.job.status === "archived" ? "云端只保留最近约 10 GB 的结果，这一段的结果已经清理，Mac 上还有。点「处理」里的「重新处理」可以重新生成。"
    : "处理完成后，结果会出现在这里。进度在「处理」里。";
  for (const pane of Object.values(panes)) if (!pane.children.length) pane.append(el("p", "empty", waiting));
  const empty = $("dOverviewEmpty");
  empty.hidden = have.has("movement.json") || have.has("review.md") || have.has("review.request.json");
  empty.textContent = waiting;
}

// 视频下面的「存到相册」「下载」。
// iPhone 上：先把整个视频取到手机上，再点一下打开分享面板，选「存储视频」就进相册
// （分享面板必须由一次点按直接打开，所以分两步）。不支持分享文件的浏览器只给「下载」。
const MB = (n) => (n / 1e6).toFixed(n < 1e7 ? 1 : 0) + " MB";
function canShareVideo() {
  try { return !!(navigator.canShare && navigator.canShare({ files: [new File([""], "x.mp4", { type: "video/mp4" })] })); } catch { return false; }
}
function saveRow(v, r) {
  const row = el("div", "saverow");
  const dl = document.createElement("a");
  dl.className = "plain btnlink";
  dl.href = resultUrl(v, r.name) + "?download=1";
  dl.textContent = "下载" + (r.size ? `（${MB(r.size)}）` : "");
  if (canShareVideo()) {
    const b = document.createElement("button");
    b.className = "btnlink";
    b.textContent = "存到相册";
    const note = el("span", "quiet", "");
    let file = null, busy = false;
    b.onclick = async () => {
      if (file) {
        try { await navigator.share({ files: [file] }); note.textContent = "在弹出的面板里选「存储视频」"; }
        catch (e) { if (e.name !== "AbortError") note.textContent = "没能打开分享：" + e.message + "，可以改用「下载」"; }
        return;
      }
      if (busy) return;
      busy = true;
      try {
        const res = await fetch(resultUrl(v, r.name), { credentials: "include", headers: { authorization: "Bearer " + token } });
        if (!res.ok) throw new Error("HTTP " + res.status);
        const total = Number(res.headers.get("content-length")) || r.size || 0;
        const reader = res.body.getReader(), parts = [];
        let got = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          parts.push(value);
          got += value.length;
          b.textContent = total ? `取视频 ${Math.floor(got * 100 / total)}%` : `取视频 ${MB(got)}`;
        }
        const stem = (v.title || v.filename.replace(/\.[^.]*$/, "")).replace(/[\\/:*?"<>|]+/g, " ").trim() || "录像";
        file = new File(parts, `${stem}_${r.name}`, { type: "video/mp4" });
        b.textContent = "点这里存到相册";
        b.classList.add("ready");
        note.textContent = "视频已取到手机上";
      } catch (e) {
        b.textContent = "存到相册";
        note.textContent = "取视频失败：" + e.message;
      } finally { busy = false; }
    };
    row.append(b, dl, note);
  } else row.append(dl);
  return row;
}

// 改标题
$("dRename").onclick = () => {
  const v = videos.find((x) => x.id === openId);
  if (!v) return;
  $("dRenameBox").hidden = !$("dRenameBox").hidden;
  $("dTitleInput").value = v.title || "";
  $("dTitleInput").placeholder = v.filename;
  if (!$("dRenameBox").hidden) $("dTitleInput").focus();
};
$("dTitleSave").onclick = async () => {
  try {
    await api("/videos/" + openId, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ title: $("dTitleInput").value }) });
    $("dRenameBox").hidden = true;
  } catch (e) { log("改标题失败：" + e.message); }
  refreshList();
};

// ---------- AI 报告 ----------

// 只认标题、列表、引用、粗体、行内代码；全部用 textContent 写入，报告里的 HTML 不会被执行
function mdInline(el, text) {
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/)) {
    if (/^\*\*[^*]+\*\*$/.test(part)) {
      const b = document.createElement("strong");
      b.textContent = part.slice(2, -2);
      el.append(b);
    } else if (/^`[^`]+`$/.test(part)) {
      const c = document.createElement("code");
      c.textContent = part.slice(1, -1);
      el.append(c);
    } else if (part) el.append(part);
  }
  return el;
}

function renderMarkdown(md) {
  const root = document.createElement("div");
  root.className = "md";
  let list = null, para = null, quote = null;
  const flush = () => { list = null; para = null; quote = null; };
  for (const raw of md.split("\n")) {
    const line = raw.trimEnd();
    let m;
    if (!line.trim()) { flush(); continue; }
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) {
      flush();
      root.append(mdInline(document.createElement(m[1].length <= 2 ? "h4" : "h5"), m[2]));
    } else if ((m = line.match(/^\s*(?:[-*]|(\d+)[.、)])\s+(.*)$/))) {
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tagName.toLowerCase() !== tag) {
        flush();
        list = document.createElement(tag);
        if (m[1]) list.start = Number(m[1]);
        root.append(list);
      }
      para = mdInline(document.createElement("li"), m[2]);
      list.append(para);
    } else if ((m = line.match(/^>\s?(.*)$/))) {
      if (!quote) { flush(); quote = document.createElement("blockquote"); root.append(quote); }
      else quote.append(" ");
      mdInline(quote, m[1]);
    } else if (para && /^\s/.test(raw)) {
      para.append(" ");
      mdInline(para, line.trim()); // 列表项的续行
    } else {
      if (!para || list) { flush(); para = document.createElement("p"); root.append(para); }
      else para.append(document.createElement("br"));
      mdInline(para, line.trim());
    }
  }
  return root;
}


let reviewKey = "";    // AI 报告区已经画过的内容
let reviewBusy = "";   // 正在生成报告或回答追问的那一段
let reviewError = "";
let reviewModel = "";  // 这次想用的模型；空着就是默认模型

const fmtTokens = (n) => (n >= 10000 ? (n / 1000).toFixed(1) + "k" : String(n));
function fmtCost(usd) {
  if (usd == null) return "价格未知（没有单价）";
  return usd < 0.01 ? `约 $${usd.toFixed(4)}` : `约 $${usd.toFixed(2)}`;
}
const modelLabel = (id) => ((ai && ai.models.find((m) => m.id === id)) || { label: id }).label;
function usageLine(u) {
  return `${modelLabel(u.model)} · 输入 ${fmtTokens(u.inputTokens)} + 输出 ${fmtTokens(u.outputTokens)} token · ${fmtCost(u.costUsd)}`;
}

function modelSelect() {
  const sel = document.createElement("select");
  for (const m of (ai && ai.models) || []) {
    const o = document.createElement("option");
    o.value = m.id;
    const price = m.priceIn != null && m.priceOut != null ? `（$${m.priceIn} / $${m.priceOut} 每百万 token）` : "（没有单价）";
    o.textContent = m.label + (ai.defaultModel === m.id ? " · 默认" : "") + price;
    sel.append(o);
  }
  sel.value = reviewModel || (ai && ai.defaultModel) || "";
  sel.onchange = () => { reviewModel = sel.value; };
  return sel;
}

async function renderReview(v, force) {
  const box = $("dReview");
  const find = (name) => v.results.find((r) => r.name === name);
  const md = find("review.md"), qa = find("review.qa.md"), can = find("review.request.json");
  const busyNow = reviewBusy === v.id;
  const key = [v.id, md && md.size, qa && qa.size, Boolean(can), busyNow, reviewError, (v.ai || []).length, ai && ai.defaultModel, ai && ai.available].join("|");
  if (!force && key === reviewKey) return;
  reviewKey = key;
  if (!md && !can) { box.textContent = ""; return; }
  const text = async (name) => {
    try { return await (await fetch(`${resultUrl(v, name)}?t=${Date.now()}`)).text(); } catch { return ""; }
  };
  const [report, answers] = await Promise.all([md ? text("review.md") : "", qa ? text("review.qa.md") : ""]);
  if (openId !== v.id || key !== reviewKey) return;
  const draft = box.querySelector("textarea") ? box.querySelector("textarea").value : "";
  box.textContent = "";
  box.append(el("h3", "", "AI 报告"));
  const usage = v.ai || [];
  const lastReport = [...usage].reverse().find((u) => u.kind === "report");
  if (report) {
    box.append(renderMarkdown(report));
    if (lastReport) box.append(el("p", "usage", usageLine(lastReport)));
  }
  if (answers) {
    box.append(el("h3", "", "追问"), renderMarkdown(answers));
    // 这份报告之后的追问用量
    const asks = usage.slice(lastReport ? usage.lastIndexOf(lastReport) + 1 : 0).filter((u) => u.kind === "ask");
    for (const [i, u] of asks.entries()) box.append(el("p", "usage", `第 ${i + 1} 问：${usageLine(u)}`));
  }
  if (usage.length > 1) {
    const total = usage.reduce((t, u) => t + (u.costUsd || 0), 0);
    const unknown = usage.some((u) => u.costUsd == null);
    box.append(el("p", "usage", `这段录像一共调用模型 ${usage.length} 次，合计${fmtCost(total)}${unknown ? "（不含没有单价的）" : ""}`));
  }
  const available = !ai || ai.available;
  const note = el("p", reviewError ? "bad" : "quiet",
    reviewError || (busyNow ? "正在请大模型写，大约需要一分钟，请不要关闭页面……"
      : !available ? "网站还没有配置模型的 API key，暂时出不了 AI 报告。"
      : report ? "" : "这段录像还没有 AI 报告。它只根据上面这些统计数字写，不看视频。"));
  if (note.textContent) box.append(note);
  if (!can || busyNow || !available) return;
  const field = el("div", "field");
  field.append(el("label", "", "用哪个模型"), modelSelect());
  box.append(field);
  const row = el("div", "row");
  const make = el("button", "plain", report ? "重新生成报告" : "生成 AI 报告");
  make.onclick = () => runReview(v.id, "/review", {});
  if (report) {
    const input = document.createElement("textarea");
    input.placeholder = "追问，例如：我下次最该先练哪一项？";
    input.value = draft;
    const ask = el("button", "plain", "追问");
    ask.onclick = () => {
      const question = input.value.trim();
      if (!question) return;
      input.value = "";
      runReview(v.id, "/review/ask", { question });
    };
    box.append(input);
    row.append(ask);
  }
  row.append(make);
  box.append(row);
}

async function runReview(id, path, body) {
  if (reviewBusy) return;
  reviewBusy = id;
  reviewError = "";
  const again = () => { const v = videos.find((x) => x.id === id); if (v && openId === id) renderReview(v); };
  again();
  try {
    await api(`/videos/${id}${path}`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...body, ...(reviewModel ? { model: reviewModel } : {}) }),
    });
  } catch (e) {
    reviewError = "AI 报告没有成功：" + e.message;
    log(reviewError);
  }
  reviewBusy = "";
  await refreshList();
  again();
}

// 概览顶部的几个大数字和球员表
function statsTable(mv) {
  const wrap = document.createElement("div");
  const rallies = (mv.rallies || []).length;
  const kpis = el("div", "kpis");
  for (const [value, label] of [[fmtDur(mv.duration_s || 0), "录像时长"], [String(rallies), "回合"], [fmtDur(mv.in_play_s || 0), "回合内时间"]]) {
    const k = el("div");
    k.append(el("b", "", value), el("span", "", label));
    kpis.append(k);
  }
  wrap.append(kpis);
  const table = document.createElement("table");
  const head = table.insertRow();
  for (const t of ["球员", "跑动", "平均速度", "最快"]) head.append(el("th", "", t));
  for (const pl of mv.players || []) {
    const row = table.insertRow();
    const cells = [
      pl.name || pl.id + " 号",
      pl.distance_m != null ? Math.round(pl.distance_m) + " 米" : "—",
      pl.mean_speed_mps != null ? pl.mean_speed_mps.toFixed(1) + " 米/秒" : "—",
      pl.max_speed_mps != null ? pl.max_speed_mps.toFixed(1) + " 米/秒" : "—",
    ];
    for (const c of cells) row.insertCell().textContent = c;
  }
  wrap.append(table);
  return wrap;
}

// ---------- 设置：默认模型、单价、累计用量 ----------

async function loadAi() {
  try { ai = await api("/ai"); } catch (e) { $("aiState").textContent = "读取失败：" + e.message; return; }
  renderAi();
  const v = openId && videos.find((x) => x.id === openId);
  if (v) renderReview(v);
}

function renderAi() {
  $("aiState").textContent = ai.available ? "每段录像处理完会自动出一份报告。模型只读程序算出的统计数字，不看视频。"
    : "网站还没有配置模型的 API key，暂时出不了 AI 报告。";
  const sel = $("aiDefault");
  sel.textContent = "";
  for (const m of ai.models) {
    const o = document.createElement("option");
    o.value = m.id;
    o.textContent = m.label;
    sel.append(o);
  }
  sel.value = ai.defaultModel;
  const usage = $("aiUsage");
  usage.textContent = "";
  if (!ai.usage.length) usage.textContent = "还没有调用过模型。";
  else {
    const table = document.createElement("table");
    const head = table.insertRow();
    for (const t of ["模型", "次数", "输入", "输出", "估算"]) head.append(el("th", "", t));
    let calls = 0, cost = 0, unknown = false;
    for (const u of ai.usage) {
      const row = table.insertRow();
      for (const c of [modelLabel(u.model), u.calls, fmtTokens(u.inputTokens), fmtTokens(u.outputTokens), u.costUsd != null ? "$" + u.costUsd.toFixed(u.costUsd < 0.1 ? 4 : 2) : "—"]) row.insertCell().textContent = c;
      calls += u.calls;
      cost += u.costUsd || 0;
      unknown = unknown || u.unpricedCalls > 0;
    }
    const sum = table.insertRow();
    for (const c of ["合计", calls, "", "", "$" + cost.toFixed(cost < 0.1 ? 4 : 2)]) sum.insertCell().textContent = c;
    usage.append(table);
    const since = ai.usage.map((u) => u.since).sort()[0];
    usage.append(el("p", "quiet", `从 ${new Date(since).toLocaleDateString("zh-CN")} 起${unknown ? "；有些调用当时没有单价，没算进合计" : ""}`));
  }
  const prices = $("prices");
  prices.textContent = "";
  const head = prices.insertRow();
  for (const t of ["模型", "输入", "输出"]) head.append(el("th", "", t));
  for (const m of ai.models) {
    const row = prices.insertRow();
    row.insertCell().textContent = m.label;
    for (const [field, value] of [["in", m.priceIn], ["out", m.priceOut]]) {
      const input = document.createElement("input");
      input.type = "number";
      input.min = "0";
      input.step = "0.01";
      input.inputMode = "decimal";
      input.placeholder = "未知";
      input.value = value != null ? value : "";
      input.dataset.model = m.id;
      input.dataset.field = field;
      row.insertCell().append(input);
    }
  }
}

async function saveAi(body, done) {
  try {
    ai = await api("/ai/settings", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    renderAi();
    log(done);
  } catch (e) { log("保存设置失败：" + e.message); $("aiState").textContent = "保存失败：" + e.message; }
}
$("aiDefault").onchange = () => saveAi({ defaultModel: $("aiDefault").value }, "默认模型已改成 " + modelLabel($("aiDefault").value));
$("aiSave").onclick = () => {
  const prices = {};
  for (const input of $("prices").querySelectorAll("input")) {
    const p = (prices[input.dataset.model] = prices[input.dataset.model] || {});
    if (input.value !== "") p[input.dataset.field] = Number(input.value);
  }
  $("aiSave").textContent = "已保存";
  setTimeout(() => { $("aiSave").textContent = "保存设置"; }, 1500);
  saveAi({ prices }, "单价已保存");
};

// ---------- 标注球的位置（标注页本身在 label.js） ----------

function renderAnnot(v) {
  const a = v.annot || { status: "none", labels: {} };
  const c = a.labels || {};
  const n = (c.human || 0) + (c.bulk || 0) + (c.skip || 0);
  const ready = v.status === "uploaded" && v.job && ["done", "failed", "archived"].includes(v.job.status);
  const st = $("aState");
  st.className = a.status === "failed" ? "bad" : "quiet";
  st.textContent = {
    none: ready ? "点「准备标注」：家里的 Mac 会挑出模型最没把握的 3 个回合，切成图片传上来，一般几分钟。" : "这段录像分析完之后才能标注。",
    queued: nodeOnline() ? "排队中，Mac 马上开始准备" : "排队中。家里的 Mac 现在离线，上线后开始准备",
    preparing: "Mac 正在准备：" + (a.stage || "开始"),
    ready: "图片准备好了" + (a.stage ? "：" + a.stage : "") + "。",
    failed: "准备失败",
  }[a.status] || "";
  $("aError").hidden = !(a.status === "failed" && a.error);
  $("aError").textContent = a.status === "failed" ? a.error || "" : "";
  const synced = a.labelsChangedAt && a.labelsSyncedAt && a.labelsSyncedAt >= a.labelsChangedAt;
  $("aCounts").textContent = n
    ? `已标 ${n} 帧：亲手确认 ${c.human || 0}，一键接受 ${c.bulk || 0}，看不清 ${c.skip || 0}。` +
      (synced ? "已同步到 Mac 上的 .labels.csv。" : "停手一分钟后会同步到 Mac 上的 .labels.csv。")
    : "";
  $("aOpen").hidden = a.status !== "ready";
  $("aOpen").href = "#/label/" + v.id;
  $("aOpen").textContent = n ? "接着标注" : "开始标注";
  $("aPrepare").hidden = !(ready && (a.status === "none" || a.status === "failed"));
  $("aPrepare").textContent = a.status === "failed" ? "重新准备" : "准备标注";
  $("aFresh").hidden = !(ready && a.status === "ready");
  $("aClear").hidden = !(a.status === "ready" || a.status === "failed");
  $("aClear").textContent = "删除图片";
  delete $("aClear").dataset.armed;

  const k = v.strokes || { status: "none", labels: 0 };
  const ks = $("kState");
  ks.className = k.status === "failed" ? "bad" : "quiet";
  ks.textContent = {
    none: ready ? "点「准备击球类型标注」：家里的 Mac 会识别每一拍的类型，再把每拍剪成一小段传上来。" : "这段录像分析完之后才能标注。",
    queued: nodeOnline() ? "排队中，Mac 马上开始准备" : "排队中。家里的 Mac 现在离线，上线后开始准备",
    preparing: "Mac 正在准备：" + (k.stage || "开始"),
    ready: "小视频准备好了" + (k.stage ? "：" + k.stage : "") + "。",
    failed: "准备失败",
  }[k.status] || "";
  $("kError").hidden = !(k.status === "failed" && k.error);
  $("kError").textContent = k.status === "failed" ? k.error || "" : "";
  const ksynced = k.labelsChangedAt && k.labelsSyncedAt && k.labelsSyncedAt >= k.labelsChangedAt;
  $("kCounts").textContent = k.labels ? `已标 ${k.labels} 拍。` + (ksynced ? "已同步到 Mac 上的 .strokes.labels.csv。" : "停手一分钟后会同步到 Mac 上的 .strokes.labels.csv。") : "";
  $("kOpen").hidden = k.status !== "ready";
  $("kOpen").href = "#/strokes/" + v.id;
  $("kOpen").textContent = k.labels ? "接着标注击球类型" : "开始标注击球类型";
  $("kPrepare").hidden = !(ready && (k.status === "none" || k.status === "failed"));
  $("kPrepare").textContent = k.status === "failed" ? "重新准备" : "准备击球类型标注";
  $("kCsv").hidden = !k.labels;
  $("kCsv").href = `${API}/videos/${v.id}/stroke-labels?format=csv`;
  $("kClear").hidden = !(k.status === "ready" || k.status === "failed");
  $("kClear").textContent = "删除小视频";
  delete $("kClear").dataset.armed;
}

async function annotAction(path, method, body, done) {
  try {
    await api(`/videos/${openId}${path}`, { method, headers: { "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
    log(done);
  } catch (e) {
    log("标注：" + e.message);
    const box = $(path.startsWith("/strokes") ? "kError" : "aError");
    box.hidden = false;
    box.textContent = e.message;
    return;
  }
  refreshList();
}
$("aPrepare").onclick = () => annotAction("/annot", "POST", {}, "已请 Mac 准备标注材料");
// 微调换了权重后：同样这些回合，用新模型重新预测（标注保留），在标注页对比准确率
$("aFresh").onclick = () => annotAction("/annot", "POST", { fresh: true }, "已请 Mac 用新模型重新预测");
$("kPrepare").onclick = () => annotAction("/strokes", "POST", {}, "已请 Mac 准备击球类型标注");
$("kClear").onclick = () => {
  const btn = $("kClear");
  if (btn.dataset.armed !== "1") { btn.dataset.armed = "1"; btn.textContent = "再点一次：删除小视频（标注保留）"; return; }
  annotAction("/strokes", "DELETE", null, "击球小视频已删除，标注保留");
};
$("aClear").onclick = () => {
  const btn = $("aClear");
  if (btn.dataset.armed !== "1") { btn.dataset.armed = "1"; btn.textContent = "再点一次：删除图片（标注保留）"; return; }
  annotAction("/annot", "DELETE", null, "标注用的图片已删除，标注保留");
};

// ---------- 处理：重新处理、改单打双打、删除 ----------

$("dRetry").onclick = async () => {
  try {
    await api(`/videos/${openId}/job/retry`, { method: "POST" });
    log("已重新排队");
  } catch (e) { log("重新处理失败：" + e.message); }
  refreshList();
};
$("dMode").onclick = async () => {
  const v = videos.find((x) => x.id === openId);
  if (!v) return;
  const mode = v.options && v.options.mode === "doubles" ? "singles" : "doubles";
  try {
    await api(`/videos/${openId}/job/retry`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ options: { ...v.options, mode } }),
    });
    log(`已改成${mode === "doubles" ? "双打" : "单打"}并重新排队`);
  } catch (e) { log("重新处理失败：" + e.message); }
  refreshList();
};
$("dDelete").onclick = async () => {
  const btn = $("dDelete");
  if (btn.dataset.armed !== "1") { btn.dataset.armed = "1"; btn.textContent = "再点一次确认删除"; return; }
  const v = videos.find((x) => x.id === openId);
  await api("/videos/" + openId, { method: "DELETE" }).catch((e) => log("删除失败：" + e.message));
  if (v) log("已从云端删除 " + v.filename);
  location.hash = "#/";
};
document.addEventListener("visibilitychange", () => { if (!document.hidden && token && !$("app").hidden) refreshList(); });

$("copyLog").onclick = async () => {
  try { await navigator.clipboard.writeText($("log").textContent); log("记录已复制"); }
  catch { log("复制失败，请手动选中"); }
};

tryLogin();
