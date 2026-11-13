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
    refreshList();
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
        filename: file.name, size: file.size, contentType: file.type, meta,
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
    $("pTitle").textContent = "上传完成";
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
let openId = null;     // 正在看哪一局
let shownKey = "";     // 结果区已经画过的内容，没变化就不重画（不然视频会被打断）
let pollTimer = null;
let node = null;       // Mac 处理节点：{ name, seenAt }
let clockSkew = 0;     // 服务器时间 - 手机时间（毫秒），时间差都按服务器时间算
const serverNow = () => Date.now() + clockSkew;
const NODE_ONLINE_MS = 90 * 1000; // 节点空闲时每 15 秒来问一次任务，处理中每 20 秒汇报一次

// 节点上报之后又过了多久（秒）：用来让“已用时间”和“预计剩余”在两次上报之间继续走
const sinceReport = (j) => (j && j.updatedAt ? Math.max(0, (serverNow() - new Date(j.updatedAt)) / 1000) : 0);
const fmtAbout = (s) => (s < 50 ? `约 ${Math.max(5, Math.round(s / 5) * 5)} 秒` : `约 ${Math.max(1, Math.round(s / 60))} 分钟`);
const nodeOnline = () => node && serverNow() - new Date(node.seenAt) < NODE_ONLINE_MS;

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

async function refreshList() {
  const ul = $("list");
  try {
    const data = await api("/videos");
    videos = data.videos;
    node = data.node;
    if (data.now) clockSkew = new Date(data.now) - Date.now();
  } catch (e) {
    ul.innerHTML = '<li class="quiet">读取失败</li>';
    return;
  }
  ul.textContent = "";
  if (!videos.length) ul.innerHTML = '<li class="quiet">还没有录像</li>';
  for (const v of videos) {
    const li = document.createElement("li");
    li.className = "video";
    const left = document.createElement("span");
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = v.filename;
    const sub = document.createElement("div");
    sub.className = "quiet";
    sub.textContent = `${new Date(v.createdAt).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })} · ${fmtBytes(v.size)} · ${statusText(v)}`;
    if (v.job && v.job.status === "failed") sub.classList.add("bad");
    if (v.job && v.job.status === "done") sub.classList.add("ok");
    left.append(name, sub);
    const chev = document.createElement("span");
    chev.className = "chev";
    chev.textContent = "›";
    li.append(left, chev);
    li.onclick = () => openDetail(v.id);
    ul.append(li);
  }
  renderNodeLine();
  if (openId) renderDetail();
  clearTimeout(pollTimer);
  if (videos.some(busy)) pollTimer = setTimeout(refreshList, 3000);
}

function renderNodeLine() {
  const el = $("nodeLine");
  el.textContent = "";
  const dot = document.createElement("span");
  dot.className = "dot " + (node ? (nodeOnline() ? "on" : "off") : "");
  let text = "Mac 上的处理程序还没连接过";
  if (node) {
    const ago = Math.max(0, (serverNow() - new Date(node.seenAt)) / 1000);
    text = nodeOnline() ? `Mac 在线（${node.name}）` : `Mac 离线（${node.name}），最后在线是 ${fmtDur(ago)}前`;
  }
  el.append(dot, text);
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

function openDetail(id) {
  openId = id;
  shownKey = "";
  $("dBody").textContent = "";
  $("detail").hidden = false;
  $("listCard").hidden = true;
  $("dDelete").textContent = "删除这局";
  delete $("dDelete").dataset.armed;
  renderDetail();
  $("detail").scrollIntoView({ block: "start" });
}
function closeDetail() {
  openId = null;
  $("dBody").textContent = ""; // 停掉正在放的视频
  $("detail").hidden = true;
  $("listCard").hidden = false;
}
$("dBack").onclick = closeDetail;

const RESULT_ORDER = [
  [/^movement\.(.+)\.png$/, (m, names) => "跑动卡片 · " + (names[m[1]] || m[1] + " 号")],
  [/^positioning\.(near|far)\.png$/, (m) => "站位卡片 · " + (m[1] === "near" ? "近端" : "远端")],
  [/^combined\.mp4$/, () => "羽毛球轨迹 + 骨架视频"],
  [/^longest\.mp4$/, () => "最长的几个回合"],
  [/^rallies\.mp4$/, () => "只有回合的精简视频"],
  [/^skeleton\.mp4$/, () => "骨架视频"],
  [/^players\.(png|jpg)$/, () => "球员编号"],
  [/^court\.(png|jpg)$/, () => "球场标定（绿线应该压在白线上）"],
  [/^court\.rejected\.(png|jpg)$/, () => "自动标定找到的最好结果（红线，没有采用）"],
];

async function renderDetail() {
  const v = videos.find((x) => x.id === openId);
  if (!v) return closeDetail();
  $("dTitle").textContent = v.filename;
  const st = $("dStatus");
  st.textContent = statusText(v) + (v.options && v.options.mode === "doubles" ? " · 双打" : " · 单打");
  st.className = v.job && v.job.status === "failed" ? "bad" : v.job && v.job.status === "done" ? "ok" : "quiet";
  renderProgress(v);
  $("dError").hidden = !(v.job && v.job.error);
  $("dError").textContent = (v.job && v.job.error) || "";
  $("dRetry").hidden = !(v.status === "uploaded" && v.job && ["failed", "done", "archived"].includes(v.job.status));
  // 处理中不能改（Mac 正在跑）；排队中、完成、失败、已清理都可以改
  $("dMode").hidden = !(v.status === "uploaded" && v.job && v.job.status !== "processing");
  $("dMode").textContent = v.options && v.options.mode === "doubles" ? "改成单打重跑" : "改成双打重跑";

  const key = v.results.map((r) => r.name + ":" + r.size).join("|");
  if (key === shownKey) return;
  shownKey = key;
  const body = $("dBody");
  body.textContent = "";
  const url = (name) => `${API}/videos/${v.id}/results/${encodeURIComponent(name)}`;
  const have = new Set(v.results.map((r) => r.name));

  let names = {};
  if (have.has("movement.json")) {
    try {
      const mv = await (await fetch(url("movement.json"))).json();
      if (openId !== v.id) return;
      body.append(statsTable(mv));
      for (const p of mv.players || []) if (p.name) names[p.id] = p.name;
    } catch {}
  }
  for (const [pattern, title] of RESULT_ORDER) {
    for (const r of v.results) {
      const m = r.name.match(pattern);
      if (!m) continue;
      const h = document.createElement("h3");
      h.textContent = title(m, names);
      body.append(h);
      if (r.name.endsWith(".mp4")) {
        const el = document.createElement("video");
        el.controls = true;
        el.playsInline = true;
        el.preload = "metadata";
        el.src = url(r.name);
        body.append(el);
      } else {
        const el = document.createElement("img");
        el.loading = "lazy";
        el.alt = h.textContent;
        el.src = url(r.name);
        body.append(el);
      }
    }
  }
}

function statsTable(mv) {
  const wrap = document.createElement("div");
  const p = document.createElement("p");
  p.className = "quiet";
  const rallies = (mv.rallies || []).length;
  p.textContent = `共 ${fmtDur(mv.duration_s || 0)}，${rallies} 个回合，回合内 ${fmtDur(mv.in_play_s || 0)}`;
  wrap.append(p);
  const table = document.createElement("table");
  const head = table.insertRow();
  for (const t of ["球员", "跑动", "平均速度", "最快"]) {
    const th = document.createElement("th");
    th.textContent = t;
    head.append(th);
  }
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
  closeDetail();
  refreshList();
};
document.addEventListener("visibilitychange", () => { if (!document.hidden && token && !$("app").hidden) refreshList(); });

$("copyLog").onclick = async () => {
  try { await navigator.clipboard.writeText($("log").textContent); log("记录已复制"); }
  catch { log("复制失败，请手动选中"); }
};

tryLogin();
