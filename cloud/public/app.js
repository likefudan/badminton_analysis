// 手机网页：把录像切片传到 /api/v1（可断点续传），看每一局的处理状态和分析结果。
"use strict";

const API = "/api/v1";
const CONCURRENCY = 3;
const MAX_TRIES = 8;          // 服务器明确报错时最多试几次；网络问题会一直重试
const STALL_MS = 30 * 1000;   // 一片这么久没有任何进展就算卡住，断开重传
const HIDDEN_KICK_MS = 3000;  // 离开页面超过这么久，回来后把在途的请求断开重传
const $ = (id) => document.getElementById(id);

let token = "";   // 用口令登录时的口令；用 Google 登录时是空的，靠 cookie
try { token = localStorage.getItem("ba.token") || ""; } catch {}
let me = null;    // 登录后 /me 的回答：email、name、role（admin | annotator）、status
let authed = false;
const isAdmin = () => Boolean(me && me.role === "admin" && me.status === "active");
// 超级管理员：只有它有「分析」这一边（上传、看结果、设置、Mac 状态），也只有它能让 Mac 准备标注材料
const isOwner = () => Boolean(isAdmin() && me.owner);
// 每个请求都带 x-ba（服务器靠它确认请求是本站的网页发的）；有口令再带口令
const authHeaders = () => (token ? { "x-ba": "1", authorization: "Bearer " + token } : { "x-ba": "1" });

// 这一批要传的录像，按这个顺序一个一个传。每一项：
// { file, key, meta, probed, state: wait | up | paused | error | done, sent, video, error, row... }
let queue = [];
let queueBusy = false; // 正在按顺序上传
let run = null; // 正在传的那一个：{ item, video, controller, paused, ... }
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
    headers: { ...authHeaders(), ...(options.headers || {}) },
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

const LOGIN_ERRORS = {
  not_configured: "网站还没有配置 Google 登录",
  cancelled: "没有完成 Google 登录",
  state: "登录超时了，再试一次",
  google: "Google 登录失败，再试一次",
  full: "申请加入的人太多了，请联系管理员",
};

async function tryLogin() {
  authed = false;
  let cfg = { google: false };
  try { cfg = await (await fetch(API + "/auth/config")).json(); } catch {}
  $("googleBtn").hidden = !cfg.google;
  $("pwBox").hidden = cfg.google;  // 有 Google 登录就不再显示口令登录
  $("pwBox").open = !cfg.google;
  $("claimBox").hidden = !cfg.claim;
  const code = new URLSearchParams(location.search).get("login_error");
  if (code) history.replaceState(null, "", location.pathname + location.hash);
  try {
    try { me = await api("/me"); }
    catch (e) {
      // 存着的口令失效了，但可能已经用 Google 登录过
      if (e.status !== 401 || !token) throw e;
      const old = token;
      token = "";
      try { me = await api("/me"); } catch { token = old; throw e; }
      try { localStorage.removeItem("ba.token"); } catch {}
    }
    if (token) await api("/session", { method: "POST" }); // 用口令登录时，图片和视频靠 cookie 读取
  } catch (e) {
    return showLogin(e.status === 401 ? (token ? "口令不对" : LOGIN_ERRORS[code] || "") : "连不上服务器：" + e.message);
  }
  $("login").hidden = true;
  if (me.status !== "active") {
    $("pending").hidden = false;
    $("app").hidden = true;
    $("pendingWho").textContent = me.email;
    $("pendingMsg").textContent = me.status === "disabled" ? "这个账号已经被管理员停用。" : "这个账号还没有被邀请。管理员已经能在「成员」页看到你的申请，放行后刷新这个页面就能进。";
    return;
  }
  authed = true;
  $("pending").hidden = true;
  $("app").hidden = false;
  document.body.classList.toggle("annotator", !isAdmin());
  document.body.classList.toggle("owner", isOwner());
  Team.setUser();
  route();
  if (isOwner()) { refreshList(); loadAi(); }
}
function showLogin(msg) {
  authed = false;
  $("login").hidden = false;
  $("app").hidden = $("pending").hidden = true;
  $("loginMsg").textContent = msg;
}
$("loginBtn").onclick = () => {
  token = $("token").value.trim();
  try { localStorage.setItem("ba.token", token); } catch {}
  tryLogin();
};
async function logout() {
  token = "";
  me = null;
  try { localStorage.removeItem("ba.token"); } catch {}
  try { await fetch(API + "/auth/logout", { method: "POST", headers: { "x-ba": "1" } }); } catch {}
  showLogin("");
}
for (const id of ["logout", "pendingLogout", "navLogout"]) $(id).onclick = logout;
// 第一个管理员：用 Google 登录后输一次口令
$("claimBtn").onclick = async () => {
  try {
    await api("/auth/claim-admin", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ password: $("claimPw").value.trim() }) });
    $("claimPw").value = "";
    tryLogin();
  } catch (e) { $("claimMsg").textContent = e.message; }
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

// ---------- 这一批要传的录像 ----------

let probeChain = Promise.resolve(); // 读参数要播放一小段，一个一个来

async function probeItem(item) {
  if (!queue.includes(item)) return;
  const f = item.file;
  const p = await probeVideo(f);
  const perMin = p.duration > 0 ? f.size / (p.duration / 60) : NaN;
  item.meta = {
    width: p.width, height: p.height,
    duration: isFinite(p.duration) ? Math.round(p.duration * 100) / 100 : null,
    fps: isFinite(p.fps) ? Math.round(p.fps * 10) / 10 : null,
    lastModified: f.lastModified,
    userAgent: navigator.userAgent,
  };
  const is4k = Math.max(p.width, p.height) >= 3840;
  const is60 = isFinite(p.fps) && p.fps > 50;
  const parts = [fmtBytes(f.size)];
  if (p.width) parts.push(`${p.width}×${p.height}`);
  if (isFinite(p.fps)) parts.push(`${p.fps.toFixed(0)} fps`);
  if (isFinite(p.duration)) parts.push(fmtDur(p.duration));
  if (isFinite(perMin)) parts.push(fmtBytes(perMin) + "/分钟");
  item.info = parts.join(" · ");
  if (!p.width) item.verdict = ["warn", "浏览器读不出这个视频的参数，仍然可以上传。"];
  else if (is4k && is60) item.verdict = ["ok", "4K60 原片"];
  else if (is4k && !isFinite(p.fps)) item.verdict = ["warn", "是 4K，帧率读不出。4K60 大约 400 MB/分钟。"];
  else item.verdict = ["bad", "不是 4K60，可能在选择时被系统压缩了。分析需要原片，建议换一种方式重新选。"];
  log(`${f.name}：${p.width}×${p.height}，约 ${isFinite(p.fps) ? p.fps.toFixed(1) : "?"} fps，${fmtDur(p.duration)}，${isFinite(perMin) ? fmtBytes(perMin) : "?"}/分钟`);
  updateRow(item);
}

const movable = (item) => item && item.state !== "up" && item.state !== "done";

function buildRow(item) {
  const row = el("li", "qitem");
  item.row = row;
  item.numEl = el("span", "qnum");
  const main = el("div", "qmain");
  item.titleEl = document.createElement("input");
  item.titleEl.type = "text";
  item.titleEl.maxLength = 80;
  item.titleEl.placeholder = "标题（可以不填）";
  item.titleEl.setAttribute("aria-label", "标题");
  item.infoEl = el("div", "qinfo");
  item.verdictEl = el("div", "qverdict");
  item.statusEl = el("div", "qstatus");
  const bar = el("div", "bar");
  item.barEl = el("div");
  bar.append(item.barEl);
  main.append(el("div", "qname", item.file.name), item.titleEl, item.infoEl, item.verdictEl, bar, item.statusEl);
  const tools = el("div", "qtools");
  const btn = (text, label, fn) => {
    const b = el("button", "plain", text);
    b.type = "button";
    b.title = label;
    b.setAttribute("aria-label", label);
    b.onclick = fn;
    tools.append(b);
    return b;
  };
  item.upBtn = btn("↑", "往前挪", () => moveItem(item, -1));
  item.downBtn = btn("↓", "往后挪", () => moveItem(item, 1));
  item.delBtn = btn("✕", "不传这个", () => removeItem(item));
  row.append(item.numEl, main, tools);
}

function updateRow(item) {
  if (!item.row) return;
  const i = queue.indexOf(item);
  item.numEl.textContent = item.state === "done" ? "✓" : String(i + 1);
  item.row.dataset.state = item.state;
  item.infoEl.textContent = item.info || `${fmtBytes(item.file.size)} · 读取参数中`;
  item.verdictEl.className = "qverdict " + (item.verdict ? item.verdict[0] : "");
  item.verdictEl.textContent = item.verdict ? item.verdict[1] : "";
  item.titleEl.disabled = Boolean(item.video); // 开始传之后标题在录像页里改
  item.titleEl.hidden = Boolean(item.video) && !item.titleEl.value;
  const pct = item.file.size ? Math.min(100, item.sent / item.file.size * 100) : 0;
  item.barEl.style.width = (item.state === "done" ? 100 : pct).toFixed(1) + "%";
  const st = item.statusEl;
  st.className = "qstatus " + ({ done: "ok", error: "bad", paused: "warn" }[item.state] || "");
  if (item.state === "done") {
    st.textContent = "第 " + (i + 1) + " 个，已传完，已经排队等 Mac 处理 ";
    if (item.video) {
      const a = el("a", "", "查看");
      a.href = "#/v/" + item.video.id;
      st.append(a);
    }
  } else if (item.state === "up") st.textContent = item.merging ? "正在合并" : `上传中 ${pct.toFixed(1)}%`;
  else if (item.state === "paused") st.textContent = `已暂停，传了 ${pct.toFixed(0)}%`;
  else if (item.state === "error") st.textContent = "出错：" + item.error;
  else st.textContent = `第 ${i + 1} 个，等待上传`;
  item.upBtn.disabled = !(movable(item) && movable(queue[i - 1]));
  item.downBtn.disabled = !(movable(item) && movable(queue[i + 1]));
  item.delBtn.hidden = item.state === "up" || item.state === "done";
  item.upBtn.hidden = item.downBtn.hidden = item.state === "done";
}

function renderQueue() {
  const list = $("queue");
  const rows = queue.map((x) => x.row);
  // 顺序没变就不动节点，不然正在填的标题会失去焦点
  if (rows.length !== list.children.length || rows.some((r, i) => list.children[i] !== r)) list.replaceChildren(...rows);
  queue.forEach(updateRow);
  $("info").hidden = !queue.length;
  const left = queue.filter((x) => x.state !== "done").length;
  $("qTitle").textContent = `要上传的录像（${queue.length} 个${left && left !== queue.length ? "，还剩 " + left + " 个" : ""}）`;
  $("qHint").hidden = queue.length < 2;
  if (!queueBusy) {
    $("startBtn").hidden = !left;
    $("startBtn").textContent = queue.some((x) => x.state === "paused" || x.state === "error" || x.state === "done" || x.resumable) ? "继续上传" : left > 1 ? `按顺序上传 ${left} 个` : "开始上传";
  }
}

function moveItem(item, d) {
  const i = queue.indexOf(item), j = i + d;
  if (i < 0 || !movable(item) || !movable(queue[j])) return;
  [queue[i], queue[j]] = [queue[j], queue[i]];
  renderQueue();
}

function removeItem(item) {
  if (item.state === "up" || item.state === "done") return;
  queue = queue.filter((x) => x !== item);
  renderQueue();
  if (!queue.length) $("progress").hidden = true;
}

function addFiles(files) {
  // 上一批已经全部传完：新选的另起一批
  if (!queueBusy && queue.length && queue.every((x) => x.state === "done")) resetQueue();
  let added = 0;
  for (const f of files) {
    const key = fileKey(f);
    if (queue.some((x) => x.key === key)) { log(`${f.name} 已经在列表里`); continue; }
    const item = { file: f, key, meta: {}, state: "wait", sent: 0, video: null, error: "", info: "", verdict: null, merging: false, resumable: false };
    try { item.resumable = Boolean(localStorage.getItem(key)); } catch {}
    buildRow(item);
    queue.push(item);
    added++;
    log(`选了 ${f.name}，${fmtBytes(f.size)}，类型 ${f.type || "未知"}`);
    item.probed = probeChain = probeChain.then(() => probeItem(item)).catch(() => {});
  }
  if (added) $("upDone").hidden = true;
  renderQueue();
  // 正在传的时候加进来的，排在后面接着传
}

function resetQueue() {
  queue = [];
  $("file").value = "";
  $("info").hidden = $("progress").hidden = $("upDone").hidden = true;
  $("queue").replaceChildren();
}

$("file").onchange = (e) => {
  // 手机相册多选时，文件按点选的先后排列
  const files = Array.from(e.target.files || []);
  e.target.value = ""; // 之后还能再选同一个文件
  if (files.length) addFiles(files);
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

function putPart(file, video, n, signal, onProgress) {
  // 用 XMLHttpRequest 而不是 fetch，是为了拿到上传进度
  return new Promise((resolve, reject) => {
    const start = (n - 1) * video.partSize;
    const blob = file.slice(start, Math.min(start + video.partSize, file.size));
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `${API}/videos/${video.id}/parts/${n}`);
    for (const [k, v] of Object.entries(authHeaders())) xhr.setRequestHeader(k, v);
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

// 传一个文件。传完返回 true，被暂停返回 false，出错就抛出。
async function uploadItem(item) {
  const file = item.file;
  const key = item.key;
  await item.probed; // 新建上传时要带上分辨率、帧率这些参数
  let video = null;
  let savedId = null;
  try { savedId = localStorage.getItem(key); } catch {}
  if (savedId) {
    try {
      video = await api("/videos/" + savedId);
      if (video.size !== file.size) video = null;
    } catch (e) {
      if (e.status !== 404) throw new Error("读不到上次的进度：" + e.message);
    }
  }
  if (video && video.status === "uploaded") {
    log(`${file.name} 之前已经传完了`);
    try { localStorage.removeItem(key); } catch {}
    item.video = video;
    item.sent = file.size;
    return true;
  }
  if (!video) {
    video = await api("/videos", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        filename: file.name, size: file.size, contentType: file.type, meta: item.meta, title: item.titleEl.value.trim(),
        options: { mode: document.querySelector('input[name="mode"]:checked').value },
      }),
    });
    try { localStorage.setItem(key, video.id); } catch {}
    log(`${file.name}：新建上传，共 ${video.partCount} 片，每片 ${fmtBytes(video.partSize)}`);
  } else {
    log(`${file.name}：继续上传，已有 ${video.uploadedParts.length}/${video.partCount} 片`);
  }
  item.video = video;

  const done = new Set(video.uploadedParts);
  const todo = [];
  for (let n = 1; n <= video.partCount; n++) if (!done.has(n)) todo.push(n);

  const controller = new AbortController();
  const r = run = {
    item, video, controller, paused: false, finished: false,
    confirmed: video.uploadedBytes, // 服务器已确认的字节
    inflight: new Map(),            // 正在传的片 -> 已发送字节
    xhrs: new Set(),                // 在途的请求，卡住时用来断开
    sentThisRun: 0, startedAt: Date.now(), retries: 0, samples: [],
  };
  if (pauseAsked) { r.paused = true; controller.abort(); } // 在新建上传的那一下点了暂停
  $("pRetries").textContent = "0";
  $("progress").hidden = false;
  const timer = setInterval(render, 500);
  render();

  async function worker() {
    while (todo.length && !controller.signal.aborted) {
      const n = todo.shift();
      for (let attempt = 1, httpFails = 0; ; attempt++) {
        try {
          r.inflight.set(n, 0);
          const size = await putPart(file, video, n, controller.signal, (loaded) => r.inflight.set(n, loaded));
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
    if (controller.signal.aborted) return false;
    item.merging = true;
    render();
    const result = await api(`/videos/${video.id}/complete`, { method: "POST" });
    const secs = (Date.now() - r.startedAt) / 1000;
    try { localStorage.removeItem(key); } catch {}
    item.sent = file.size;
    log(`${file.name} 传完：${fmtBytes(result.size)}，用时 ${fmtDur(secs)}，平均 ${fmtSpeed(r.sentThisRun / Math.max(secs, 1))}，重试 ${r.retries} 次。已经排队等 Mac 处理`);
    return true;
  } catch (e) {
    controller.abort();
    throw e;
  } finally {
    clearInterval(timer);
    item.merging = false;
    r.finished = true;
    render();
  }
}

let pauseAsked = false;

// 按列表顺序一个一个传；传完一个它就进了处理队列，Mac 不用等后面的
async function runQueue() {
  if (queueBusy) return;
  queueBusy = true;
  pauseAsked = false;
  hiddenCount = 0; hiddenTotalMs = 0; hiddenSince = 0;
  $("pHidden").textContent = "0 次";
  $("startBtn").hidden = true;
  $("pauseBtn").hidden = false;
  $("upDone").hidden = true;
  requestWakeLock();
  let stopped = "";
  try {
    for (;;) {
      const item = queue.find((x) => x.state !== "done");
      if (!item || pauseAsked) { if (item) stopped = "paused"; break; }
      item.state = "up";
      item.error = "";
      renderQueue();
      $("progress").hidden = false;
      $("pTitle").textContent = `正在传第 ${queue.indexOf(item) + 1} 个，共 ${queue.length} 个`;
      try {
        if (await uploadItem(item)) {
          item.state = "done";
          refreshList();
        } else {
          item.state = "paused";
          stopped = "paused";
          log("已暂停");
        }
      } catch (e) {
        item.state = "error";
        item.error = e.message;
        stopped = "error";
        log(`${item.file.name} 上传出错：${e.message}。可以点“继续上传”，或者去掉这一个。`);
      }
      renderQueue();
      if (stopped) break;
    }
  } finally {
    queueBusy = false;
    if (wakeLock) wakeLock.release().catch(() => {});
    $("pauseBtn").hidden = true;
    const n = queue.length;
    if (stopped) $("pTitle").textContent = stopped === "paused" ? "已暂停" : "上传出错";
    else if (n) {
      $("pTitle").textContent = n > 1 ? `${n} 个都传完了，已经按顺序排队等 Mac 处理` : "上传完成，已经排队等 Mac 处理";
      $("upDone").hidden = false;
    }
    renderQueue();
  }
}

function render() {
  if (!run) return;
  const r = run;
  let flying = 0;
  for (const v of r.inflight.values()) flying += v;
  const sent = Math.min(r.video.size, r.confirmed + flying);
  r.item.sent = sent;
  updateRow(r.item);
  const now = Date.now();
  r.samples.push([now, r.sentThisRun + flying]);
  while (r.samples.length > 2 && now - r.samples[0][0] > 10000) r.samples.shift();
  const [t0, b0] = r.samples[0];
  const speed = now > t0 ? Math.max(0, (r.sentThisRun + flying - b0) / ((now - t0) / 1000)) : 0;
  const elapsed = (now - r.startedAt) / 1000;
  const avg = (r.sentThisRun + flying) / Math.max(elapsed, 1);
  let leftAll = 0;
  for (const x of queue) if (x.state !== "done") leftAll += Math.max(0, x.file.size - x.sent);
  $("pName").textContent = r.item.file.name;
  $("pBar").style.width = (sent / r.video.size * 100).toFixed(1) + "%";
  $("pBytes").textContent = `${fmtBytes(sent)} / ${fmtBytes(r.video.size)}（${(sent / r.video.size * 100).toFixed(1)}%）`;
  $("pSpeed").textContent = fmtSpeed(speed);
  $("pAvg").textContent = fmtSpeed(avg);
  $("pEta").textContent = speed > 0 ? fmtDur((r.video.size - sent) / speed) : "—";
  $("pEtaAll").textContent = speed > 0 ? fmtDur(leftAll / speed) : "—";
  $("pElapsed").textContent = fmtDur(elapsed);
}

$("startBtn").onclick = () => runQueue().catch((e) => log("出错：" + e.message));
$("pauseBtn").onclick = () => {
  pauseAsked = true;
  if (run && !run.finished) { run.paused = true; run.controller.abort(); }
};

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
  if (videos.some(busy) || (run && !run.finished)) pollTimer = setTimeout(refreshList, 3000);
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
    const more = el("a", "", "看它在做什么 ›");
    more.href = "#/mac";
    line.append(el("span", "dot " + (node ? (nodeOnline() ? "on" : "off") : "")), text + " ", more);
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


// ---------- 页面切换：#/ 首页，#/upload 上传，#/v/<id> 一段录像，#/settings 设置，#/mac Mac 状态 ----------

// 分析：#/ 我的录像，#/upload 上传，#/v/<id> 一段录像。标注：#/tasks 任务，#/materials 素材，#/review 审核，
// #/label/<id>、#/strokes/<id> 标注页。账号菜单：#/team 成员，#/settings 设置，#/mac Mac 状态。
function route() {
  let h = location.hash || "#/";
  // 超级管理员什么都能进；别的管理员只有标注这一边和成员；标注员只有任务和标注页
  const labeling = /^#\/(tasks$|label\/|strokes\/)/.test(h);
  const allowed = isOwner() || labeling || (isAdmin() && /^#\/(materials|review|team)$/.test(h));
  if (!allowed) h = "#/tasks";
  const m = h.match(/^#\/v\/([0-9a-f-]{36})$/);
  const lm = h.match(/^#\/label\/([0-9a-f-]{36})(?:\/t(\d+))?$/);
  const sm = h.match(/^#\/strokes\/([0-9a-f-]{36})(?:\/t(\d+))?$/);
  const PAGES = { "#/upload": "vUpload", "#/settings": "vSettings", "#/mac": "vMac", "#/tasks": "vTasks", "#/materials": "vMaterials", "#/review": "vReview", "#/team": "vTeam" };
  const name = m ? "vDetail" : lm ? "vLabel" : sm ? "vStrokes" : PAGES[h] || "vHome";
  for (const v of document.querySelectorAll(".view")) v.hidden = v.id !== name;
  $("navAnalysis").classList.toggle("on", ["vHome", "vUpload", "vDetail"].includes(name));
  $("navLabeling").classList.toggle("on", ["vTasks", "vMaterials", "vReview"].includes(name));
  $("navHome").href = isOwner() ? "#/" : "#/tasks";
  $("navMenu").open = false;
  document.body.classList.toggle("labeling", Boolean(lm || sm));
  document.querySelector("main").classList.toggle("wide", Boolean(lm || sm));
  if (m) openDetail(m[1]);
  else closeDetail();
  // 带任务号（#/label/<录像>/t<任务>）：只标这个任务的范围，标完可以提交；不带任务号是从「素材」里整段打开
  const taskId = lm && lm[2] ? Number(lm[2]) : sm && sm[2] ? Number(sm[2]) : null;
  const back = taskId ? (Team.cameFrom() === "review" ? "#/review" : "#/tasks") : "#/materials";
  Team.showBar(null);
  if (sm) {
    $("sBack").href = back;
    if (taskId) Team.task(taskId).then((t) => { if (location.hash === h) { Strokes.open(sm[1], t); Team.showBar(t); } }).catch((e) => { $("sMsg").textContent = e.message; });
    else Strokes.open(sm[1]);
  } else Strokes.close();
  if (lm) {
    $("lBack").href = back;
    if (taskId) Team.task(taskId).then((t) => { if (location.hash === h) { Label.open(lm[1], t); Team.showBar(t); } }).catch((e) => { $("lMsg").textContent = e.message; });
    else Label.open(lm[1]);
  }
  Team.show(name === "vTasks" ? "tasks" : name === "vReview" ? "review" : name === "vMaterials" ? "materials" : null);
  if (name === "vTeam") Team.openTeam();
  // 上一批已经传完：回到空白的上传页，方便接着传下一批
  if (name === "vUpload" && !$("upDone").hidden) resetQueue();
  if (name === "vSettings") loadAi();
  if (name === "vMac") Mac.open();
  else Mac.close();
  if (name === "vHome") refreshList();
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", () => { if (authed && !$("app").hidden) route(); });

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

const TABS = { overview: "tOverview", videos: "tVideos", cards: "tCards", process: "tProcess" };
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
    v.options && v.options.region ? "有分析区域" : "",
  ].filter(Boolean).join(" · ");
  const st = $("dStatus");
  st.textContent = statusText(v);
  st.className = v.job && v.job.status === "failed" ? "bad" : v.job && v.job.status === "done" ? "ok" : "quiet";
  st.hidden = Boolean(v.job && v.job.status === "done");
  renderProgress(v);
  if (typeof renderRegion === "function") renderRegion(v); // region.js
  $("dError").hidden = !(v.job && v.job.error);
  $("dError").textContent = (v.job && v.job.error) || "";
  $("dRetry").hidden = !(v.status === "uploaded" && v.job && ["failed", "done", "archived"].includes(v.job.status));
  $("dCards").hidden = !(v.status === "uploaded" && v.job && v.job.status === "done");
  // 处理中不能改（Mac 正在跑）；排队中、完成、失败、已清理都可以改
  $("dMode").hidden = !(v.status === "uploaded" && v.job && v.job.status !== "processing");
  $("dMode").textContent = v.options && v.options.mode === "doubles" ? "改成单打重跑" : "改成双打重跑";

  renderReview(v);
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
  if (have.has("technique.json")) {
    try {
      const doc = await (await fetch(resultUrl(v, "technique.json"))).json();
      if (openId !== v.id || key !== shownKey) return;
      const card = techniqueCard(doc, names);
      if (card) stats.append(card);
    } catch {}
  }
  if (have.has("points.json")) {
    try {
      const doc = await (await fetch(resultUrl(v, "points.json"))).json();
      if (openId !== v.id || key !== shownKey) return;
      await showPoints(v, doc, names, stats);
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
        const res = await fetch(resultUrl(v, r.name), { credentials: "include", headers: authHeaders() });
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

// ---------- 回合和得失分（badminton points 的 points.json） ----------
// 程序按「死球后谁在哪捡球」判谁赢，只有看到捡球的回合才计分。超级管理员可以在每个回合点谁赢
// （或「不算」），点了马上存进 rally_labels、这里当场重算；Mac 节点稍后同步，重算 points.json 和 AI 报告的摘要。

const SIDE_ZH = { near: "近端", far: "远端" };
const COUNTED = new Set(["high", "manual"]);

// 和 points.apply_labels 一样：标注对到最后一拍最近的回合（1.5 秒以内）
function applyRallyLabels(doc, labels) {
  const fps = doc.fps || 30;
  const rows = (doc.rallies || []).map((r) => {
    const base = r.program || r;
    return { ...r, winner_side: base.winner_side, confidence: base.confidence, dead_ball: base.dead_ball, label: null };
  });
  for (const [frame, lab] of Object.entries(labels || {})) {
    let best = null;
    for (const r of rows) if (!best || Math.abs(r.last_hit.frame - frame) < Math.abs(best.last_hit.frame - frame)) best = r;
    if (!best || Math.abs(best.last_hit.frame - frame) > 1.5 * fps) continue;
    best.label = { frame: Number(frame), winner: lab.winner };
    best.confidence = "manual";
    best.dead_ball = lab.winner === "none";
    best.winner_side = lab.winner === "none" ? null : lab.winner;
  }
  let n = 0;
  for (const r of rows) {
    const side = r.last_hit.side;
    r.n = r.dead_ball ? 0 : ++n;
    r.ending = null;
    r.error = null;
    if (r.dead_ball || !r.winner_side || !side) continue;
    if (r.winner_side === side) r.ending = "winner";
    else {
      r.ending = "error";
      // 没过网 / 出界 只在球的位置和谁丢分对得上时才有，沿用程序算的
      const orig = (doc.rallies || []).find((x) => x.last_hit.frame === r.last_hit.frame);
      const how = orig && orig.error && orig.winner_side === r.winner_side ? orig.error.how : null;
      r.error = { kind: r.error_kind_if_lost || (orig && orig.error && orig.error.kind) || "unforced", how };
    }
  }
  const players = (doc.players || []).map((p) => ({ ...p, winners: 0, errors: 0, unforced_errors: 0, forced_errors: 0 }));
  const sides = { near: { points: 0, winners: 0, errors: 0 }, far: { points: 0, winners: 0, errors: 0 } };
  for (const r of rows) {
    if (r.dead_ball || !r.winner_side || !COUNTED.has(r.confidence)) continue;
    sides[r.winner_side].points += 1;
    sides[r.winner_side][r.ending === "winner" ? "winners" : "errors"] += r.ending ? 1 : 0;
    const p = players.find((x) => x.id === r.last_hit.player_id);
    if (!p || !r.ending) continue;
    if (r.ending === "winner") p.winners += 1;
    else { p.errors += 1; p[r.error.kind + "_errors"] += 1; }
  }
  const real = rows.filter((r) => !r.dead_ball);
  return {
    rows, players, sides, total: real.length,
    judged: real.filter((r) => r.winner_side && COUNTED.has(r.confidence)).length,
    labelled: rows.filter((r) => r.label).length,
  };
}

function pointsCard(doc, names, labels = {}, onLabel = null) {
  if (!(doc.players || []).length || !(doc.rallies || []).length) return null;
  const d = applyRallyLabels(doc, labels);
  const wrap = el("div", "tech points");
  wrap.append(el("h3", "", "回合和得失分"));
  const who = (p) => p.name || names[p.id] || p.id + " 号";
  const sideName = (side) => {
    const ps = d.players.filter((p) => p.side === side).map(who);
    return SIDE_ZH[side] + (ps.length ? `（${ps.join("、")}）` : "");
  };
  wrap.append(el("p", "small",
    `${d.total} 个回合，判出谁赢的 ${d.judged} 个` + (d.labelled ? `（${d.labelled} 个是你标的）` : "") +
    (d.judged ? `：${sideName("near")} ${d.sides.near.points} 分，${sideName("far")} ${d.sides.far.points} 分` : "") +
    "。程序是看死球后谁弯腰捡球、在哪捡推出来的，没看到捡球的回合不判" +
    (onLabel ? "；下面每个回合可以自己点谁赢。" : "。") + "击球类型是模型认的，每回合第一拍按规则算发球" +
    (d.players.some((p) => Object.keys(p.strokes_guess || {}).length) ? "；「+n?」是模型没把握、按最可能的大类猜的次数。" : "。")));
  // 有把握的次数，后面「+n?」是没把握、按最可能的大类猜的次数（旧结果没有 strokes_guess）
  const sure = (p, g) => (p.strokes || {})[g] || 0;
  const guess = (p, g) => (p.strokes_guess || {})[g] || 0;
  const groups = [...new Set(d.players.flatMap((p) => [...Object.keys(p.strokes || {}), ...Object.keys(p.strokes_guess || {})]))]
    .sort((a, b) => d.players.reduce((n, p) => n + sure(p, b) + guess(p, b) - sure(p, a) - guess(p, a), 0));
  const hasGuess = d.players.some((p) => Object.keys(p.strokes_guess || {}).length);
  const table = document.createElement("table");
  const head = table.insertRow();
  head.append(el("th", "", ""));
  for (const p of d.players) head.append(el("th", "", who(p)));
  const line = (label, f, cls) => {
    const row = table.insertRow();
    row.insertCell().textContent = label;
    for (const p of d.players) {
      const c = row.insertCell();
      c.textContent = f(p);
      if (cls && f(p)) c.className = cls;
    }
  };
  line("击球", (p) => p.hits);
  for (const g of groups) line(g, (p) => (sure(p, g) || "") + (guess(p, g) ? ` +${guess(p, g)}?` : ""));
  line(hasGuess ? "其中没把握（+?）" : "类型没把握", (p) => p.hits_unsure || "");
  line("制胜球", (p) => p.winners || "");
  line("主动失误", (p) => p.unforced_errors || "", "warn");
  line("受迫失误", (p) => p.forced_errors || "");
  wrap.append(table);

  const det = document.createElement("details");
  det.open = Boolean(onLabel) && d.judged < d.total;
  det.append(el("summary", "", onLabel ? "每个回合：点谁赢（点时间跳到视频）" : "每个回合（点时间跳到视频）"));
  const ul = el("ul", "technotes rallies");
  const nameOf = (id) => { const p = d.players.find((x) => x.id === id); return p ? who(p) : id + " 号"; };
  for (const r of d.rows) {
    // 程序当成「把球打回去」的一两拍，不显示，除非有人标过
    if ((r.program || r).dead_ball && !r.label) continue;
    const li = el("li");
    const a = el("button", "linkish", fmtClock2(r.start_s));
    a.onclick = () => seekResultVideo(r.start_s + 1.5);
    let text;
    const last = r.last_hit;
    if (r.dead_ball) text = "不算回合";
    else if (r.winner_side && COUNTED.has(r.confidence)) {
      const how = r.ending === "winner" ? "制胜球"
        : r.ending === "error" ? (r.error.kind === "unforced" ? "主动失误" : "受迫失误") + (r.error.how ? `（${r.error.how}）` : "") : "";
      text = `${SIDE_ZH[r.winner_side]}得分：${nameOf(last.player_id)} ${last.needs_review ? "" : last.type + " "}${how}`;
    } else text = "没判出谁赢";
    const title = r.dead_ball ? " " : ` 回合 ${r.n} · ${r.hits} 拍：`;
    li.append(a, el("b", "", title), document.createTextNode(text));
    if (r.label) li.append(el("span", "by", " · 你标的"));
    else if (r.confidence === "high") li.append(el("span", "by", " · 程序看到捡球"));
    if (onLabel) {
      const pick = el("span", "winpick");
      const now = r.label ? r.label.winner : null;
      for (const [w, t] of [["near", "近端赢"], ["far", "远端赢"], ["none", "不算"]]) {
        const btn = el("button", "plain" + (now === w ? " on" : ""), t);
        btn.onclick = () => onLabel(r.label ? r.label.frame : last.frame, now === w ? null : w);
        pick.append(btn);
      }
      li.append(pick);
    }
    ul.append(li);
  }
  det.append(ul);
  wrap.append(det);
  return wrap;
}

// 在结果页上显示、并让超级管理员标谁赢
async function showPoints(v, doc, names, holder) {
  let labels = {};
  const editable = isOwner();
  if (editable) {
    try { labels = (await api(`/videos/${v.id}/rally-labels`)).labels || {}; } catch {}
  }
  let card = null;
  const draw = () => {
    const next = pointsCard(doc, names, labels, editable ? save : null);
    if (!next) return;
    const wasOpen = card && card.querySelector("details") && card.querySelector("details").open;
    if (card) {
      if (wasOpen) next.querySelector("details").open = true;
      card.replaceWith(next);
    } else holder.append(next);
    card = next;
  };
  async function save(frame, winner) {
    const before = labels;
    labels = { ...labels };
    if (winner) labels[frame] = { winner }; else delete labels[frame];
    draw();
    try {
      await api(`/videos/${v.id}/rally-labels`, {
        method: "PUT", headers: { "content-type": "application/json" },
        body: JSON.stringify({ items: [winner ? { frame, winner } : { frame, delete: true }] }),
      });
    } catch (e) {
      labels = before;
      draw();
      alert("没存上：" + e.message);
    }
  }
  draw();
}

// ---------- 击球动作：每拍和职业近端参考比（badminton technique 的 technique.json） ----------
// 只列提示用到的指标（noted_metrics）；手腕速度、转体、起跳等受机位影响，算了但不显示。

const TECH_LABEL = {
  contact_height: ["击球点", "", 2], stance: ["步幅", "", 2], knee_deg: ["膝角", "°", 0],
  trunk_lean: ["前倾", "°", 0], disguise: ["像杀球", "", 2],
};
const median = (xs) => {
  const a = xs.filter((x) => typeof x === "number" && isFinite(x)).sort((p, q) => p - q);
  if (!a.length) return null;
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
};
const fmtNum = (x, d) => (x == null ? "—" : x.toFixed(d));
const fmtClock2 = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function seekResultVideo(t) {
  setTab("videos");
  const vids = [...$("tVideos").querySelectorAll("video")];
  const vid = vids.find((x) => /combined\.mp4|skeleton\.mp4/.test(decodeURIComponent(x.src))) || vids[0];
  if (!vid) return;
  const go = () => { vid.currentTime = Math.max(0, t - 1.5); vid.play().catch(() => {}); };
  if (vid.readyState >= 1) go(); else vid.addEventListener("loadedmetadata", go, { once: true });
  vid.scrollIntoView({ behavior: "smooth", block: "center" });
}

function techniqueCard(doc, names) {
  const keys = (doc.noted_metrics || Object.keys(TECH_LABEL)).filter((k) => TECH_LABEL[k]);
  const strokes = (doc.strokes || []).filter((r) => r.ok);
  if (!strokes.length) return null;
  const ref = doc.reference_summary || {};
  const wrap = el("div", "tech");
  wrap.append(el("h3", "", "击球动作（和职业比）"));
  const players = new Map();
  for (const r of strokes) {
    if (!players.has(r.player_id)) players.set(r.player_id, []);
    players.get(r.player_id).push(r);
  }
  for (const [pid, rs] of players) {
    const who = rs[0].player_name || names[pid] || pid + " 号";
    wrap.append(el("h4", "", `${who} · ${rs.length} 拍`));
    const groups = new Map();
    for (const r of rs) {
      if (!groups.has(r.group)) groups.set(r.group, []);
      groups.get(r.group).push(r);
    }
    const table = document.createElement("table");
    const head = table.insertRow();
    for (const t of ["类型", "指标", "你", "职业", "偏离"]) head.append(el("th", "", t));
    for (const [g, gs] of groups) {
      let first = true;
      for (const k of keys) {
        const vals = gs.map((r) => r.metrics[k]).filter((x) => typeof x === "number");
        if (!vals.length) continue;
        // 参考里有这个类型却没有这项（例如推扑的击球点：推球和扑球混在一起），就不显示
        if (ref[g] && !ref[g][k]) continue;
        const [label, unit, d] = TECH_LABEL[k];
        const pro = ref[g] && ref[g][k] ? ref[g][k].p50 : null;
        const off = gs.filter((r) => r.percentiles && r.percentiles[k] != null && (r.percentiles[k] <= 10 || r.percentiles[k] >= 90)).length;
        const row = table.insertRow();
        row.insertCell().textContent = first ? `${g} ${gs.length}拍` : "";
        row.insertCell().textContent = label;
        row.insertCell().textContent = fmtNum(median(vals), d) + (unit === "°" ? "°" : "");
        row.insertCell().textContent = pro == null ? "—" : fmtNum(pro, d) + (unit === "°" ? "°" : "");
        const c = row.insertCell();
        c.textContent = off ? `${off}拍` : "";
        if (off) c.className = "warn";
        first = false;
      }
    }
    if (table.rows.length > 1) wrap.append(table);
    const noted = rs.filter((r) => r.notes && r.notes.length);
    if (noted.length) {
      const det = document.createElement("details");
      det.append(el("summary", "", `有提示的 ${noted.length} 拍（点时间跳到视频）`));
      const ul = el("ul", "technotes");
      for (const r of noted) {
        const li = el("li");
        const a = el("button", "linkish", fmtClock2(r.time_s));
        a.onclick = () => seekResultVideo(r.time_s);
        li.append(a, el("b", "", " " + r.type + "："), document.createTextNode(r.notes.join("；")));
        ul.append(li);
      }
      det.append(ul);
      wrap.append(det);
    }
  }
  const skipped = (doc.strokes || []).length - strokes.length;
  wrap.append(el("p", "quiet small",
    "只看近端球员，比的是中位数。击球点 = 手腕高出肩多少，步幅 = 两脚间距，都以躯干长（肩到髋）为单位；" +
    "像杀球 = 吊球的引拍和自己杀球的差别，1 左右说明看不出来。「偏离」= 低于职业第 10 或高于第 90 百分位的拍数。" +
    "职业参考：14 场职业单打转播的近端球员。" +
    "击球类型是模型认的，可能有错" + (skipped ? `；另有 ${skipped} 拍关节看不清或类型可能判错，没有算` : "") +
    "。手腕速度、转体、起跳、发力顺序受机位影响大，算了但没列出。"));
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

// ---------- 处理：重新处理、改单打双打、删除 ----------

$("dRetry").onclick = async () => {
  try {
    await api(`/videos/${openId}/job/retry`, { method: "POST" });
    log("已重新排队");
  } catch (e) { log("重新处理失败：" + e.message); }
  refreshList();
};
$("dCards").onclick = async () => {
  try {
    await api(`/videos/${openId}/job/retry`, {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ only: "cards" }),
    });
    log("已排队：只更新卡片，视频和 AI 点评不动");
  } catch (e) { log("更新卡片失败：" + e.message); }
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
document.addEventListener("visibilitychange", () => { if (!document.hidden && authed && isOwner() && !$("app").hidden) refreshList(); });

$("copyLog").onclick = async () => {
  try { await navigator.clipboard.writeText($("log").textContent); log("记录已复制"); }
  catch { log("复制失败，请手动选中"); }
};

tryLogin();
